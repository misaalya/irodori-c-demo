#!/usr/bin/env python3
"""Minimal web UI for the Irodori C engine.

Standard library only: serves static/index.html and a JSON API that runs the
engine binary as a subprocess (one request at a time) and returns the WAV
plus the timing line the CLI prints.

    python3 demo/server.py                    # from the engine directory, auto-detects binary/weights
    python3 server.py --model /path/to/model.safetensors --binary ../irodori-blas
    python3 server.py --help
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
TIMING_RE = re.compile(
    r"encode ([\d.]+) s \| Euler ([\d.]+) s \| decode ([\d.]+) s \| total ([\d.]+) s \| RTF ([\d.]+)"
)
GENERATED_RE = re.compile(r"-> (\d+) latent frame -> ([\d.]+) s audio")


ENGINE_DIR = ROOT.parent  # demo/ is checked out inside the engine directory
BINARY_CANDIDATES = ("irodori-onemkl", "irodori-blas", "irodori-mkl")


def locate(value: str | None, default_name: str | None, must_exist: bool = True) -> Path | None:
    """Resolve a path relative to the current directory, then to the engine
    directory, so the server works from either location."""
    if value:
        for base in (Path.cwd(), ENGINE_DIR):
            candidate = (base / value).resolve()
            if candidate.exists():
                return candidate
        return Path(value).resolve() if not must_exist else None
    if default_name:
        candidate = (ENGINE_DIR / default_name).resolve()
        return candidate if candidate.exists() or not must_exist else None
    return None


def is_model_file(path: Path | None) -> bool:
    # HF-cache entries are symlinks to hash-named blobs, so only require a file.
    return bool(path) and path.is_file()


class Engine:
    """Serialized access to the CLI binary."""

    def __init__(self, args: argparse.Namespace) -> None:
        problems = []
        binary = locate(args.binary, None)
        if binary is None and not args.binary:
            for name in BINARY_CANDIDATES:
                binary = locate(None, name)
                if binary:
                    break
        if binary is None:
            problems.append(f"engine binary not found ({args.binary or ', '.join(BINARY_CANDIDATES)}); "
                            f"build it first, e.g. `make irodori-onemkl MKL_ROOT=...` or `make blas` in {ENGINE_DIR}")
        weights = locate(args.weights, "weights", must_exist=False) or ENGINE_DIR / "weights"
        model = None
        tried = []
        for label, candidate in (("--model", Path(args.model) if args.model else None),
                                 ("weights/", weights / "model.safetensors"),
                                 ("$IRO_MODEL", Path(os.environ["IRO_MODEL"]) if os.environ.get("IRO_MODEL") else None)):
            if candidate is None:
                continue
            resolved = locate(str(candidate), None, must_exist=False)
            if is_model_file(resolved):
                model = resolved
                break
            tried.append(f"{label}: {candidate} ({'is a directory' if resolved and resolved.is_dir() else 'not found'})")
        if model is None:
            problems.append("no usable model.safetensors (" + "; ".join(tried) +
                            "); put it in weights/, set IRO_MODEL to the file, or pass --model /path/to/model.safetensors")
        if binary is not None and not (binary.is_file() and os.access(binary, os.X_OK)):
            problems.append(f"{binary} is not an executable file")
        self.tokenizer = weights / "tokenizer.bin"
        self.decoder = weights / "dacvae_decoder.safetensors"
        self.encoder = weights / "dacvae_encoder.safetensors"
        for path, hint in ((self.tokenizer, "tools/compile_tokenizer.py"), (self.decoder, "tools/export_dacvae_decoder.py")):
            if not path.exists():
                problems.append(f"{path} missing (create it with {hint}, see the engine README)")
        if problems:
            raise SystemExit("cannot start:\n  - " + "\n  - ".join(problems))
        self.binary, self.model = binary, model
        self.threads = args.threads
        self.outputs = locate(args.outputs, None, must_exist=False) or (Path.cwd() / args.outputs).resolve()
        self.outputs.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        if not self.encoder.exists():
            print("note: dacvae_encoder.safetensors missing, voice cloning disabled", flush=True)

    def info(self) -> dict:
        # Names only: the page must not expose local filesystem paths.
        model_name = self.model.name
        if not model_name.endswith(".safetensors"):  # e.g. a resolved HF-cache blob
            model_name = "model.safetensors"
        return {
            "binary": self.binary.name,
            "model": model_name,
            "encoder_available": self.encoder.exists(),
            "threads": self.threads,
        }

    def generate(self, req: dict) -> dict:
        text = (req.get("text") or "").strip()
        if not text:
            raise ValueError("text is required")
        steps = int(req.get("steps") or 8)
        seed = int(req.get("seed") or 42)
        if not 1 <= steps <= 200:
            raise ValueError("steps must be 1..200")
        dit = req.get("dit_precision") or "fp32"
        codec = req.get("codec_precision") or "fp32"
        if dit not in ("fp32", "int8") or codec not in ("fp32", "int8"):
            raise ValueError("precision must be fp32 or int8")
        caption = (req.get("caption") or "").strip()
        job = uuid.uuid4().hex[:12]
        out = self.outputs / f"{job}.wav"
        ref_path = None
        if req.get("reference_wav_base64"):
            if not self.encoder.exists():
                raise ValueError("voice cloning needs dacvae_encoder.safetensors")
            ref_path = self.outputs / f"{job}-ref.wav"
            ref_path.write_bytes(base64.b64decode(req["reference_wav_base64"]))
        cmd = [str(self.binary), "--text", text,
               "--model", str(self.model), "--tokenizer", str(self.tokenizer),
               "--decoder", str(self.decoder), "--steps", str(steps), "--seed", str(seed),
               "--dit-precision", dit, "--codec-precision", codec, "--out", str(out)]
        if caption:
            cmd += ["--caption", caption]
        if ref_path:
            cmd += ["--encoder", str(self.encoder), "--ref", str(ref_path)]
        env = dict(os.environ, IRO_NUM_THREADS=str(self.threads))
        with self.lock:
            t0 = time.monotonic()
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
            wall = time.monotonic() - t0
        if ref_path:
            ref_path.unlink(missing_ok=True)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "engine failed")
        result = {"job": job, "wav_url": f"/outputs/{job}.wav", "wall_seconds": round(wall, 3),
                  "dit_precision": dit, "codec_precision": codec, "steps": steps, "seed": seed,
                  "log": proc.stdout.strip().replace(str(self.outputs), "outputs")}
        m = TIMING_RE.search(proc.stdout)
        if m:
            result.update({"encode_seconds": float(m[1]), "sample_seconds": float(m[2]),
                           "decode_seconds": float(m[3]), "total_seconds": float(m[4]), "rtf": float(m[5])})
        g = GENERATED_RE.search(proc.stdout)
        if g:
            result.update({"latent_frames": int(g[1]), "audio_seconds": float(g[2])})
        return result


class Handler(BaseHTTPRequestHandler):
    engine: Engine

    def log_message(self, fmt, *args):  # quieter log line
        print(f"{self.address_string()} {fmt % args}", flush=True)

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _send_wav(self, path: Path, head: bool = False) -> None:
        """Serve a WAV with byte-range support so browser players can seek."""
        data = path.read_bytes()
        total = len(data)
        start, end = 0, total - 1
        status = 200
        rng = self.headers.get("Range")
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng or "")
        if m and (m[1] or m[2]):
            if m[1]:
                start = int(m[1]); end = int(m[2]) if m[2] else total - 1
            else:
                start = max(total - int(m[2]), 0)
            end = min(end, total - 1)
            if start > end:
                self.send_response(416); self.send_header("Content-Range", f"bytes */{total}"); self.end_headers(); return
            status = 206
        body = data[start:end + 1]
        self.send_response(status)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(body)))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _route(self, head: bool = False) -> None:
        route = urlsplit(self.path).path
        if route in ("/", "/index.html"):
            self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif route == "/api/info":
            self._json(200, self.engine.info())
        elif route.startswith("/outputs/"):
            name = Path(route).name
            path = self.engine.outputs / name
            if re.fullmatch(r"[0-9a-f]{12}\.wav", name) and path.is_file():
                self._send_wav(path, head)
            else:
                self._json(404, {"error": "not found"})
        else:
            self._json(404, {"error": "not found"})

    def do_GET(self) -> None:
        self._route()

    def do_HEAD(self) -> None:
        self._route(head=True)

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/generate":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length > 64 * 1024 * 1024:
            self._json(413, {"error": "request too large"})
            return
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
            self._json(200, self.engine.generate(req))
        except (ValueError, KeyError) as e:
            self._json(400, {"error": str(e)})
        except Exception as e:  # engine failure: log details, return a generic message
            print(f"engine error: {e}", flush=True)
            self._json(500, {"error": "engine failed; see the server log"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default=None, help="engine executable; default: first of irodori-onemkl, irodori-blas, irodori-mkl in the engine directory")
    ap.add_argument("--weights", default=None, help="directory with tokenizer.bin and codec safetensors (default: <engine>/weights)")
    ap.add_argument("--model", default=None, help="model.safetensors (default: <weights>/model.safetensors, then $IRO_MODEL)")
    ap.add_argument("--threads", type=int, default=2, help="IRO_NUM_THREADS for the engine")
    ap.add_argument("--outputs", default="outputs", help="where generated WAVs are stored")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    Handler.engine = Engine(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Irodori C demo: http://{args.host}:{args.port}  engine={Handler.engine.binary.name}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
