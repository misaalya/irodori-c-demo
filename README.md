# Irodori C demo

A minimal web UI for the [Irodori C](https://github.com/misaalya/irodori-c)
inference engine (Japanese text-to-speech on CPU). Python standard library
only — no framework, no build step. You do not need the engine's source code:
`setup-engine.sh` fetches the prebuilt engine release into this directory.

![screenshot](screenshot.png)

## Quick start

Requirements: Linux x86-64 with AVX2 (any CPU from ~2013 on), glibc ≥ 2.35
(Ubuntu / Pop!_OS 22.04 or newer), `git`, `curl`, `python3`, ≥ 4 GB RAM and
≈ 4 GB of disk. Windows users: run the same steps inside WSL2 (Ubuntu 22.04).

```sh
git clone https://github.com/misaalya/irodori-c-demo.git
cd irodori-c-demo
./setup-engine.sh      # engine release + tokenizer/codec + model (~3.5 GB, once)
python3 server.py      # open http://127.0.0.1:8080
```

`setup-engine.sh` downloads the engine tarballs from the
[irodori-c releases](https://github.com/misaalya/irodori-c/releases), verifies
their checksums, unpacks them into `./engine`, and fetches the FP32 checkpoint
into `./engine/weights/`. For the anime fine-tune use
`./setup-engine.sh phasefield-audio/Irodori-TTS-v4.1-Anime`. Everything it
creates (`downloads/`, `engine/`, `outputs/`) is gitignored.

The int8 "fast" options in the UI need a CPU with AVX-512 VNNI (Intel 10th-gen
Ice Lake or newer, AMD Zen 4 or newer). On other CPUs pick **fp32** in the UI,
or start the OpenBLAS build: `python3 server.py --binary engine/bin/irodori-blas`.

Other options: `--threads N` (physical cores, default 2), `--port`,
`--host 0.0.0.0` (LAN access; there is no authentication), `--model` (any
checkpoint with the v4.1-Small architecture), `--binary`, `--weights`,
`--outputs`. Missing pieces are reported at startup with what to do.

## Using the page

- **Text** (Japanese), optional **caption** (voice design), optional
  **reference WAV** (voice cloning; clean single-speaker recording).
- **Steps** 8 for quick previews, 40 for final quality; **seed** for
  reproducibility.
- **DiT / codec precision**: `int8` is 2–4× faster; `fp32` reproduces the
  PyTorch reference.
- The result shows the player, stage timings (encode / sampling / decode),
  engine total with RTF, wall time and the engine log.

## Developers: running against a source build

This repository is also the `demo/` submodule of the engine. With the engine
built from source and `weights/` prepared (see the engine README),
`python3 demo/server.py` from the engine directory picks up the binary and
weights automatically; `./engine` (if present) takes precedence.

## API

- `GET /api/info` — engine binary name, model name, thread count, encoder
  availability (no filesystem paths are exposed).
- `POST /api/generate` — JSON `{text, caption?, reference_wav_base64?, steps,
  seed, dit_precision, codec_precision}` → `{wav_url, audio_seconds,
  encode_seconds, sample_seconds, decode_seconds, total_seconds, rtf,
  wall_seconds, log}`.
- `GET /outputs/<job>.wav` — generated audio (byte ranges supported).

## License and credits

MIT (see [LICENSE](LICENSE)). Engine base implementation by OpenAI Codex;
performance refinement, int8 paths and this demo by Claude. Model:
[Irodori-TTS](https://github.com/Aratako/Irodori-TTS) by Chihiro Arata (MIT,
with the authors' ethical restrictions on voice cloning). The engine release
bundles Intel oneMKL / OpenMP (Intel Simplified Software License) and
OpenBLAS (BSD-3).
