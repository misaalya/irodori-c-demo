# Irodori C demo

A minimal web UI for the [Irodori C](https://github.com/misaalya/irodori-c)
inference engine. Python standard library only — no framework, no build step.
The page posts JSON to a tiny HTTP server which runs the engine binary as a
subprocess (one request at a time) and returns the WAV together with the
engine's timing line.

![screenshot](screenshot.png)

## Run

Assumptions: this repo is checked out as `demo/` inside the engine directory,
the engine is built (`make blas` or `make irodori-onemkl MKL_ROOT=…`), and
`weights/` holds `tokenizer.bin`, `dacvae_decoder.safetensors` and, for
cloning, `dacvae_encoder.safetensors` (see the engine README). Then:

```sh
python3 demo/server.py            # from the engine directory, or `python3 server.py` from demo/
# open http://127.0.0.1:8080
```

Without arguments the server picks the first of `irodori-onemkl`,
`irodori-blas`, `irodori-mkl` found next to `demo/`, uses `<engine>/weights`,
and looks for the model at `weights/model.safetensors`, then `$IRO_MODEL`.
Relative paths are resolved against the current directory first and the
engine directory second. Options: `--binary`, `--weights`, `--model` (any
compatible checkpoint, e.g. Irodori-TTS-v4.1-Anime), `--threads`, `--host`,
`--port`, `--outputs`. Missing pieces are reported with what to do.

The int8 options in the UI require an oneMKL build (`make irodori-onemkl
MKL_ROOT=…`) on a CPU with AVX-512 VNNI; with `irodori-blas` choose fp32.

## API

- `GET /api/info` — engine binary, model, thread count, encoder availability.
- `POST /api/generate` — JSON `{text, caption?, reference_wav_base64?, steps,
  seed, dit_precision, codec_precision}` → `{wav_url, audio_seconds,
  encode_seconds, sample_seconds, decode_seconds, total_seconds, rtf,
  wall_seconds, log}`.
- `GET /outputs/<job>.wav` — generated audio.

Generated files are kept in `outputs/` (gitignored). The server is meant for
local use; it has no authentication.

## Credits

Engine base implementation by OpenAI Codex; performance refinement, int8
paths and this demo by Claude. Model: [Irodori-TTS](https://github.com/Aratako/Irodori-TTS)
by Chihiro Arata (MIT, with the authors' ethical restrictions on voice cloning).
