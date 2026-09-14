#!/usr/bin/env bash
# Fetch the prebuilt Irodori C engine release, its assets and the model into
# ./engine so the demo runs from a bare clone of this repository.
#   ./setup-engine.sh                                   # v0.2.0, base model
#   ./setup-engine.sh phasefield-audio/Irodori-TTS-v4.1-Anime
#   RELEASE=v0.2.0 ./setup-engine.sh
set -euo pipefail
cd "$(dirname "$0")"
RELEASE=${RELEASE:-v0.2.0}
MODEL_REPO=${1:-Aratako/Irodori-TTS-v4.1-Small}
BASE="https://github.com/misaalya/irodori-c/releases/download/$RELEASE"
ENGINE="irodori-c-$RELEASE-linux-x86_64"
mkdir -p downloads
for f in "$ENGINE.tar.gz" "irodori-c-assets-$RELEASE.tar.gz" SHA256SUMS; do
  [ -f "downloads/$f" ] || curl -L --fail --progress-bar -o "downloads/$f" "$BASE/$f"
done
(cd downloads && sha256sum -c --ignore-missing SHA256SUMS)
rm -rf engine && mkdir engine
tar -C engine --strip-components=1 -xzf "downloads/$ENGINE.tar.gz"
tar -C engine --strip-components=1 -xzf "downloads/irodori-c-assets-$RELEASE.tar.gz"
if [ ! -f engine/weights/model.safetensors ]; then
  echo "Downloading model $MODEL_REPO (~3 GB)"
  curl -L --fail --progress-bar -C - -o engine/weights/model.safetensors \
    "https://huggingface.co/$MODEL_REPO/resolve/main/model.safetensors"
fi
echo "Engine ready in ./engine. Start the UI with: python3 server.py"
