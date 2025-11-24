# scripts/setup_project.sh
#!/usr/bin/env bash
set -euo pipefail

echo "=== 🛠  Setting up local-transcribe ==="

# --- sanity checks -----------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "❌ 'uv' is not installed."
  echo "   Install with:  brew install uv"
  echo "   Docs: https://docs.astral.sh/uv/"
  exit 1
fi

UNAME_OUT="$(uname -s || true)"
if [[ "${UNAME_OUT}" != "Darwin" ]]; then
  echo "⚠️  This script is tailored for macOS (Apple Silicon). Continuing anyway…"
fi

# Choose macOS-compatible in-place sed
SED_INPLACE=(sed -i '')
if [[ "${UNAME_OUT}" != "Darwin" ]]; then
  SED_INPLACE=(sed -i)
fi

# --- dependencies ------------------------------------------------------------
echo "⬇️  Adding runtime dependencies (this may take a minute)…"
uv sync

# --- environment notes -------------------------------------------------------
echo "🔍 Checking for ffmpeg…"
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "⚠️  'ffmpeg' not found on PATH."
  echo "   Install via Homebrew:  brew install ffmpeg"
else
  echo "✅ ffmpeg found: $(command -v ffmpeg)"
fi

echo "🔍 Checking for whisper.cpp…"
if ! command -v whisper-cli >/dev/null 2>&1; then
  echo "⚠️  'whisper-cli' not found on PATH."
  echo "   Install via Homebrew:  brew install whisper-cpp"
else
  echo "✅ whisper-cli found: $(command -v whisper-cli)"
fi
