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

# --- project init ------------------------------------------------------------
if [[ ! -f "pyproject.toml" ]]; then
  echo "📦 Initializing uv project…"
  uv init --name local-transcribe
else
  echo "📦 Using existing pyproject.toml"
fi

# Ensure [project] requires-python is 3.12-compatible BEFORE pinning
if grep -qE '^\s*requires-python\s*=' pyproject.toml; then
  echo "🛠  Adjusting requires-python to 3.12 in pyproject.toml…"
  "${SED_INPLACE[@]}" 's/requires-python *= *".*"/requires-python = ">=3.12,<3.13"/' pyproject.toml
else
  echo "🛠  Adding requires-python to [project] in pyproject.toml…"
  # Insert under [project]; if not present, create it at top
  if ! grep -q '^\[project\]' pyproject.toml; then
    printf "[project]\n" | cat - pyproject.toml > pyproject.tmp && mv pyproject.tmp pyproject.toml
  fi
  awk '
    BEGIN{inserted=0}
    /^\[project\]/{print; print "requires-python = \">=3.12,<3.13\""; inserted=1; next}
    {print}
    END{if(!inserted)print "requires-python = \">=3.12,<3.13\""}
  ' pyproject.toml > pyproject.tmp && mv pyproject.tmp pyproject.toml
fi

echo "📌 Pinning Python version to 3.12…"
uv python pin 3.12

# --- dependencies ------------------------------------------------------------
echo "⬇️  Adding runtime dependencies (this may take a minute)…"
uv add torch torchaudio
uv add whisperx faster-whisper pyannote.audio ffmpeg-python pydub \
       numpy soundfile librosa rich tqdm pandas

echo "⬇️  Adding dev/notebook tools…"
uv add --dev jupyterlab ipykernel ipywidgets

# --- directory skeleton ------------------------------------------------------
echo "📁 Creating project folders…"
mkdir -p models/asr models/align models/diarization
mkdir -p data/input data/output
mkdir -p notebooks
mkdir -p src
mkdir -p scripts

# --- .gitignore --------------------------------------------------------------
if [[ ! -f ".gitignore" ]]; then
  echo "📝 Creating .gitignore…"
  cat > .gitignore <<'EOF'
# local-transcribe ignores
models/
data/output/
.env
.uv/
**/__pycache__/
.ipynb_checkpoints/
.DS_Store
EOF
else
  echo "📝 .gitignore already exists (leaving as-is)"
fi

# --- .env.example ------------------------------------------------------------
if [[ ! -f ".env.example" ]]; then
  echo "📝 Creating .env.example…"
  cat > .env.example <<'EOF'
# Populate this then copy to .env
# Your HF token is used only by scripts/download_models.py (one-time).
HUGGINGFACE_TOKEN=

# Force all caches to live inside the repo (keeps everything local & offline)
HF_HOME=./models
TRANSFORMERS_CACHE=./models
PYANNOTE_CACHE=./models/diarization
XDG_CACHE_HOME=./models/.xdg

# Enforce offline behavior at runtime (recommended once models are downloaded)
HF_HUB_OFFLINE=1
EOF
else
  echo "📝 .env.example already exists (leaving as-is)"
fi

# Create .env if missing (from example)
if [[ ! -f ".env" ]] && [[ -f ".env.example" ]]; then
  cp .env.example .env
  echo "✅ Created .env from .env.example (fill in HUGGINGFACE_TOKEN before downloading models)"
fi

# --- environment notes -------------------------------------------------------
echo "🔍 Checking for ffmpeg…"
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "⚠️  'ffmpeg' not found on PATH."
  echo "   Install via Homebrew:  brew install ffmpeg"
else
  echo "✅ ffmpeg found: $(command -v ffmpeg)"
fi

echo
echo "=== ✅ Setup complete ==="
echo "Next steps:"
echo "  1) Start Jupyter:            uv run jupyter lab"
echo "  2) Download models (one-time):"
echo "       uv run python scripts/download_models.py"
echo "     This fetches BOTH ASR models (medium.en, large-v3-turbo),"
echo "     WhisperX English aligners, and pyannote diarization models into ./models/."
echo
echo "After that, notebooks/Transcribe.ipynb will run fully offline on Python 3.12."

