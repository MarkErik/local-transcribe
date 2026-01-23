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

# --- MFA Setup ---------------------------------------------------------------
echo ""
echo "=== Setting up Montreal Forced Aligner (MFA) ==="

# Get the directory where this script is located (project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MFA_ENV_DIR="$SCRIPT_DIR/.mfa_env"

echo "MFA environment will be created in: $MFA_ENV_DIR"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. MFA works best with conda."
    echo "   Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    echo "   Or continue without MFA (will use simple alignment fallback)"
    exit 1
fi

echo "✓ Conda found"

# Check if MFA environment already exists
if [ -d "$MFA_ENV_DIR" ]; then
    echo "✓ MFA environment already exists at $MFA_ENV_DIR"
    echo "   To recreate, remove the directory first: rm -rf $MFA_ENV_DIR"
    echo "   To update, activate and run: conda update -p $MFA_ENV_DIR montreal-forced-aligner"
else
    # Create MFA environment in project directory
    echo "⬇️  Creating MFA environment in project directory..."
    conda create -p "$MFA_ENV_DIR" -c conda-forge montreal-forced-aligner -y
fi

# Activate the environment and verify installation
echo "✓ Activating MFA environment..."
# Try different activation methods
if command -v conda &> /dev/null && conda activate "$MFA_ENV_DIR" 2>/dev/null; then
    echo "✓ Environment activated"
elif source "$MFA_ENV_DIR/bin/activate" 2>/dev/null; then
    echo "✓ Environment activated (source method)"
else
    echo "⚠️  Could not activate environment automatically"
    echo "   You may need to activate it manually: conda activate $MFA_ENV_DIR"
fi

if command -v mfa &> /dev/null; then
    echo "✓ MFA installed successfully"
    mfa version
else
    echo "❌ MFA installation failed or not activated"
    echo "   Try activating manually: conda activate $MFA_ENV_DIR"
    exit 1
fi

# Deactivate environment if it was activated
conda deactivate 2>/dev/null || true

echo ""
echo "=== Setup Complete ==="
echo ""
echo "MFA environment created at: $MFA_ENV_DIR"
echo "MFA models will be downloaded automatically on first use to: .models/aligners/mfa/"
echo "To test: python main.py -i audio.m4a --transcriber-provider openai_whisper --aligner-provider mfa -o test-output/"
echo ""
echo "Note: The MFA environment is self-contained within this project directory."
echo "      No system-wide conda installation is required."
