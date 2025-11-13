#!/usr/bin/env bash
# Setup script for MFA integration

set -e

echo "=== Setting up Montreal Forced Aligner (MFA) ==="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. MFA works best with conda."
    echo "   Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    echo "   Or continue without MFA (will use simple alignment fallback)"
    exit 1
fi

echo "✓ Conda found"

# Install MFA via conda
echo "⬇️  Installing Montreal Forced Aligner..."
conda install -c conda-forge montreal-forced-aligner -y

# Verify installation
if command -v mfa &> /dev/null; then
    echo "✓ MFA installed successfully"
    mfa version
else
    echo "❌ MFA installation failed"
    exit 1
fi

# Install Python dependencies
echo "⬇️  Installing Python dependencies..."
if command -v uv &> /dev/null; then
    echo "Using uv..."
    uv sync
else
    echo "Using pip..."
    pip install textgrid
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "MFA models will be downloaded automatically on first use."
echo "To test: python main.py -i audio.m4a --transcriber-provider granite --aligner-provider mfa -o test-output/"
echo ""
