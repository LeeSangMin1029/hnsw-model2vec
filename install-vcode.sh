#!/bin/bash
set -e
echo "[v-code] Installing..."

# Detect OS
OS=$(uname -s)
case "$OS" in
    MINGW*|MSYS*|CYGWIN*) EXT=".exe" ;;
    *) EXT="" ;;
esac

# Download
echo "[v-code] Downloading v-code${EXT}..."
curl -sLO "https://github.com/LeeSangMin1029/hnsw-model2vec/releases/latest/download/v-code${EXT}"
chmod +x "v-code${EXT}" 2>/dev/null || true

# Install to PATH
DEST="${HOME}/.cargo/bin/v-code${EXT}"
mkdir -p "$(dirname "$DEST")"
mv "v-code${EXT}" "$DEST"
echo "[v-code] Installed to $DEST"

# Install nightly
echo "[v-code] Installing nightly rustc..."
if command -v rustup &>/dev/null; then
    rustup toolchain install nightly
    rustup component add rust-src rustc-dev llvm-tools-preview --toolchain nightly
else
    echo "[v-code] rustup not found. Install Rust first: https://rustup.rs"
    exit 1
fi

echo "[v-code] Done! Run: v-code add .code.db ."
