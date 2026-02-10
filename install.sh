#!/usr/bin/env bash
# v-hnsw installer for Linux / macOS
# Tries to download pre-built binary from GitHub Release.
# Falls back to building from source if no binary is available.

set -euo pipefail

REPO="LeeSangMin1029/hnsw-model2vec"
INSTALL_DIR="${HOME}/.local/bin"

echo "=== v-hnsw installer ==="

# Detect platform
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux)  TARGET_OS="unknown-linux-gnu" ;;
    Darwin) TARGET_OS="apple-darwin" ;;
    *)      echo "ERROR: Unsupported OS: $OS"; exit 1 ;;
esac

case "$ARCH" in
    x86_64)        TARGET_ARCH="x86_64" ;;
    aarch64|arm64) TARGET_ARCH="aarch64" ;;
    *)             echo "ERROR: Unsupported architecture: $ARCH"; exit 1 ;;
esac

TARGET="${TARGET_ARCH}-${TARGET_OS}"
ASSET_NAME="v-hnsw-${TARGET}.tar.gz"
echo "Platform: $TARGET"

# --- Try download from GitHub Release via gh CLI ---
download_with_gh() {
    if ! command -v gh &>/dev/null; then
        return 1
    fi

    local tag
    tag=$(gh release list -R "$REPO" --limit 1 --json tagName -q '.[0].tagName' 2>/dev/null || true)
    if [[ -z "$tag" ]]; then
        return 1
    fi

    echo "Found release: $tag (via gh CLI)"
    local tmpdir
    tmpdir=$(mktemp -d)
    if gh release download "$tag" -R "$REPO" -p "$ASSET_NAME" -D "$tmpdir" 2>/dev/null; then
        tar -xzf "$tmpdir/$ASSET_NAME" -C "$tmpdir"
        mkdir -p "$INSTALL_DIR"
        mv "$tmpdir/v-hnsw" "$INSTALL_DIR/v-hnsw"
        chmod +x "$INSTALL_DIR/v-hnsw"
        rm -rf "$tmpdir"
        return 0
    fi

    rm -rf "$tmpdir"
    return 1
}

# --- Build from source ---
build_from_source() {
    echo ""
    echo "Building from source..."

    if ! command -v rustc &>/dev/null; then
        echo "Rust not found. Installing via rustup..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
        echo "Rust installed: $(rustc --version)"
    else
        echo "Rust found: $(rustc --version)"
    fi

    echo ""
    echo "Building v-hnsw (release mode, this may take a few minutes)..."
    cargo install --path crates/v-hnsw-cli
}

# --- Main ---
if download_with_gh; then
    echo "Downloaded pre-built binary."
else
    echo "No pre-built binary available. Building from source..."
    build_from_source
fi

# Verify
echo ""
echo "=== Installation complete ==="

if [[ -x "$INSTALL_DIR/v-hnsw" ]]; then
    echo "Binary: $INSTALL_DIR/v-hnsw"
    if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
        echo ""
        echo "Add to PATH (add to your shell rc file):"
        echo "  export PATH=\"$INSTALL_DIR:\$PATH\""
    fi
elif command -v v-hnsw &>/dev/null; then
    echo "Binary: $(which v-hnsw)"
fi

echo ""
echo "First run will auto-download:"
echo "  - Embedding model (~500MB, minishlab/potion-multilingual-128M)"
echo "  - Korean dictionary (ko-dic)"
echo ""
echo "Run 'v-hnsw --help' to get started."
