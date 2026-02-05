#!/bin/bash
# v-hnsw Installer for Linux/macOS
# Usage: curl -fsSL https://raw.githubusercontent.com/LeeSangMin1029/v-hnsw/main/install.sh | bash

set -e

REPO="LeeSangMin1029/v-hnsw"
INSTALL_DIR="${HOME}/.local/bin"
CUDA=${CUDA:-false}
VERSION=${VERSION:-latest}

echo "v-hnsw Installer"
echo "================"

# Detect OS and architecture
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

case "$OS" in
    linux)
        TARGET="x86_64-unknown-linux-gnu"
        ;;
    darwin)
        if [ "$ARCH" = "arm64" ]; then
            TARGET="aarch64-apple-darwin"
        else
            TARGET="x86_64-apple-darwin"
        fi
        ;;
    *)
        echo "Unsupported OS: $OS"
        exit 1
        ;;
esac

SUFFIX=""
if [ "$CUDA" = "true" ] && [ "$OS" = "linux" ]; then
    SUFFIX="-cuda"
fi

# Get release URL
if [ "$VERSION" = "latest" ]; then
    echo "Fetching latest release..."
    RELEASE_URL="https://api.github.com/repos/$REPO/releases/latest"
else
    RELEASE_URL="https://api.github.com/repos/$REPO/releases/tags/$VERSION"
fi

# Get download URL
ASSET_NAME="v-hnsw-${TARGET}${SUFFIX}.tar.gz"
DOWNLOAD_URL=$(curl -s "$RELEASE_URL" | grep "browser_download_url.*$ASSET_NAME" | cut -d '"' -f 4)

if [ -z "$DOWNLOAD_URL" ]; then
    echo "Error: Could not find release asset: $ASSET_NAME"
    echo "Available assets:"
    curl -s "$RELEASE_URL" | grep "browser_download_url" | cut -d '"' -f 4
    exit 1
fi

VERSION_TAG=$(curl -s "$RELEASE_URL" | grep '"tag_name"' | cut -d '"' -f 4)
echo "Installing v-hnsw $VERSION_TAG ($TARGET$SUFFIX)..."

# Create install directory
mkdir -p "$INSTALL_DIR"

# Download and extract
echo "Downloading from $DOWNLOAD_URL..."
curl -fsSL "$DOWNLOAD_URL" | tar -xzf - -C "$INSTALL_DIR"

# Make executable
chmod +x "$INSTALL_DIR/v-hnsw"

# Check PATH
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    echo ""
    echo "Add the following to your shell profile (.bashrc, .zshrc, etc.):"
    echo ""
    echo "  export PATH=\"\$PATH:$INSTALL_DIR\""
    echo ""
fi

echo ""
echo "Installation complete!"
echo "Run 'v-hnsw --help' to get started."
