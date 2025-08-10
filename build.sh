#!/bin/bash
set -e

# Detect current platform
PLATFORM=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    ARCH="amd64"
elif [ "$ARCH" = "aarch64" ]; then
    ARCH="arm64"
fi
PLATFORM_DIR="${PLATFORM}_${ARCH}"

echo "Building for platform: $PLATFORM_DIR"

# Verify dependencies
if [ ! -d ".git" ]; then
    echo "Error: This script must be run from a git repository root"
    exit 1
fi

if [ ! -d "hnswlib" ] || [ ! -f "hnswlib/hnswlib.h" ]; then
    echo "Error: hnswlib headers not found"
    exit 1
fi

if ! command -v go &> /dev/null; then
    echo "Error: Go is not installed or not in PATH"
    exit 1
fi

if ! command -v clang++ &> /dev/null && ! command -v g++ &> /dev/null; then
    echo "Error: No C++ compiler found (clang++ or g++)"
    exit 1
fi

if ! command -v c-for-go &> /dev/null; then
    echo "Installing c-for-go..."
    go install github.com/xlab/c-for-go@latest
    if ! command -v c-for-go &> /dev/null; then
        echo "Error: Failed to install c-for-go"
        exit 1
    fi
fi

# Build process
echo "Cleaning previous builds..."
make clean 2>/dev/null || true
rm -rf bindings/bindings

echo "Building C++ wrapper library..."
make

if [ ! -f "build/$PLATFORM_DIR/libhnsw_wrapper.a" ]; then
    echo "Error: Failed to build static library"
    exit 1
fi

echo "Generating Go bindings..."
go generate ./...

if [ ! -d "bindings/bindings" ] || [ ! -f "bindings/bindings/bindings.go" ]; then
    echo "Error: Failed to generate Go bindings"
    exit 1
fi

echo "Creating platform-specific binding files..."
./scripts/create_platform_bindings.sh "$PLATFORM_DIR"

echo "Building Go module..."
go build ./...

echo "Build completed successfully for $PLATFORM_DIR"
