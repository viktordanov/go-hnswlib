#!/bin/bash
set -e

echo "🚀 Building go-hnswlib from scratch..."

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Error: This script must be run from a git repository root"
    exit 1
fi

# Initialize and update git submodules (hnswlib)
echo "📦 Initializing and updating git submodules..."
git submodule update --init --recursive

# Verify hnswlib submodule exists
if [ ! -d "hnswlib" ] || [ ! -f "hnswlib/hnswlib/hnswlib.h" ]; then
    echo "❌ Error: hnswlib submodule not found or incomplete"
    echo "   Make sure the hnswlib submodule is properly configured"
    exit 1
fi

echo "✅ hnswlib submodule ready"

# Check for required tools
echo "🔍 Checking for required tools..."

# Check for Go
if ! command -v go &> /dev/null; then
    echo "❌ Error: Go is not installed or not in PATH"
    exit 1
fi
echo "✅ Go found: $(go version)"

# Check for C++ compiler
if ! command -v clang++ &> /dev/null && ! command -v g++ &> /dev/null; then
    echo "❌ Error: No C++ compiler found (clang++ or g++)"
    exit 1
fi
if command -v clang++ &> /dev/null; then
    echo "✅ C++ compiler found: clang++"
else
    echo "✅ C++ compiler found: g++"
fi

# Check for c-for-go
if ! command -v c-for-go &> /dev/null; then
    echo "📥 Installing c-for-go..."
    go install github.com/xlab/c-for-go@latest
    if ! command -v c-for-go &> /dev/null; then
        echo "❌ Error: Failed to install c-for-go"
        echo "   Make sure GOPATH/bin is in your PATH"
        exit 1
    fi
fi
echo "✅ c-for-go found: $(c-for-go --help | head -1 || echo 'c-for-go')"

# Clean previous builds
echo "🧹 Cleaning previous builds..."
make clean 2>/dev/null || true
rm -rf bindings/bindings

# Build C++ wrapper library
echo "🔨 Building C++ wrapper library..."
make

# Verify static library was created
if [ ! -f "build/libhnsw_wrapper.a" ]; then
    echo "❌ Error: Failed to build static library"
    exit 1
fi
echo "✅ Static library built: build/libhnsw_wrapper.a"

# Generate Go bindings
echo "🎯 Generating Go bindings..."
go generate ./...

# Verify bindings were generated
if [ ! -d "bindings/bindings" ] || [ ! -f "bindings/bindings/bindings.go" ]; then
    echo "❌ Error: Failed to generate Go bindings"
    exit 1
fi

# Add cgo directives manually (workaround for c-for-go)
echo "🔧 Adding cgo directives to generated bindings..."
if ! grep -q "#cgo CFLAGS" bindings/bindings/bindings.go; then
    # Create a temp file with the cgo directives
    cat > /tmp/cgo_header << 'EOF'
/*
#cgo CFLAGS: -I${SRCDIR}/../.. -I${SRCDIR}/../../hnswlib -I${SRCDIR}/../../hnswlib/hnswlib
#cgo LDFLAGS: ${SRCDIR}/../../build/libhnsw_wrapper.a -lc++
#include "hnsw_wrapper.h"
#include <stdlib.h>
#include "cgo_helpers.h"
*/
EOF
    
    # Replace the comment block in bindings.go
    sed -i.bak '/^\/\*/,/^\*\// {
        /^\/\*/ r /tmp/cgo_header
        d
    }' bindings/bindings/bindings.go
    
    rm -f bindings/bindings/bindings.go.bak /tmp/cgo_header
fi
echo "✅ cgo directives added"

# Build and test Go module
echo "🔧 Building Go module..."
go build ./...

echo "✅ Go module built successfully"

# Run basic test
echo "🧪 Running basic functionality test..."
if go run example/main.go > /dev/null 2>&1; then
    echo "✅ Basic functionality test passed"
else
    echo "❌ Basic functionality test failed"
    exit 1
fi

# Clean up intermediate build artifacts (keep only what Go needs)
echo "🧹 Cleaning up intermediate build artifacts..."

# Remove object files but keep the static library
make clean-objects

# Remove any temporary files
rm -f /tmp/cgo_header 2>/dev/null || true

echo ""
echo "🎉 Build completed successfully!"
echo ""
echo "📁 Final files (kept for Go module):"
echo "   - build/libhnsw_wrapper.a (C++ static library - required for linking)"
echo "   - bindings/bindings/ (Generated Go bindings - required for compilation)"
echo ""
echo "🧹 Cleaned up:"
echo "   - Removed intermediate object files"
echo "   - Removed temporary files"
echo ""
echo "🚀 Ready to use! Try: go run example/main.go"
