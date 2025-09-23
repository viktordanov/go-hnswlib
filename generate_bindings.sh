#!/bin/bash
set -e

echo "🔧 Generating Go bindings..."

# Check if c-for-go is installed
if ! command -v c-for-go &> /dev/null; then
    echo "Installing c-for-go..."
    go install github.com/xlab/c-for-go@latest
fi

# Clean up any existing temp bindings
rm -rf bindings/

# Generate bindings (c-for-go outputs in bindings directory)
echo "Running c-for-go..."
c-for-go -out . cforgo.yml

# Move generated files to root and clean up
echo "Moving bindings to root package..."
if [ -d "bindings" ]; then
    # Update package name in generated files to match root package
    sed -i.bak 's/package bindings/package hnswlib/' bindings/bindings.go
    sed -i.bak 's/package bindings/package hnswlib/' bindings/types.go  
    sed -i.bak 's/package bindings/package hnswlib/' bindings/cgo_helpers.go
    
    # Move files to root
    mv bindings/bindings.go ./bindings.go
    mv bindings/types.go ./types.go
    mv bindings/cgo_helpers.go ./cgo_helpers.go
    mv bindings/cgo_helpers.h ./cgo_helpers.h
    
    # Clean up temp directory and backup files
    rm -rf bindings/
    rm -f *.bak
fi

# Add static compilation cgo directives
echo "Adding static compilation directives..."
if [ -f "bindings.go" ] && ! grep -q "#cgo CPPFLAGS" bindings.go; then
    # Add cgo directives after the package line
    sed -i.bak '/^package hnswlib$/a\
\
/*\
#cgo CPPFLAGS: -I${SRCDIR} -I${SRCDIR}/hnswlib -DHNSWLIB_NO_MANUAL_VECTORIZATION\
#cgo CXXFLAGS: -std=c++17 -O2 -march=native\
#cgo LDFLAGS: -lstdc++\
#cgo linux LDFLAGS: -lm -lpthread -static\
#cgo darwin LDFLAGS: -framework Foundation\
\
#include "hnsw_wrapper.h"\
#include <stdlib.h>\
#include "cgo_helpers.h"\
*/' bindings.go
    rm -f bindings.go.bak
fi

echo "✅ Done! Bindings generated in root package."
echo "Test with: go build ./..."