# Quick Start Guide

## One Command Build

```bash
git clone <your-repo-url>
cd go-hnswlib
./build.sh
```

That's it! This single command will:

1. ✅ **Fetch hnswlib submodule** - Gets the latest hnswlib C++ library
2. ✅ **Check dependencies** - Verifies Go, C++, installs c-for-go if needed  
3. ✅ **Build C++ wrapper** - Compiles static library from hnswlib
4. ✅ **Generate Go bindings** - Uses c-for-go to create Go interfaces
5. ✅ **Build Go module** - Compiles and tests everything
6. ✅ **Verify functionality** - Runs example to ensure it works

## Test It

```bash
go run example/main.go
```

Expected output:
```
Creating HNSW index...
Adding vectors...
Searching for nearest neighbors...
Found 5 nearest neighbors:
  Label: 49, Distance: 14.903757
  Label: 19, Distance: 17.062328
  ...
HNSW wrapper working successfully!
```

## Use In Your Project

```go
import "github.com/vikimaster2/go-hnswlib/hnsw"

// Create index
index := hnsw.NewL2(128, 1000, 16, 200, 42)
defer index.Close()

// Add data  
index.Add(vector, label)

// Search
labels, distances, count := index.SearchK(query, 5)
```

## Update hnswlib

```bash
make update-submodule  # Updates to latest hnswlib
./build.sh            # Rebuilds everything
```

## Verify Everything Works

```bash
./verify.sh  # Complete test from clean state
```

## What Gets Built

- `build/libhnsw_wrapper.a` - Static C++ library
- `bindings/bindings/` - Generated Go bindings  
- `hnsw/` - High-level Go wrapper

## Dependencies

- Go 1.18+
- C++11 compiler (clang++/g++)
- Git (for submodules)

The build script installs c-for-go automatically.
