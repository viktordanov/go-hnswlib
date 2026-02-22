# go-hnswlib

Go bindings for [hnswlib](https://github.com/nmslib/hnswlib) - fast approximate nearest neighbor search.

## Installation

```bash
go get github.com/viktordanov/go-hnswlib
```

Uses cgo: the bundled C++ wrapper is compiled by the Go toolchain on the target machine.
A working C++17 toolchain is required where you build.

## Usage

```go
package main

import (
    "fmt"
    "github.com/viktordanov/go-hnswlib/hnsw"
)

func main() {
    // Create index: 128 dimensions, max 1000 elements
    index := hnsw.NewL2(128, 1000, 16, 200, 42)
    defer index.Close()

    // Add vectors (safe - returns error)
    vec := []float32{0.1, 0.2, 0.3, ...} // 128 dimensions
    if err := index.Add(vec, 0); err != nil {
        log.Fatal(err)
    }

    // Search for 5 nearest neighbors
    query := []float32{0.1, 0.2, 0.3, ...} // 128 dimensions
    labels, distances, count := index.SearchK(query, 5)
    
    fmt.Printf("Found %d neighbors\n", count)
}
```

**📖 More Examples:** See [`examples.go`](examples.go) for comprehensive examples of all spaces and features.

**⚡ Benchmarks:** See [`experiments/`](experiments/) for comprehensive EF parameter studies and performance analysis.

## API

**Create Index:**
- `hnsw.NewL2(dim, maxElements, M, efConstruction, seed)` - Euclidean distance
- `hnsw.NewIP(dim, maxElements, M, efConstruction, seed)` - Inner product
- `hnsw.NewCosine(dim, maxElements, M, efConstruction, seed)` - Cosine similarity
- `index, err := hnsw.Load(space, dim, path)` - Load from file

**Operations:**
- `err := index.Add(vec, label)` - Add vector with label (safe, checks capacity)
- `labels, distances, count := index.SearchK(query, k)` - Find k nearest neighbors  
- `labels, similarities, count := index.SearchKSimilarity(query, k)` - Get similarities instead of distances
- `err := index.Save(path)` - Save to file (safe)
- `err := index.Resize(newMaxElements)` - Resize index capacity (safe)
- `index.SetEf(ef)` - Set search accuracy
- `index.Close()` - Free memory

**Introspection:**
- `index.GetCurrentCount()` - Number of elements in index
- `index.GetMaxElements()` - Maximum capacity
- `index.GetDeletedCount()` - Number of deleted elements
- `index.IsCosineSpace()` - Check if using cosine similarity

**Delete Management:**
- `err := index.MarkDeleted(label)` - Soft delete element (safe)
- `err := index.UnmarkDeleted(label)` - Restore deleted element (safe)

## Development

**Build from source:**
```bash
git clone https://github.com/viktordanov/go-hnswlib
cd go-hnswlib
go build ./...
go test ./...
```

**Regenerate low-level bindings after C API changes:**
```bash
./generate_bindings.sh
go build ./...
```

**Project structure:**
```
├── hnswlib/           # Upstream hnswlib headers
├── hnsw_wrapper.h     # C wrapper API
├── hnsw_wrapper.cpp   # C++ wrapper implementation
├── bindings.go        # Generated c-for-go bindings
├── cgo_helpers.*      # Generated cgo helpers
├── types.go           # Generated C type mappings
└── hnsw/              # High-level Go API
```

**Requirements for building:**
- Go 1.18+
- C++ compiler (clang++ or g++)
- c-for-go (auto-installed)

## Supported Platforms

Tested on:
- darwin_arm64 (macOS Apple Silicon)
- linux_amd64 (Linux x86-64)
- linux_arm64 (Linux ARM64)

## License

Same as [hnswlib](https://github.com/nmslib/hnswlib).
