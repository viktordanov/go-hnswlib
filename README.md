# go-hnswlib

Go bindings for [hnswlib](https://github.com/nmslib/hnswlib) - fast approximate nearest neighbor search.

## Installation

```bash
go get github.com/viktordanov/go-hnswlib
```

No C++ compilation required. Static libraries for common platforms are included.

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

    // Add vectors
    vec := []float32{0.1, 0.2, 0.3, ...} // 128 dimensions
    index.Add(vec, 0)

    // Search for 5 nearest neighbors
    query := []float32{0.1, 0.2, 0.3, ...} // 128 dimensions
    labels, distances, count := index.SearchK(query, 5)
    
    fmt.Printf("Found %d neighbors\n", count)
}
```

## API

**Create Index:**
- `hnsw.NewL2(dim, maxElements, M, efConstruction, seed)` - Euclidean distance
- `hnsw.NewIP(dim, maxElements, M, efConstruction, seed)` - Inner product
- `hnsw.Load(space, dim, path)` - Load from file

**Operations:**
- `index.Add(vec, label)` - Add vector with label
- `index.SearchK(query, k)` - Find k nearest neighbors  
- `index.Save(path)` - Save to file
- `index.SetEf(ef)` - Set search accuracy
- `index.Close()` - Free memory

## Development

**Build from source:**
```bash
git clone https://github.com/viktordanov/go-hnswlib
cd go-hnswlib
./build.sh
```

**Cross-compile for different platforms:**
```bash
PLATFORM=linux ARCH=amd64 make
PLATFORM=linux ARCH=arm64 make  
PLATFORM=darwin ARCH=amd64 make
PLATFORM=darwin ARCH=arm64 make
```

**Project structure:**
```
├── hnswlib/           # C++ headers
├── wrapper/           # C++ wrapper 
├── bindings/          # Generated Go bindings
├── hnsw/              # High-level Go API
└── build/             # Platform-specific static libraries
```

**Requirements for building:**
- Go 1.18+
- C++ compiler (clang++ or g++)
- c-for-go (auto-installed)

## Supported Platforms

Pre-built static libraries included for:
- darwin_arm64 (macOS Apple Silicon)
- linux_amd64 (Linux x86-64)
- linux_arm64 (Linux ARM64)

## License

Same as [hnswlib](https://github.com/nmslib/hnswlib).