# HNSW Experimental Tests & Benchmarks

This directory contains comprehensive testing and benchmarking suites for the HNSW library, including performance analysis, safety testing, and experimental features.

## 🔧 Prerequisites

```bash
# From project root
cd experimental/
go mod tidy
```

## 📊 Test Categories

### Index Performance & Persistence Tests
**File:** `index_persistence_test.go`

Comprehensive analysis of index build, save, load performance and storage efficiency with beautiful table output.

```bash
# Run all persistence tests
go test -v index_persistence_test.go

# Individual test functions
go test -run TestIndexBuildTime -v        # Build time analysis
go test -run TestIndexSaveTime -v         # Save performance & file sizes  
go test -run TestIndexLoadTime -v         # Load performance analysis
go test -run TestIndexSizeAnalysis -v     # Storage overhead analysis
go test -run BenchmarkIndexConstruction   # Construction benchmarks
```



### Algorithm Comparison & Accuracy Tests  
**File:** `benchmark_test.go`

Compare HNSW vs naive implementations for accuracy and performance validation.

```bash
# Run accuracy and performance tests
go test -v benchmark_test.go

# Individual test functions
go test -run TestCosineSimilarityAccuracy -v    # Recall accuracy testing
go test -run TestPerformanceComparison -v       # Speed comparison tables

# Benchmark functions  
go test -bench=BenchmarkHNSWCosine -v          # HNSW performance
go test -bench=BenchmarkNaiveCosine -v         # Naive baseline
go test -bench=. -v                            # All benchmarks
```

### Parallel Safety & Stress Tests
**File:** `safety_test.go`

Critical thread safety validation and high-load stress testing with comprehensive result tables.

```bash
# Essential safety tests (recommended for production validation)
go test -run TestParallelSafety -v              # Data integrity (10K vectors, 1K queries)
go test -run TestConcurrentAccess -v            # Concurrent access (500-2000 goroutines) 
go test -run TestLargeScaleStressTest -v        # Stress test (20K vectors, 5K queries)

# Race condition detection (CRITICAL for production)
go test -race -run TestParallelSafety -v
go test -race -run TestConcurrentAccess -v  
go test -race -run TestLargeScaleStressTest -v

# Run ALL safety tests with race detection
go test -race -v safety_test.go
```



## 🚀 Interactive Examples & Demos

### Parallel Operations Demo
```bash
# Live parallel performance demonstration (6-10x speedup)
go run cmd/parallel_examples/main.go
```

### Advanced Benchmarking Suite
```bash  
# Comprehensive benchmark with configurable parameters
go run cmd/benchmark/main.go

# Access the full benchmark suite in benchmark/ directory
go run benchmark/benchmark.go
```

## 📈 Performance Analysis Features

### Table-Formatted Output
All tests now use the `tablewr` library for professional table formatting:
- **Clean alignment** with proper column separation
- **Consistent formatting** across all test suites  
- **Easy-to-read metrics** for performance analysis
- **Status indicators** (✅/❌) for quick assessment

### Key Metrics Tracked
- **Build Performance**: Construction time, vectors/second
- **Search Performance**: Query time, QPS (queries per second) 
- **Storage Efficiency**: File sizes, compression ratios
- **Accuracy Metrics**: Recall@K, precision analysis
- **Concurrency Safety**: Thread safety validation, race detection
- **Scalability Limits**: High-load performance, worker scaling

## 🔍 Quick Start Workflows

### Development Validation
```bash
# Quick validation for development
go test -v benchmark_test.go                    # Accuracy check
go test -run TestParallelSafety -v             # Basic safety  
go test -run TestIndexBuildTime -v             # Performance baseline
```

### Production Readiness Check
```bash
# Comprehensive production validation
go test -race -v safety_test.go                # Full safety suite with race detection
go test -v index_persistence_test.go           # Complete performance analysis  
go test -bench=. -v                            # All benchmarks
```

### Performance Research
```bash
# Deep performance analysis
go test -run TestIndexSizeAnalysis -v          # Storage optimization
go test -run TestLargeScaleStressTest -v       # Scalability limits
go run cmd/benchmark/main.go                   # Configurable benchmarks
```

## 📁 Directory Structure

```
experimental/
├── README.md                    # This file
├── go.mod                       # Module dependencies (includes tablewr)
├── index_persistence_test.go    # Performance & storage tests
├── benchmark_test.go            # Accuracy & comparison tests  
├── safety_test.go              # Thread safety & stress tests
├── benchmark/                  # Advanced benchmarking suite
│   ├── benchmark.go           # Comprehensive benchmark framework
│   └── naive.go               # Naive implementation for comparison
├── parallel/                  # Parallel operation utilities
│   └── parallel.go           # Parallel search & batch operations
├── cmd/                      # Executable examples
│   ├── benchmark/           # Interactive benchmark tool
│   └── parallel_examples/   # Parallel demo application
└── results/                 # Benchmark output storage
    └── benchmark_results.txt
```

## ⚡ Performance Expectations

**Parallel Speedup:** 6-10x improvement over sequential processing

## 🛡️ Safety & Production Notes

- **Always run race detection** before production deployment
- **Monitor recall metrics** to ensure accuracy requirements  
- **Validate thread safety** under expected load patterns
- **Test memory pressure** scenarios for your specific use case
- **Profile performance** with realistic data distributions

## 📋 Dependencies

- **Core:** `github.com/viktordanov/go-hnswlib` (main library)
- **Tables:** `github.com/shubhang93/tablewr` (formatted output)
- **Go Version:** 1.24+ recommended

---

*For additional performance tuning and advanced configuration, see the main project documentation.*
