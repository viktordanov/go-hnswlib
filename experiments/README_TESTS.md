# Running Tests & Experiments

## Core Library
```bash
# Run main examples (from project root)
go run examples.go

# Build library (from project root)
go build ./...
```

## Experiments
*Note: All commands below should be run from the `experiments/` directory*

### Parallel Operations Demo
```bash
# Run parallel examples (6-10x speedup demo)
go run parallel_examples.go parallel.go

# Run performance comparison
go test -run TestPerformanceComparison -v parallel_benchmark_test.go parallel.go

# Run all parallel benchmarks
go test -bench=. parallel_benchmark_test.go parallel.go
```

### Safety & Thread Safety Tests
```bash
# Test parallel safety and data integrity (10K vectors, 1K queries)
go test -run TestParallelSafety -v safety_test.go parallel.go

# Test concurrent access safety (500-2000 goroutines)
go test -run TestConcurrentAccess -v safety_test.go parallel.go

# Large scale stress test (20K vectors, 5K queries, high worker counts)
go test -run TestLargeScaleStressTest -v safety_test.go parallel.go

# Run with race detection (critical for production use)
go test -race -run TestParallelSafety -v safety_test.go parallel.go
go test -race -run TestConcurrentAccess -v safety_test.go parallel.go
go test -race -run TestLargeScaleStressTest -v safety_test.go parallel.go

# Run ALL safety tests (comprehensive stress testing)
go test -race -v safety_test.go parallel.go
```

### Other Benchmarks
```bash
# Run comprehensive benchmarks (EF parameter study)
go run comprehensive_benchmark.go naive.go
```

## Quick Start
```bash
# From project root
cd experiments/

# See parallel performance demo
go run parallel_examples.go parallel.go

# Verify thread safety (important!)
go test -race -v safety_test.go parallel.go

# Test large scale limits
go test -run TestLargeScaleStressTest -v safety_test.go parallel.go
```
