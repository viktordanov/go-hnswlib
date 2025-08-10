# Running Tests & Experiments

## Core Library
```bash
# Run main examples
go run examples.go

# Build library
go build ./...
```

## Experiments
```bash
cd experiments/

# Run parallel examples (6-10x speedup demo)
go run parallel_examples.go parallel.go

# Run performance tests  
go test -run TestPerformanceComparison -v parallel_benchmark_test.go parallel.go

# Run parallel benchmarks
go test -bench=. parallel_benchmark_test.go parallel.go

# Run comprehensive benchmarks (EF parameter study)
go run comprehensive_benchmark.go naive.go
```

## Quick Performance Test
```bash
# See 6-10x speedup with parallel operations
cd experiments && go run parallel_examples.go parallel.go
```
