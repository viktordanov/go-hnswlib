package main

import (
	"fmt"
	"math/rand"
	"runtime"
	"testing"
	"time"

	"github.com/viktordanov/go-hnswlib/hnsw"
)

const (
	benchDim       = 128
	benchMaxElems  = 50000
	benchDataSize  = 10000
	benchQuerySize = 1000
	benchK         = 10
	benchM         = 16
	benchEF        = 200
)

// generateRandomVectors creates random vectors for benchmarking
func generateRandomVectors(count, dim int, seed int64) [][]float32 {
	rand.Seed(seed)
	vectors := make([][]float32, count)
	for i := 0; i < count; i++ {
		vector := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vector[j] = rand.Float32()*2 - 1 // Random values between -1 and 1
		}
		vectors[i] = vector
	}
	return vectors
}

// prepareBenchmarkIndex creates and populates an index for benchmarking
func prepareBenchmarkIndex(b *testing.B) (*hnsw.Index, [][]float32) {
	b.Helper()
	
	// Create index
	index := hnsw.NewCosine(benchDim, benchMaxElems, benchM, benchEF, 42)
	
	// Generate data
	vectors := generateRandomVectors(benchDataSize, benchDim, 42)
	
	// Add vectors sequentially for consistent baseline
	for i, vec := range vectors {
		if err := index.Add(vec, uint64(i)); err != nil {
			b.Fatalf("Failed to add vector: %v", err)
		}
	}
	
	// Generate query vectors
	queries := generateRandomVectors(benchQuerySize, benchDim, 123)
	
	return index, queries
}

// BenchmarkSequentialAdd tests traditional sequential adding
func BenchmarkSequentialAdd(b *testing.B) {
	vectors := generateRandomVectors(benchDataSize/10, benchDim, 42) // Smaller for add benchmark
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		index := hnsw.NewCosine(benchDim, benchMaxElems, benchM, benchEF, 42)
		
		for j, vec := range vectors {
			if err := index.Add(vec, uint64(j)); err != nil {
				b.Fatalf("Failed to add vector: %v", err)
			}
		}
		
		index.Close()
	}
}

// BenchmarkBatchAdd tests parallel batch adding
func BenchmarkBatchAdd(b *testing.B) {
	vectors := generateRandomVectors(benchDataSize/10, benchDim, 42) // Smaller for add benchmark
	
	// Convert to VectorData format
	vectorData := make([]VectorData, len(vectors))
	for i, vec := range vectors {
		vectorData[i] = VectorData{
			Vector: vec,
			Label:  uint64(i),
		}
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		index := hnsw.NewCosine(benchDim, benchMaxElems, benchM, benchEF, 42)
		
		if err := BatchAdd(index, vectorData, nil); err != nil {
			b.Fatalf("Failed to batch add vectors: %v", err)
		}
		
		index.Close()
	}
}

// BenchmarkSequentialSearch tests traditional sequential search
func BenchmarkSequentialSearch(b *testing.B) {
	index, queries := prepareBenchmarkIndex(b)
	defer index.Close()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, query := range queries {
			_, _, _ = index.SearchK(query, benchK)
		}
	}
}

// BenchmarkParallelSearch tests parallel search
func BenchmarkParallelSearch(b *testing.B) {
	index, queries := prepareBenchmarkIndex(b)
	defer index.Close()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := ParallelSearch(index, queries, benchK, nil)
		if err != nil {
			b.Fatalf("Failed to perform parallel search: %v", err)
		}
	}
}

// BenchmarkBatchSearchK tests batch search convenience function
func BenchmarkBatchSearchK(b *testing.B) {
	index, queries := prepareBenchmarkIndex(b)
	defer index.Close()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _, err := BatchSearchK(index, queries, benchK, nil)
		if err != nil {
			b.Fatalf("Failed to perform batch search: %v", err)
		}
	}
}

// BenchmarkParallelSearchWithWorkers tests parallel search with different worker counts
func BenchmarkParallelSearchWithWorkers(b *testing.B) {
	index, queries := prepareBenchmarkIndex(b)
	defer index.Close()
	
	workerCounts := []int{1, 2, 4, 8, runtime.GOMAXPROCS(0)}
	
	for _, workers := range workerCounts {
		b.Run(fmt.Sprintf("workers-%d", workers), func(b *testing.B) {
			opts := &ParallelSearchOptions{MaxWorkers: workers}
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := ParallelSearch(index, queries, benchK, opts)
				if err != nil {
					b.Fatalf("Failed to perform parallel search: %v", err)
				}
			}
		})
	}
}

// Performance comparison test (not a benchmark, but useful for comparison)
func TestPerformanceComparison(t *testing.T) {
	fmt.Printf("Performance Comparison Test\n")
	fmt.Printf("===========================\n")
	fmt.Printf("Vector dimensions: %d\n", benchDim)
	fmt.Printf("Data size: %d vectors\n", benchDataSize)
	fmt.Printf("Query size: %d queries\n", benchQuerySize)
	fmt.Printf("CPU cores available: %d\n", runtime.GOMAXPROCS(0))
	fmt.Printf("\n")

	// Create index and populate
	index := hnsw.NewCosine(benchDim, benchMaxElems, benchM, benchEF, 42)
	defer index.Close()
	
	vectors := generateRandomVectors(benchDataSize, benchDim, 42)
	
	// Test batch add vs sequential add
	fmt.Printf("=== Adding %d vectors ===\n", len(vectors))
	
	// Sequential add timing
	index1 := hnsw.NewCosine(benchDim, benchMaxElems, benchM, benchEF, 42)
	start := time.Now()
	for i, vec := range vectors {
		if err := index1.Add(vec, uint64(i)); err != nil {
			t.Fatalf("Failed to add vector: %v", err)
		}
	}
	sequentialAddTime := time.Since(start)
	index1.Close()
	
	// Batch add timing
	vectorData := make([]VectorData, len(vectors))
	for i, vec := range vectors {
		vectorData[i] = VectorData{
			Vector: vec,
			Label:  uint64(i),
		}
	}
	
	index2 := hnsw.NewCosine(benchDim, benchMaxElems, benchM, benchEF, 42)
	start = time.Now()
	if err := BatchAdd(index2, vectorData, nil); err != nil {
		t.Fatalf("Failed to batch add vectors: %v", err)
	}
	batchAddTime := time.Since(start)
	index2.Close()
	
	fmt.Printf("Sequential add: %v\n", sequentialAddTime)
	fmt.Printf("Batch add:      %v\n", batchAddTime)
	fmt.Printf("Speedup:        %.2fx\n", float64(sequentialAddTime)/float64(batchAddTime))
	fmt.Printf("\n")
	
	// Populate index for search tests
	for i, vec := range vectors {
		if err := index.Add(vec, uint64(i)); err != nil {
			t.Fatalf("Failed to add vector: %v", err)
		}
	}
	
	// Generate queries
	queries := generateRandomVectors(benchQuerySize, benchDim, 123)
	
	fmt.Printf("=== Searching %d queries ===\n", len(queries))
	
	// Sequential search timing
	start = time.Now()
	for _, query := range queries {
		_, _, _ = index.SearchK(query, benchK)
	}
	sequentialSearchTime := time.Since(start)
	
	// Parallel search timing
	start = time.Now()
	_, err := ParallelSearch(index, queries, benchK, nil)
	if err != nil {
		t.Fatalf("Failed to perform parallel search: %v", err)
	}
	parallelSearchTime := time.Since(start)
	
	fmt.Printf("Sequential search: %v\n", sequentialSearchTime)
	fmt.Printf("Parallel search:   %v\n", parallelSearchTime)
	fmt.Printf("Speedup:           %.2fx\n", float64(sequentialSearchTime)/float64(parallelSearchTime))
	fmt.Printf("Queries per sec:   %.0f (sequential) vs %.0f (parallel)\n",
		float64(len(queries))/sequentialSearchTime.Seconds(),
		float64(len(queries))/parallelSearchTime.Seconds())
}
