package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/viktordanov/go-hnswlib/experimental/parallel"
	"github.com/viktordanov/go-hnswlib/hnsw"
)

func main() {
	fmt.Printf("🚀 HNSW Parallel Operations Examples\n")
	fmt.Printf("====================================\n\n")

	runBatchAddExample()
	runParallelSearchExample()
	runBatchSearchExample()
}

func runBatchAddExample() {
	fmt.Printf("1. Batch Add Example\n")
	fmt.Printf("   Adding 1000 vectors using parallel workers\n")

	index := hnsw.NewCosine(128, 2000, 32, 400, 42)
	defer index.Close()

	vectors := generateRandomVectors(1000, 128)

	vectorData := make([]parallel.VectorData, len(vectors))
	for i, vec := range vectors {
		vectorData[i] = parallel.VectorData{
			Vector: vec,
			Label:  uint64(i),
		}
	}

	start := time.Now()
	err := parallel.BatchAdd(index, vectorData, &parallel.BatchAddOptions{
		MaxWorkers: 4,
	})
	batchTime := time.Since(start)

	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("   ✅ Added %d vectors in %v\n", len(vectors), batchTime)
	fmt.Printf("   📊 Rate: %.0f vectors/sec\n\n", float64(len(vectors))/batchTime.Seconds())
}

func runParallelSearchExample() {
	fmt.Printf("2. Parallel Search Example\n")
	fmt.Printf("   Searching with multiple workers\n")

	index := hnsw.NewCosine(128, 2000, 32, 400, 42)
	defer index.Close()

	vectors := generateRandomVectors(1000, 128)
	vectorData := make([]parallel.VectorData, len(vectors))
	for i, vec := range vectors {
		vectorData[i] = parallel.VectorData{
			Vector: vec,
			Label:  uint64(i),
		}
	}

	err := parallel.BatchAdd(index, vectorData, &parallel.BatchAddOptions{
		MaxWorkers: 4,
	})
	if err != nil {
		log.Fatal(err)
	}

	index.SetEf(100)

	queries := generateRandomVectors(100, 128)

	start := time.Now()
	results, err := parallel.ParallelSearch(index, queries, 10, &parallel.ParallelSearchOptions{
		MaxWorkers: 4,
	})
	searchTime := time.Since(start)

	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("   ✅ Searched %d queries in %v\n", len(queries), searchTime)
	fmt.Printf("   📊 Rate: %.0f queries/sec\n", float64(len(queries))/searchTime.Seconds())
	fmt.Printf("   🎯 Found %d results for first query\n\n", len(results[0].Labels))
}

func runBatchSearchExample() {
	fmt.Printf("3. Batch Search Example\n")
	fmt.Printf("   Convenience wrapper for batch searching\n")

	index := hnsw.NewCosine(128, 2000, 32, 400, 42)
	defer index.Close()

	vectors := generateRandomVectors(500, 128)
	vectorData := make([]parallel.VectorData, len(vectors))
	for i, vec := range vectors {
		vectorData[i] = parallel.VectorData{
			Vector: vec,
			Label:  uint64(i),
		}
	}

	err := parallel.BatchAdd(index, vectorData, &parallel.BatchAddOptions{
		MaxWorkers: 4,
	})
	if err != nil {
		log.Fatal(err)
	}

	index.SetEf(50)

	queries := generateRandomVectors(50, 128)

	start := time.Now()
	labels, err := parallel.BatchSearchK(index, queries, 5, &parallel.ParallelSearchOptions{
		MaxWorkers: 4,
	})
	batchSearchTime := time.Since(start)

	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("   ✅ Batch searched %d queries in %v\n", len(queries), batchSearchTime)
	fmt.Printf("   📊 Rate: %.0f queries/sec\n", float64(len(queries))/batchSearchTime.Seconds())
	fmt.Printf("   🎯 First query results: %v\n\n", labels[0])

	fmt.Printf("🎉 All parallel examples completed successfully!\n")
}

func generateRandomVectors(count, dim int) [][]float32 {
	rand.Seed(42)
	vectors := make([][]float32, count)
	for i := 0; i < count; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float32()*2 - 1
		}
		vectors[i] = vec
	}
	return vectors
}
