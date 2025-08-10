package main

import (
	"fmt"
	"log"
	"math/rand"
	"runtime"
	"time"

	"github.com/viktordanov/go-hnswlib/hnsw"
)

func parallelExamples() {
	fmt.Printf("🚀 Go-HNSWLIB Parallel Operations Examples\n")
	fmt.Printf("==========================================\n")
	fmt.Printf("CPU cores available: %d\n\n", runtime.GOMAXPROCS(0))

	// Example 1: Batch Add
	batchAddExample()

	// Example 2: Parallel Search
	parallelSearchExample()

	// Example 3: Performance Comparison
	performanceComparisonExample()
}

func batchAddExample() {
	fmt.Printf("📦 Example 1: Batch Add Operations\n")
	fmt.Printf("----------------------------------\n")

	// Create index
	index := hnsw.NewCosine(128, 10000, 16, 200, 42)
	defer index.Close()

	// Generate some random vectors
	vectors := generateExampleVectors(5000, 128)

	// Convert to VectorData format for batch operations
	vectorData := make([]VectorData, len(vectors))
	for i, vec := range vectors {
		vectorData[i] = VectorData{
			Vector: vec,
			Label:  uint64(i),
		}
	}

	// Batch add with default settings (uses all CPU cores)
	start := time.Now()
	err := BatchAdd(index, vectorData, nil)
	if err != nil {
		log.Fatalf("Batch add failed: %v", err)
	}
	batchTime := time.Since(start)

	fmt.Printf("✅ Added %d vectors in %v using batch add\n", len(vectors), batchTime)
	fmt.Printf("📊 Current index size: %d vectors\n", index.GetCurrentCount())

	// Example with custom options
	moreVectors := generateExampleVectors(1000, 128)
	moreVectorData := make([]VectorData, len(moreVectors))
	for i, vec := range moreVectors {
		moreVectorData[i] = VectorData{
			Vector: vec,
			Label:  uint64(len(vectors) + i), // Continue labeling
		}
	}

	opts := &BatchAddOptions{
		MaxWorkers: 4,  // Limit to 4 workers
		ChunkSize:  50, // Process in smaller chunks
	}

	start = time.Now()
	err = BatchAdd(index, moreVectorData, opts)
	if err != nil {
		log.Fatalf("Batch add with options failed: %v", err)
	}
	batchTimeWithOpts := time.Since(start)

	fmt.Printf("✅ Added %d more vectors in %v using batch add (4 workers)\n", len(moreVectors), batchTimeWithOpts)
	fmt.Printf("📊 Final index size: %d vectors\n\n", index.GetCurrentCount())
}

func parallelSearchExample() {
	fmt.Printf("🔍 Example 2: Parallel Search Operations\n")
	fmt.Printf("----------------------------------------\n")

	// Create and populate index
	index := hnsw.NewCosine(64, 10000, 16, 200, 42)
	defer index.Close()

	// Add some data first
	vectors := generateExampleVectors(5000, 64)
	vectorData := make([]VectorData, len(vectors))
	for i, vec := range vectors {
		vectorData[i] = VectorData{
			Vector: vec,
			Label:  uint64(i),
		}
	}

	if err := BatchAdd(index, vectorData, nil); err != nil {
		log.Fatalf("Failed to populate index: %v", err)
	}

	// Generate query vectors
	queries := generateExampleVectors(100, 64)
	k := 5

	// Example 1: ParallelSearch with full results
	start := time.Now()
	results, err := ParallelSearch(index, queries, k, nil)
	if err != nil {
		log.Fatalf("Parallel search failed: %v", err)
	}
	parallelTime := time.Since(start)

	fmt.Printf("✅ Parallel search completed in %v\n", parallelTime)
	fmt.Printf("📊 Found results for %d queries\n", len(results))

	// Show example results
	if len(results) > 0 {
		fmt.Printf("🔍 Example result for query 0:\n")
		result := results[0]
		fmt.Printf("   Labels: %v\n", result.Labels[:min(3, len(result.Labels))])
		fmt.Printf("   Distances: %.4f, %.4f, %.4f...\n",
			result.Distances[0], result.Distances[1], result.Distances[2])
	}

	// Example 2: BatchSearchK convenience function
	start = time.Now()
	labels, distances, counts, err := BatchSearchK(index, queries, k, nil)
	if err != nil {
		log.Fatalf("Batch search failed: %v", err)
	}
	batchTime := time.Since(start)

	fmt.Printf("✅ Batch search completed in %v\n", batchTime)
	fmt.Printf("📊 Processed %d queries, found %d results for first query\n", len(labels), counts[0])
	if len(distances) > 0 && len(distances[0]) > 2 {
		fmt.Printf("🔍 Example distances for query 0: %.4f, %.4f, %.4f...\n",
			distances[0][0], distances[0][1], distances[0][2])
	}

	// Example 3: ParallelSearchSimilarity for cosine similarity scores
	start = time.Now()
	simResults, err := ParallelSearchSimilarity(index, queries, k, nil)
	if err != nil {
		log.Fatalf("Parallel similarity search failed: %v", err)
	}
	simTime := time.Since(start)

	fmt.Printf("✅ Parallel similarity search completed in %v\n", simTime)
	if len(simResults) > 0 {
		fmt.Printf("🔍 Example similarity scores for query 0: %.4f, %.4f, %.4f...\n",
			simResults[0].Similarities[0], simResults[0].Similarities[1], simResults[0].Similarities[2])
	}

	// Example 4: Custom parallel options
	opts := &ParallelSearchOptions{
		MaxWorkers: 2, // Limit to 2 workers
	}

	start = time.Now()
	_, err = ParallelSearch(index, queries, k, opts)
	if err != nil {
		log.Fatalf("Parallel search with options failed: %v", err)
	}
	customTime := time.Since(start)

	fmt.Printf("✅ Parallel search with 2 workers completed in %v\n\n", customTime)
}

func performanceComparisonExample() {
	fmt.Printf("⚡ Example 3: Performance Comparison\n")
	fmt.Printf("-----------------------------------\n")

	// Create and populate index
	index := hnsw.NewCosine(100, 20000, 16, 200, 42)
	defer index.Close()

	// Add data
	vectors := generateExampleVectors(10000, 100)
	vectorData := make([]VectorData, len(vectors))
	for i, vec := range vectors {
		vectorData[i] = VectorData{
			Vector: vec,
			Label:  uint64(i),
		}
	}

	if err := BatchAdd(index, vectorData, nil); err != nil {
		log.Fatalf("Failed to populate index: %v", err)
	}

	// Generate queries
	queries := generateExampleVectors(500, 100)
	k := 10

	fmt.Printf("📊 Testing with %d vectors, %d queries, k=%d\n", len(vectors), len(queries), k)

	// Sequential search
	start := time.Now()
	for _, query := range queries {
		_, _, _ = index.SearchK(query, k)
	}
	sequentialTime := time.Since(start)

	// Parallel search
	start = time.Now()
	_, err := ParallelSearch(index, queries, k, nil)
	if err != nil {
		log.Fatalf("Parallel search failed: %v", err)
	}
	parallelTime := time.Since(start)

	// Results
	speedup := float64(sequentialTime) / float64(parallelTime)
	qpsSeq := float64(len(queries)) / sequentialTime.Seconds()
	qpsPar := float64(len(queries)) / parallelTime.Seconds()

	fmt.Printf("\n📈 Performance Results:\n")
	fmt.Printf("   Sequential: %v (%.0f queries/sec)\n", sequentialTime, qpsSeq)
	fmt.Printf("   Parallel:   %v (%.0f queries/sec)\n", parallelTime, qpsPar)
	fmt.Printf("   Speedup:    %.2fx\n", speedup)

	if speedup > 1.5 {
		fmt.Printf("   🎉 Great! Parallel processing is significantly faster!\n")
	} else if speedup > 1.0 {
		fmt.Printf("   ✅ Parallel processing provides modest improvement\n")
	} else {
		fmt.Printf("   ⚠️  Parallel overhead may be too high for this workload\n")
	}

	// Test with different worker counts
	fmt.Printf("\n🔧 Testing different worker counts:\n")
	workerCounts := []int{1, 2, 4, 8, runtime.GOMAXPROCS(0)}

	for _, workers := range workerCounts {
		opts := &ParallelSearchOptions{MaxWorkers: workers}
		start = time.Now()
		_, err := ParallelSearch(index, queries, k, opts)
		if err != nil {
			log.Printf("Failed with %d workers: %v", workers, err)
			continue
		}
		elapsed := time.Since(start)
		qps := float64(len(queries)) / elapsed.Seconds()

		fmt.Printf("   %d workers: %v (%.0f queries/sec)\n", workers, elapsed, qps)
	}

	fmt.Printf("\n✅ All examples completed successfully!\n")
}

func generateExampleVectors(count, dim int) [][]float32 {
	rand.Seed(42) // Fixed seed for reproducible results
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

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func main() {
	parallelExamples()
}
