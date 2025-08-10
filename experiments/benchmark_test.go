package main

import (
	"fmt"
	"testing"
	"time"

	"github.com/viktordanov/go-hnswlib/hnsw"
)

// Benchmark HNSW Cosine Implementation
func BenchmarkHNSWCosine(b *testing.B) {
	dimensions := []int{50, 100, 300}
	datasetSizes := []int{1000, 5000, 10000}

	for _, dim := range dimensions {
		for _, size := range datasetSizes {
			b.Run(fmt.Sprintf("HNSW_dim%d_size%d", dim, size), func(b *testing.B) {
				// Generate test data
				vectors := generateRandomVectors(size, dim, 42)
				queries := generateRandomVectors(100, dim, 123) // 100 query vectors

				// Setup HNSW index with better parameters for recall
				index := hnsw.NewCosine(dim, size*2, 32, 400, 42)
				defer index.Close()

				// Add vectors to index
				for i, vec := range vectors {
					if err := index.Add(vec, uint64(i)); err != nil {
						b.Fatal(err)
					}
				}

				// Set higher ef for better search quality
				index.SetEf(100)

				// Benchmark search
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					query := queries[i%len(queries)]
					_, _, _ = index.SearchKSimilarity(query, 10)
				}
			})
		}
	}
}

// Benchmark Naive Brute-Force Implementation
func BenchmarkNaiveCosine(b *testing.B) {
	dimensions := []int{50, 100, 300}
	datasetSizes := []int{1000, 5000, 10000}

	for _, dim := range dimensions {
		for _, size := range datasetSizes {
			b.Run(fmt.Sprintf("Naive_dim%d_size%d", dim, size), func(b *testing.B) {
				// Generate test data
				vectors := generateRandomVectors(size, dim, 42)
				queries := generateRandomVectors(100, dim, 123) // 100 query vectors

				// Setup naive implementation
				naive := NewNaiveCosineSimilarity()

				// Add vectors to naive implementation
				for i, vec := range vectors {
					naive.Add(vec, uint64(i))
				}

				// Benchmark search
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					query := queries[i%len(queries)]
					_, _, _ = naive.SearchK(query, 10)
				}
			})
		}
	}
}

// Accuracy test to ensure HNSW results are reasonable
func TestCosineSimilarityAccuracy(t *testing.T) {
	const (
		dim  = 100
		size = 1000
		k    = 10
	)

	// Generate test data
	vectors := generateRandomVectors(size, dim, 42)
	query := generateRandomVectors(1, dim, 123)[0]

	// Setup both implementations
	index := hnsw.NewCosine(dim, size*2, 32, 400, 42)
	defer index.Close()

	naive := NewNaiveCosineSimilarity()

	// Add same vectors to both
	for i, vec := range vectors {
		if err := index.Add(vec, uint64(i)); err != nil {
			t.Fatal(err)
		}
		naive.Add(vec, uint64(i))
	}

	// Set higher ef for better search quality
	index.SetEf(100)

	// Get results from both
	hnswLabels, hnswSims, hnswCount := index.SearchKSimilarity(query, k)
	naiveLabels, naiveSims, naiveCount := naive.SearchK(query, k)

	if hnswCount != k || naiveCount != k {
		t.Fatalf("Expected %d results, got HNSW: %d, Naive: %d", k, hnswCount, naiveCount)
	}

	// Calculate recall@k (how many of the top-k naive results appear in HNSW top-k)
	naiveSet := make(map[uint64]bool)
	for i := 0; i < naiveCount; i++ {
		naiveSet[naiveLabels[i]] = true
	}

	overlap := 0
	for i := 0; i < hnswCount; i++ {
		if naiveSet[hnswLabels[i]] {
			overlap++
		}
	}

	recall := float64(overlap) / float64(k)
	t.Logf("Recall@%d: %.2f%% (%d/%d matches)", k, recall*100, overlap, k)

	// Print top similarities for comparison
	t.Logf("HNSW top 5 similarities: %.4f, %.4f, %.4f, %.4f, %.4f",
		hnswSims[0], hnswSims[1], hnswSims[2], hnswSims[3], hnswSims[4])
	t.Logf("Naive top 5 similarities: %.4f, %.4f, %.4f, %.4f, %.4f",
		naiveSims[0], naiveSims[1], naiveSims[2], naiveSims[3], naiveSims[4])

	// HNSW should achieve reasonable recall (typically > 80% for these parameters)
	if recall < 0.7 {
		t.Errorf("Recall too low: %.2f%%, expected > 70%%", recall*100)
	}
}

// Performance comparison test
func TestPerformanceComparison(t *testing.T) {
	const (
		dim     = 100
		size    = 5000
		queries = 100
		k       = 10
	)

	// Generate test data
	vectors := generateRandomVectors(size, dim, 42)
	queryVectors := generateRandomVectors(queries, dim, 123)

	// Setup HNSW
	index := hnsw.NewCosine(dim, size*2, 32, 400, 42)
	defer index.Close()

	for i, vec := range vectors {
		if err := index.Add(vec, uint64(i)); err != nil {
			t.Fatal(err)
		}
	}

	// Set higher ef for better search quality
	index.SetEf(100)

	// Setup Naive
	naive := NewNaiveCosineSimilarity()
	for i, vec := range vectors {
		naive.Add(vec, uint64(i))
	}

	// Benchmark HNSW
	start := time.Now()
	for _, query := range queryVectors {
		_, _, _ = index.SearchKSimilarity(query, k)
	}
	hnswTime := time.Since(start)

	// Benchmark Naive
	start = time.Now()
	for _, query := range queryVectors {
		_, _, _ = naive.SearchK(query, k)
	}
	naiveTime := time.Since(start)

	speedup := float64(naiveTime) / float64(hnswTime)

	t.Logf("Performance Comparison (%d vectors, %d queries):", size, queries)
	t.Logf("HNSW time:  %v", hnswTime)
	t.Logf("Naive time: %v", naiveTime)
	t.Logf("Speedup:    %.1fx", speedup)

	if speedup < 2.0 {
		t.Logf("Warning: Expected speedup > 2x, got %.1fx", speedup)
	}
}
