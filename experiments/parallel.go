package main

import (
	"errors"
	"runtime"
	"sync"

	"github.com/viktordanov/go-hnswlib/hnsw"
)

// ==============================================================================
// EXPERIMENTAL: Parallel and Batch Operations for HNSW
// ==============================================================================

// VectorData represents a vector with its label for batch operations
type VectorData struct {
	Vector []float32
	Label  uint64
}

// BatchAddOptions configures batch add operations
type BatchAddOptions struct {
	// MaxWorkers limits the number of concurrent goroutines (default: runtime.GOMAXPROCS(0))
	MaxWorkers int
	// ChunkSize controls how many vectors each worker processes at once (default: 100)
	ChunkSize int
}

// ParallelSearchOptions configures parallel search operations
type ParallelSearchOptions struct {
	// MaxWorkers limits the number of concurrent goroutines (default: runtime.GOMAXPROCS(0))
	MaxWorkers int
}

// SearchResult represents the result of a single search query
type SearchResult struct {
	Labels     []uint64
	Distances  []float32
	Count      int
	QueryIndex int // Index of the original query
}

// SearchSimilarityResult represents the result of a single similarity search query
type SearchSimilarityResult struct {
	Labels       []uint64
	Similarities []float32
	Count        int
	QueryIndex   int // Index of the original query
}

// BatchAdd adds multiple vectors to the index in parallel.
// This is significantly faster than calling Add() sequentially for large batches.
func BatchAdd(index *hnsw.Index, vectors []VectorData, opts *BatchAddOptions) error {
	if index == nil {
		return errors.New("index is nil")
	}

	if len(vectors) == 0 {
		return nil
	}

	// Set default options
	if opts == nil {
		opts = &BatchAddOptions{}
	}
	if opts.MaxWorkers <= 0 {
		opts.MaxWorkers = runtime.GOMAXPROCS(0)
	}
	if opts.ChunkSize <= 0 {
		opts.ChunkSize = 100
	}

	// Check capacity upfront
	currentCount := index.GetCurrentCount()
	maxElements := index.GetMaxElements()
	if currentCount+len(vectors) > maxElements {
		return errors.New("batch would exceed index capacity - use Resize() to increase capacity")
	}

	// Create work channel
	workCh := make(chan VectorData, len(vectors))
	errorCh := make(chan error, opts.MaxWorkers)

	// Start workers
	var wg sync.WaitGroup
	for w := 0; w < opts.MaxWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for vector := range workCh {
				if err := index.Add(vector.Vector, vector.Label); err != nil {
					select {
					case errorCh <- err:
					default:
						// Error channel is full, continue to avoid blocking
					}
					return
				}
			}
		}()
	}

	// Send work
	go func() {
		defer close(workCh)
		for _, vector := range vectors {
			workCh <- vector
		}
	}()

	// Wait for completion
	wg.Wait()
	close(errorCh)

	// Check for errors
	select {
	case err := <-errorCh:
		return err
	default:
		return nil
	}
}

// ParallelSearch performs multiple search queries concurrently.
// Returns results in the same order as the input queries.
func ParallelSearch(index *hnsw.Index, queries [][]float32, k int, opts *ParallelSearchOptions) ([]SearchResult, error) {
	if index == nil {
		return nil, errors.New("index is nil")
	}

	if len(queries) == 0 {
		return nil, nil
	}

	// Set default options
	if opts == nil {
		opts = &ParallelSearchOptions{}
	}
	if opts.MaxWorkers <= 0 {
		opts.MaxWorkers = runtime.GOMAXPROCS(0)
	}

	results := make([]SearchResult, len(queries))
	workCh := make(chan int, len(queries))

	// Start workers
	var wg sync.WaitGroup
	for w := 0; w < opts.MaxWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for queryIdx := range workCh {
				labels, distances, count := index.SearchK(queries[queryIdx], k)
				results[queryIdx] = SearchResult{
					Labels:     labels,
					Distances:  distances,
					Count:      count,
					QueryIndex: queryIdx,
				}
			}
		}()
	}

	// Send work
	go func() {
		defer close(workCh)
		for idx := range queries {
			workCh <- idx
		}
	}()

	// Wait for completion
	wg.Wait()
	return results, nil
}

// ParallelSearchSimilarity performs multiple similarity search queries concurrently.
// Returns results in the same order as the input queries.
func ParallelSearchSimilarity(index *hnsw.Index, queries [][]float32, k int, opts *ParallelSearchOptions) ([]SearchSimilarityResult, error) {
	if index == nil {
		return nil, errors.New("index is nil")
	}

	if len(queries) == 0 {
		return nil, nil
	}

	// Set default options
	if opts == nil {
		opts = &ParallelSearchOptions{}
	}
	if opts.MaxWorkers <= 0 {
		opts.MaxWorkers = runtime.GOMAXPROCS(0)
	}

	results := make([]SearchSimilarityResult, len(queries))
	workCh := make(chan int, len(queries))

	// Start workers
	var wg sync.WaitGroup
	for w := 0; w < opts.MaxWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for queryIdx := range workCh {
				labels, similarities, count := index.SearchKSimilarity(queries[queryIdx], k)
				results[queryIdx] = SearchSimilarityResult{
					Labels:       labels,
					Similarities: similarities,
					Count:        count,
					QueryIndex:   queryIdx,
				}
			}
		}()
	}

	// Send work
	go func() {
		defer close(workCh)
		for idx := range queries {
			workCh <- idx
		}
	}()

	// Wait for completion
	wg.Wait()
	return results, nil
}

// BatchSearchK is a convenience function that performs parallel search and returns
// just the labels and distances in the same format as SearchK but for multiple queries.
func BatchSearchK(index *hnsw.Index, queries [][]float32, k int, opts *ParallelSearchOptions) ([][]uint64, [][]float32, []int, error) {
	results, err := ParallelSearch(index, queries, k, opts)
	if err != nil {
		return nil, nil, nil, err
	}

	labels := make([][]uint64, len(results))
	distances := make([][]float32, len(results))
	counts := make([]int, len(results))

	for i, result := range results {
		labels[i] = result.Labels
		distances[i] = result.Distances
		counts[i] = result.Count
	}

	return labels, distances, counts, nil
}

// BatchSearchKSimilarity is a convenience function that performs parallel similarity search and returns
// just the labels and similarities in the same format as SearchKSimilarity but for multiple queries.
func BatchSearchKSimilarity(index *hnsw.Index, queries [][]float32, k int, opts *ParallelSearchOptions) ([][]uint64, [][]float32, []int, error) {
	results, err := ParallelSearchSimilarity(index, queries, k, opts)
	if err != nil {
		return nil, nil, nil, err
	}

	labels := make([][]uint64, len(results))
	similarities := make([][]float32, len(results))
	counts := make([]int, len(results))

	for i, result := range results {
		labels[i] = result.Labels
		similarities[i] = result.Similarities
		counts[i] = result.Count
	}

	return labels, similarities, counts, nil
}
