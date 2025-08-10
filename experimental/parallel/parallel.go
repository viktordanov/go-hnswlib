package parallel

import (
	"errors"
	"runtime"
	"sync"

	"github.com/viktordanov/go-hnswlib/hnsw"
)

type VectorData struct {
	Vector []float32
	Label  uint64
}

type BatchAddOptions struct {
	MaxWorkers int
}

type ParallelSearchOptions struct {
	MaxWorkers int
}

type SearchResult struct {
	QueryIndex int
	Labels     []uint64
	Distances  []float32
}

type SearchSimilarityResult struct {
	QueryIndex   int
	Labels       []uint64
	Similarities []float32
}

func BatchAdd(index *hnsw.Index, vectors []VectorData, options *BatchAddOptions) error {
	if index == nil {
		return errors.New("index cannot be nil")
	}

	maxWorkers := runtime.GOMAXPROCS(0)
	if options != nil && options.MaxWorkers > 0 {
		maxWorkers = options.MaxWorkers
	}

	jobs := make(chan VectorData, len(vectors))
	results := make(chan error, len(vectors))

	var wg sync.WaitGroup
	for w := 0; w < maxWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for vector := range jobs {
				err := index.Add(vector.Vector, vector.Label)
				results <- err
			}
		}()
	}

	for _, vector := range vectors {
		jobs <- vector
	}
	close(jobs)

	wg.Wait()
	close(results)

	for err := range results {
		if err != nil {
			return err
		}
	}

	return nil
}

func ParallelSearch(index *hnsw.Index, queries [][]float32, k int, options *ParallelSearchOptions) ([]SearchResult, error) {
	if index == nil {
		return nil, errors.New("index cannot be nil")
	}

	maxWorkers := runtime.GOMAXPROCS(0)
	if options != nil && options.MaxWorkers > 0 {
		maxWorkers = options.MaxWorkers
	}

	type job struct {
		query []float32
		index int
	}

	jobs := make(chan job, len(queries))
	results := make(chan SearchResult, len(queries))

	var wg sync.WaitGroup
	for w := 0; w < maxWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				labels, distances, _ := index.SearchK(j.query, k)
				results <- SearchResult{
					QueryIndex: j.index,
					Labels:     labels,
					Distances:  distances,
				}
			}
		}()
	}

	for i, query := range queries {
		jobs <- job{query: query, index: i}
	}
	close(jobs)

	wg.Wait()
	close(results)

	searchResults := make([]SearchResult, len(queries))
	for result := range results {
		searchResults[result.QueryIndex] = result
	}

	return searchResults, nil
}

func ParallelSearchSimilarity(index *hnsw.Index, queries [][]float32, k int, options *ParallelSearchOptions) ([]SearchSimilarityResult, error) {
	if index == nil {
		return nil, errors.New("index cannot be nil")
	}

	maxWorkers := runtime.GOMAXPROCS(0)
	if options != nil && options.MaxWorkers > 0 {
		maxWorkers = options.MaxWorkers
	}

	type job struct {
		query []float32
		index int
	}

	jobs := make(chan job, len(queries))
	results := make(chan SearchSimilarityResult, len(queries))

	var wg sync.WaitGroup
	for w := 0; w < maxWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				labels, similarities, _ := index.SearchKSimilarity(j.query, k)
				results <- SearchSimilarityResult{
					QueryIndex:   j.index,
					Labels:       labels,
					Similarities: similarities,
				}
			}
		}()
	}

	for i, query := range queries {
		jobs <- job{query: query, index: i}
	}
	close(jobs)

	wg.Wait()
	close(results)

	searchResults := make([]SearchSimilarityResult, len(queries))
	for result := range results {
		searchResults[result.QueryIndex] = result
	}

	return searchResults, nil
}

func BatchSearchK(index *hnsw.Index, queries [][]float32, k int, options *ParallelSearchOptions) ([][]uint64, error) {
	results, err := ParallelSearch(index, queries, k, options)
	if err != nil {
		return nil, err
	}

	labels := make([][]uint64, len(results))
	for i, result := range results {
		labels[i] = result.Labels
	}

	return labels, nil
}

func BatchSearchKSimilarity(index *hnsw.Index, queries [][]float32, k int, options *ParallelSearchOptions) ([][]uint64, error) {
	results, err := ParallelSearchSimilarity(index, queries, k, options)
	if err != nil {
		return nil, err
	}

	labels := make([][]uint64, len(results))
	for i, result := range results {
		labels[i] = result.Labels
	}

	return labels, nil
}
