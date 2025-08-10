package main

import (
	"math"
	"sort"
)

// Naive brute-force cosine similarity implementation
type NaiveCosineSimilarity struct {
	vectors [][]float32
	labels  []uint64
}

func NewNaiveCosineSimilarity() *NaiveCosineSimilarity {
	return &NaiveCosineSimilarity{
		vectors: make([][]float32, 0),
		labels:  make([]uint64, 0),
	}
}

func (n *NaiveCosineSimilarity) Add(vec []float32, label uint64) {
	// Normalize the vector for cosine similarity
	normalized := normalizeVec(vec)
	n.vectors = append(n.vectors, normalized)
	n.labels = append(n.labels, label)
}

func (n *NaiveCosineSimilarity) SearchK(query []float32, k int) ([]uint64, []float32, int) {
	if len(n.vectors) == 0 {
		return nil, nil, 0
	}

	// Normalize query vector
	normalizedQuery := normalizeVec(query)

	type result struct {
		label      uint64
		similarity float32
	}

	results := make([]result, 0, len(n.vectors))

	// Brute force: calculate similarity with all vectors
	for i, vec := range n.vectors {
		similarity := cosineSimilarity(normalizedQuery, vec)
		results = append(results, result{
			label:      n.labels[i],
			similarity: similarity,
		})
	}

	// Sort by similarity (descending - higher is better)
	sort.Slice(results, func(i, j int) bool {
		return results[i].similarity > results[j].similarity
	})

	// Return top k results
	count := k
	if count > len(results) {
		count = len(results)
	}

	labels := make([]uint64, count)
	similarities := make([]float32, count)
	for i := 0; i < count; i++ {
		labels[i] = results[i].label
		similarities[i] = results[i].similarity
	}

	return labels, similarities, count
}

// Helper functions
func normalizeVec(vec []float32) []float32 {
	var norm float32
	for _, v := range vec {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))

	if norm == 0 {
		return vec
	}

	normalized := make([]float32, len(vec))
	for i, v := range vec {
		normalized[i] = v / norm
	}
	return normalized
}

func cosineSimilarity(a, b []float32) float32 {
	var dot float32
	for i := range a {
		dot += a[i] * b[i]
	}
	return dot // Already normalized, so this is the cosine similarity
}
