package main

import (
	"fmt"
	"math/rand"

	"github.com/vikimaster2/go-hnswlib/hnsw"
)

func main() {
	// Create a new L2 index
	dim := 128
	maxElements := 1000
	M := 16
	efConstruction := 200
	seed := 42

	fmt.Println("Creating HNSW index...")
	index := hnsw.NewL2(dim, maxElements, M, efConstruction, seed)
	defer index.Close()

	// Add some random vectors
	fmt.Println("Adding vectors...")
	for i := 0; i < 100; i++ {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rand.Float32()
		}
		index.Add(vec, uint64(i))
	}

	// Search for nearest neighbors
	fmt.Println("Searching for nearest neighbors...")
	query := make([]float32, dim)
	for j := range query {
		query[j] = rand.Float32()
	}

	labels, distances, count := index.SearchK(query, 5)
	fmt.Printf("Found %d nearest neighbors:\n", count)
	for i := 0; i < count; i++ {
		fmt.Printf("  Label: %d, Distance: %f\n", labels[i], distances[i])
	}

	fmt.Println("HNSW wrapper working successfully!")
}
