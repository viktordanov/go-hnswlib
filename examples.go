package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/viktordanov/go-hnswlib/hnsw"
)

func main() {
	fmt.Println("HNSW Library Examples")
	fmt.Println("====================")

	// Run different examples
	exampleL2Space()
	exampleInnerProductSpace()
	exampleCosineSpace()
	exampleCapacityManagement()
	exampleSaveAndLoad()
	exampleDeleteManagement()
}

// Example 1: L2 (Euclidean) Distance Space
func exampleL2Space() {
	fmt.Println("\n1. L2 (Euclidean) Distance Example")
	fmt.Println("----------------------------------")

	// Create L2 index
	index := hnsw.NewL2(3, 1000, 16, 200, 42)
	defer index.Close()

	// Add some 3D points
	points := [][]float32{
		{0.0, 0.0, 0.0}, // Origin
		{1.0, 0.0, 0.0}, // Unit X
		{0.0, 1.0, 0.0}, // Unit Y
		{0.0, 0.0, 1.0}, // Unit Z
		{1.0, 1.0, 1.0}, // Corner
	}

	for i, point := range points {
		if err := index.Add(point, uint64(i+1)); err != nil {
			log.Printf("Failed to add point %d: %v", i+1, err)
		}
	}

	// Search near the origin
	query := []float32{0.1, 0.1, 0.1}
	labels, distances, count := index.SearchK(query, 3)

	fmt.Printf("Query: [%.1f, %.1f, %.1f]\n", query[0], query[1], query[2])
	fmt.Printf("Found %d neighbors:\n", count)
	for i := 0; i < count; i++ {
		fmt.Printf("  Label: %d, Distance: %.6f\n", labels[i], distances[i])
	}
}

// Example 2: Inner Product Space
func exampleInnerProductSpace() {
	fmt.Println("\n2. Inner Product Space Example")
	fmt.Println("------------------------------")

	index := hnsw.NewIP(4, 1000, 16, 200, 42)
	defer index.Close()

	// Add some feature vectors (like word embeddings)
	embeddings := map[string][]float32{
		"cat":   {0.2, 0.8, 0.1, 0.3},
		"dog":   {0.3, 0.7, 0.2, 0.4}, // Similar to cat
		"car":   {0.8, 0.1, 0.6, 0.2},
		"bike":  {0.7, 0.2, 0.7, 0.1}, // Similar to car
		"apple": {0.1, 0.3, 0.2, 0.9},
	}

	labelMap := make(map[uint64]string)
	label := uint64(1)
	for word, embedding := range embeddings {
		if err := index.Add(embedding, label); err != nil {
			log.Printf("Failed to add %s: %v", word, err)
		}
		labelMap[label] = word
		label++
	}

	// Search for something similar to "cat"
	query := embeddings["cat"]
	labels, distances, count := index.SearchK(query, 3)

	fmt.Printf("Query: '%s' [%.1f, %.1f, %.1f, %.1f]\n", "cat", query[0], query[1], query[2], query[3])
	fmt.Printf("Found %d similar items:\n", count)
	for i := 0; i < count; i++ {
		word := labelMap[labels[i]]
		fmt.Printf("  '%s' (Label: %d), Inner Product: %.6f\n", word, labels[i], distances[i])
	}
}

// Example 3: Cosine Similarity Space
func exampleCosineSpace() {
	fmt.Println("\n3. Cosine Similarity Space Example")
	fmt.Println("----------------------------------")

	index := hnsw.NewCosine(3, 1000, 16, 200, 42)
	defer index.Close()

	fmt.Printf("Is cosine space: %v\n", index.IsCosineSpace())

	// Add directional vectors (different magnitudes, same directions)
	directions := [][]float32{
		{1.0, 0.0, 0.0},  // East (small)
		{10.0, 0.0, 0.0}, // East (large) - same direction as above
		{0.0, 1.0, 0.0},  // North
		{1.0, 1.0, 0.0},  // Northeast
		{-1.0, 0.0, 0.0}, // West (opposite to East)
	}

	for i, dir := range directions {
		if err := index.Add(dir, uint64(i+1)); err != nil {
			log.Printf("Failed to add direction %d: %v", i+1, err)
		}
	}

	// Query with East direction (different magnitude)
	query := []float32{5.0, 0.0, 0.0}

	// Compare distances vs similarities
	labels1, distances, count1 := index.SearchK(query, 4)
	labels2, similarities, count2 := index.SearchKSimilarity(query, 4)

	fmt.Printf("Query: [%.1f, %.1f, %.1f] (East direction)\n", query[0], query[1], query[2])

	fmt.Println("Distances:")
	for i := 0; i < count1; i++ {
		fmt.Printf("  Label: %d, Distance: %.6f\n", labels1[i], distances[i])
	}

	fmt.Println("Similarities:")
	for i := 0; i < count2; i++ {
		fmt.Printf("  Label: %d, Similarity: %.6f\n", labels2[i], similarities[i])
	}
}

// Example 4: Capacity Management
func exampleCapacityManagement() {
	fmt.Println("\n4. Capacity Management Example")
	fmt.Println("------------------------------")

	// Create small index to demonstrate capacity
	index := hnsw.NewL2(2, 3, 16, 200, 42) // Only 3 elements max
	defer index.Close()

	fmt.Printf("Initial capacity: %d\n", index.GetMaxElements())
	fmt.Printf("Current count: %d\n", index.GetCurrentCount())

	// Add vectors until capacity is reached
	for i := 0; i < 5; i++ {
		vec := []float32{float32(i), float32(i + 1)}
		err := index.Add(vec, uint64(i+1))

		if err != nil {
			fmt.Printf("Add vector %d: %v\n", i+1, err)
			if i == 3 { // Try to resize after hitting capacity
				fmt.Println("Resizing index to capacity 10...")
				if resizeErr := index.Resize(10); resizeErr != nil {
					log.Printf("Resize failed: %v", resizeErr)
				} else {
					fmt.Printf("New capacity: %d\n", index.GetMaxElements())
					// Retry adding
					if retryErr := index.Add(vec, uint64(i+1)); retryErr != nil {
						fmt.Printf("Retry add failed: %v\n", retryErr)
					} else {
						fmt.Printf("Successfully added vector %d after resize\n", i+1)
					}
				}
			}
		} else {
			fmt.Printf("Added vector %d successfully\n", i+1)
		}

		fmt.Printf("Current count: %d\n", index.GetCurrentCount())
	}
}

// Example 5: Save and Load
func exampleSaveAndLoad() {
	fmt.Println("\n5. Save and Load Example")
	fmt.Println("------------------------")

	// Create and populate an index
	original := hnsw.NewCosine(3, 100, 16, 200, 42)

	// Add some data
	for i := 0; i < 10; i++ {
		vec := []float32{
			rand.Float32() - 0.5,
			rand.Float32() - 0.5,
			rand.Float32() - 0.5,
		}
		if err := original.Add(vec, uint64(i+1)); err != nil {
			log.Printf("Failed to add vector %d: %v", i+1, err)
		}
	}

	fmt.Printf("Original index: %d elements\n", original.GetCurrentCount())

	// Save to file
	filename := "/tmp/test_index.hnsw"
	if err := original.Save(filename); err != nil {
		log.Printf("Failed to save: %v", err)
		original.Close()
		return
	}
	fmt.Printf("Saved index to %s\n", filename)
	original.Close()

	// Load from file
	loaded, err := hnsw.Load(hnsw.SpaceCosine, 3, filename)
	if err != nil {
		log.Printf("Failed to load: %v", err)
		return
	}
	defer loaded.Close()

	fmt.Printf("Loaded index: %d elements\n", loaded.GetCurrentCount())
	fmt.Printf("Is cosine space: %v\n", loaded.IsCosineSpace())

	// Test search on loaded index
	query := []float32{0.1, 0.2, 0.3}
	labels, similarities, count := loaded.SearchKSimilarity(query, 3)
	fmt.Printf("Search results: found %d neighbors\n", count)
	for i := 0; i < count; i++ {
		fmt.Printf("  Label: %d, Similarity: %.6f\n", labels[i], similarities[i])
	}
}

// Example 6: Delete Management
func exampleDeleteManagement() {
	fmt.Println("\n6. Delete Management Example")
	fmt.Println("----------------------------")

	index := hnsw.NewL2(2, 100, 16, 200, 42)
	defer index.Close()

	// Add some vectors
	vectors := [][]float32{
		{1.0, 1.0}, // Label 1
		{2.0, 2.0}, // Label 2
		{3.0, 3.0}, // Label 3
		{4.0, 4.0}, // Label 4
		{5.0, 5.0}, // Label 5
	}

	for i, vec := range vectors {
		if err := index.Add(vec, uint64(i+1)); err != nil {
			log.Printf("Failed to add vector %d: %v", i+1, err)
		}
	}

	fmt.Printf("Total elements: %d\n", index.GetCurrentCount())
	fmt.Printf("Deleted elements: %d\n", index.GetDeletedCount())

	// Search before deletion
	query := []float32{2.5, 2.5}
	labels, distances, count := index.SearchK(query, 3)
	fmt.Printf("\nBefore deletion - found %d neighbors:\n", count)
	for i := 0; i < count; i++ {
		fmt.Printf("  Label: %d, Distance: %.6f\n", labels[i], distances[i])
	}

	// Delete label 2
	if err := index.MarkDeleted(2); err != nil {
		log.Printf("Failed to mark deleted: %v", err)
	} else {
		fmt.Println("\nMarked label 2 as deleted")
	}

	fmt.Printf("Deleted elements: %d\n", index.GetDeletedCount())

	// Search after deletion
	labels, distances, count = index.SearchK(query, 3)
	fmt.Printf("\nAfter deletion - found %d neighbors:\n", count)
	for i := 0; i < count; i++ {
		fmt.Printf("  Label: %d, Distance: %.6f\n", labels[i], distances[i])
	}

	// Restore the deleted element
	if err := index.UnmarkDeleted(2); err != nil {
		log.Printf("Failed to unmark deleted: %v", err)
	} else {
		fmt.Println("\nRestored label 2")
	}

	fmt.Printf("Deleted elements: %d\n", index.GetDeletedCount())

	// Search after restoration
	labels, distances, count = index.SearchK(query, 3)
	fmt.Printf("\nAfter restoration - found %d neighbors:\n", count)
	for i := 0; i < count; i++ {
		fmt.Printf("  Label: %d, Distance: %.6f\n", labels[i], distances[i])
	}

	fmt.Println("\n✅ All examples completed successfully!")
}
