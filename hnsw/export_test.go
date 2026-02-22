package hnsw_test

import (
	"errors"
	"math"
	"testing"

	"github.com/viktordanov/go-hnswlib/hnsw"
)

func TestGetDimension(t *testing.T) {
	tests := []struct {
		name string
		dim  int
	}{
		{"dim 32", 32},
		{"dim 128", 128},
		{"dim 768", 768},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			index := hnsw.NewCosine(tt.dim, 100, 16, 200, 42)
			defer index.Close()

			if dim := index.GetDimension(); dim != tt.dim {
				t.Errorf("expected dimension %d, got %d", tt.dim, dim)
			}
		})
	}
}

func TestGetVector(t *testing.T) {
	index := hnsw.NewCosine(64, 100, 16, 200, 42)
	defer index.Close()

	// Add a vector
	vec := make([]float32, 64)
	for i := range vec {
		vec[i] = float32(i) + 0.5
	}
	if err := index.Add(vec, 42); err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// Retrieve it
	retrieved, err := index.GetVector(42)
	if err != nil {
		t.Fatalf("GetVector failed: %v", err)
	}

	if len(retrieved) != 64 {
		t.Errorf("expected 64 dimensions, got %d", len(retrieved))
	}

	// For cosine space, vectors are normalized - verify it's a unit vector
	var norm float64
	for _, v := range retrieved {
		norm += float64(v) * float64(v)
	}
	norm = math.Sqrt(norm)

	if norm < 0.99 || norm > 1.01 {
		t.Errorf("expected unit vector (norm ~1.0), got norm %f", norm)
	}
}

func TestGetVectorNotFound(t *testing.T) {
	index := hnsw.NewCosine(32, 100, 16, 200, 42)
	defer index.Close()

	// Add a vector
	vec := make([]float32, 32)
	for i := range vec {
		vec[i] = float32(i)
	}
	index.Add(vec, 1)

	// Try to get non-existent label
	_, err := index.GetVector(999)
	if err == nil {
		t.Error("expected error for non-existent label, got nil")
	}
}

func TestGetVectorDeleted(t *testing.T) {
	index := hnsw.NewL2(32, 100, 16, 200, 42)
	defer index.Close()

	// Add a vector
	vec := make([]float32, 32)
	for i := range vec {
		vec[i] = float32(i)
	}
	index.Add(vec, 1)

	// Delete it
	if err := index.MarkDeleted(1); err != nil {
		t.Fatalf("MarkDeleted failed: %v", err)
	}

	// Try to get deleted label
	_, err := index.GetVector(1)
	if err == nil {
		t.Error("expected error for deleted label, got nil")
	}
}

func TestGetVectors(t *testing.T) {
	index := hnsw.NewL2(16, 100, 16, 200, 42)
	defer index.Close()

	// Add vectors
	for i := 0; i < 10; i++ {
		vec := make([]float32, 16)
		for j := range vec {
			vec[j] = float32(i*16 + j)
		}
		index.Add(vec, uint64(i))
	}

	// Get multiple vectors
	result, err := index.GetVectors([]uint64{0, 2, 4, 6, 8, 999})
	if err != nil {
		t.Fatalf("GetVectors failed: %v", err)
	}

	// Should have 5 vectors (999 doesn't exist)
	if len(result) != 5 {
		t.Errorf("expected 5 vectors, got %d", len(result))
	}

	// Verify expected labels are present
	for _, label := range []uint64{0, 2, 4, 6, 8} {
		if _, ok := result[label]; !ok {
			t.Errorf("missing label %d", label)
		}
	}

	// 999 should not be present
	if _, ok := result[999]; ok {
		t.Error("label 999 should not be present")
	}
}

func TestIterator(t *testing.T) {
	index := hnsw.NewCosine(32, 100, 16, 200, 42)
	defer index.Close()

	// Add 50 vectors
	for i := 0; i < 50; i++ {
		vec := make([]float32, 32)
		for j := range vec {
			vec[j] = float32(i*32 + j)
		}
		index.Add(vec, uint64(i*10)) // Labels: 0, 10, 20, ...
	}

	// Iterate
	iter, err := index.NewIterator()
	if err != nil {
		t.Fatalf("NewIterator failed: %v", err)
	}

	count := 0
	labels := make(map[uint64]bool)

	for iter.Next() {
		elem, err := iter.Element()
		if err != nil {
			t.Fatalf("Element() failed: %v", err)
		}

		labels[elem.Label] = true

		vec, err := iter.Vector()
		if err != nil {
			t.Fatalf("Vector() failed: %v", err)
		}

		if len(vec) != 32 {
			t.Errorf("expected 32 dimensions, got %d", len(vec))
		}

		count++
		iter.Advance()
	}

	if count != 50 {
		t.Errorf("expected 50 elements, got %d", count)
	}

	// Verify all expected labels present
	for i := 0; i < 50; i++ {
		if !labels[uint64(i*10)] {
			t.Errorf("missing label %d", i*10)
		}
	}
}

func TestIteratorProgress(t *testing.T) {
	index := hnsw.NewL2(16, 100, 16, 200, 42)
	defer index.Close()

	// Add 20 vectors
	for i := 0; i < 20; i++ {
		vec := make([]float32, 16)
		for j := range vec {
			vec[j] = float32(i + j)
		}
		index.Add(vec, uint64(i))
	}

	iter, err := index.NewIterator()
	if err != nil {
		t.Fatalf("NewIterator failed: %v", err)
	}

	// Check initial progress
	current, total := iter.Progress()
	if current != 0 || total != 20 {
		t.Errorf("expected progress (0, 20), got (%d, %d)", current, total)
	}

	// Iterate halfway
	for i := 0; i < 10; i++ {
		if !iter.Next() {
			t.Fatal("unexpected end of iteration")
		}
		iter.Advance()
	}

	current, total = iter.Progress()
	if current != 10 || total != 20 {
		t.Errorf("expected progress (10, 20), got (%d, %d)", current, total)
	}

	// Reset
	iter.Reset()
	current, total = iter.Progress()
	if current != 0 {
		t.Errorf("expected current=0 after reset, got %d", current)
	}
}

func TestIteratorWithDeleted(t *testing.T) {
	index := hnsw.NewL2(16, 100, 16, 200, 42)
	defer index.Close()

	// Add vectors
	for i := 0; i < 10; i++ {
		vec := make([]float32, 16)
		for j := range vec {
			vec[j] = float32(i + j)
		}
		index.Add(vec, uint64(i))
	}

	// Delete some
	index.MarkDeleted(3)
	index.MarkDeleted(7)

	iter, err := index.NewIterator()
	if err != nil {
		t.Fatalf("NewIterator failed: %v", err)
	}

	deletedCount := 0
	activeCount := 0

	for iter.Next() {
		elem, _ := iter.Element()
		if elem.IsDeleted {
			deletedCount++
		} else {
			activeCount++
		}
		iter.Advance()
	}

	if deletedCount != 2 {
		t.Errorf("expected 2 deleted, got %d", deletedCount)
	}
	if activeCount != 8 {
		t.Errorf("expected 8 active, got %d", activeCount)
	}
}

func TestExport(t *testing.T) {
	index := hnsw.NewL2(16, 100, 16, 200, 42)
	defer index.Close()

	// Add vectors, delete some
	for i := 0; i < 20; i++ {
		vec := make([]float32, 16)
		for j := range vec {
			vec[j] = float32(i + j)
		}
		index.Add(vec, uint64(i))
	}

	// Delete labels 5, 10, 15
	index.MarkDeleted(5)
	index.MarkDeleted(10)
	index.MarkDeleted(15)

	// Export excluding deleted
	activeCount := 0
	err := index.Export(false, func(label uint64, vector []float32, isDeleted bool) error {
		if isDeleted {
			t.Error("received deleted element when includeDeleted=false")
		}
		activeCount++
		return nil
	})
	if err != nil {
		t.Fatalf("Export failed: %v", err)
	}

	if activeCount != 17 {
		t.Errorf("expected 17 active elements, got %d", activeCount)
	}

	// Export including deleted
	totalCount := 0
	deletedCount := 0
	err = index.Export(true, func(label uint64, vector []float32, isDeleted bool) error {
		totalCount++
		if isDeleted {
			deletedCount++
		}
		return nil
	})
	if err != nil {
		t.Fatalf("Export failed: %v", err)
	}

	if totalCount != 20 {
		t.Errorf("expected 20 total elements, got %d", totalCount)
	}
	if deletedCount != 3 {
		t.Errorf("expected 3 deleted elements, got %d", deletedCount)
	}
}

func TestExportEarlyStop(t *testing.T) {
	index := hnsw.NewL2(16, 100, 16, 200, 42)
	defer index.Close()

	// Add vectors
	for i := 0; i < 100; i++ {
		vec := make([]float32, 16)
		for j := range vec {
			vec[j] = float32(i + j)
		}
		index.Add(vec, uint64(i))
	}

	// Export with early stop after 10 elements
	stopErr := errors.New("stop")
	count := 0
	err := index.Export(false, func(label uint64, vector []float32, isDeleted bool) error {
		count++
		if count >= 10 {
			return stopErr
		}
		return nil
	})

	if err != stopErr {
		t.Errorf("expected stopErr, got %v", err)
	}
	if count != 10 {
		t.Errorf("expected 10 elements before stop, got %d", count)
	}
}

func TestExportL2VsCosineFidelity(t *testing.T) {
	// L2 space should preserve exact vectors
	indexL2 := hnsw.NewL2(8, 100, 16, 200, 42)
	defer indexL2.Close()

	original := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
	indexL2.Add(original, 1)

	retrieved, err := indexL2.GetVector(1)
	if err != nil {
		t.Fatalf("GetVector failed: %v", err)
	}

	// L2 should preserve exact values
	for i := range original {
		if math.Abs(float64(retrieved[i]-original[i])) > 1e-6 {
			t.Errorf("L2: expected %f at index %d, got %f", original[i], i, retrieved[i])
		}
	}

	// Cosine space normalizes vectors
	indexCosine := hnsw.NewCosine(8, 100, 16, 200, 42)
	defer indexCosine.Close()

	indexCosine.Add(original, 1)

	retrievedCosine, err := indexCosine.GetVector(1)
	if err != nil {
		t.Fatalf("GetVector failed: %v", err)
	}

	// Cosine should return unit vector
	var norm float64
	for _, v := range retrievedCosine {
		norm += float64(v) * float64(v)
	}
	norm = math.Sqrt(norm)

	if math.Abs(norm-1.0) > 1e-6 {
		t.Errorf("Cosine: expected unit vector, got norm %f", norm)
	}
}
