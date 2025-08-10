package hnsw

import (
	"errors"
	"math"
	"runtime"

	"github.com/viktordanov/go-hnswlib/bindings/bindings"
)

type Space byte

const (
	SpaceL2     Space = 'l' // L2 (Euclidean) distance
	SpaceIP     Space = 'i' // Inner product
	SpaceCosine Space = 'c' // Cosine similarity
)

type Index struct {
	h         *bindings.HNSW
	normalize bool // true for cosine space
}

func New(space Space, dim, maxElements, M, efConstruction, seed int) *Index {
	idx := &Index{
		normalize: space == SpaceCosine,
	}
	idx.h = bindings.InitHNSW(int32(dim), uint64(maxElements), int32(M), int32(efConstruction), int32(seed), byte(space))
	runtime.SetFinalizer(idx, (*Index).Close)
	return idx
}

func Load(space Space, dim int, path string) (*Index, error) {
	pathBytes := []byte(path + "\x00") // null terminate
	h := bindings.LoadHNSWSafe(pathBytes, int32(dim), byte(space))
	if h == nil {
		return nil, errors.New("failed to load index (check file exists and is valid)")
	}
	idx := &Index{
		h:         h,
		normalize: space == SpaceCosine,
	}
	runtime.SetFinalizer(idx, (*Index).Close)
	return idx, nil
}

func (i *Index) Close() {
	if i == nil || i.h == nil {
		return
	}
	bindings.FreeHNSW(i.h)
	i.h = nil
	runtime.SetFinalizer(i, nil)
}

// normalizeVector normalizes a vector to unit length
// Creates a copy to avoid race conditions when used concurrently
func normalizeVector(vector []float32) []float32 {
	// Calculate norm
	var norm float32
	for i := 0; i < len(vector); i++ {
		norm += vector[i] * vector[i]
	}
	norm = 1.0 / (float32(math.Sqrt(float64(norm))) + 1e-15)

	// Create normalized copy (thread-safe)
	normalized := make([]float32, len(vector))
	for i := 0; i < len(vector); i++ {
		normalized[i] = vector[i] * norm
	}
	return normalized
}
func (i *Index) Add(vec []float32, label uint64) error {
	if i == nil || i.h == nil {
		return errors.New("index is closed")
	}

	// Check if index is at capacity
	currentCount := i.GetCurrentCount()
	maxElements := i.GetMaxElements()
	if currentCount >= maxElements {
		return errors.New("index is at maximum capacity - use Resize() to increase capacity")
	}

	// Normalize vector for cosine space
	vecToAdd := vec
	if i.normalize {
		vecToAdd = normalizeVector(vec)
	}

	result := bindings.AddPointSafe(i.h, vecToAdd, label)
	if result != 0 {
		return errors.New("failed to add point (check vector dimensions and label uniqueness)")
	}
	return nil
}

func (i *Index) SearchK(query []float32, k int) (labels []uint64, distances []float32, count int) {
	if i == nil || i.h == nil {
		return nil, nil, 0
	}

	// Normalize query vector for cosine space
	queryToSearch := query
	if i.normalize {
		queryToSearch = normalizeVector(query)
	}

	labels = make([]uint64, k)
	distances = make([]float32, k)
	count = int(bindings.SearchKnn(i.h, queryToSearch, int32(k), labels, distances))
	if count < k {
		labels = labels[:count]
		distances = distances[:count]
	}
	return labels, distances, count
}

// SearchKSimilarity searches for k nearest neighbors and returns cosine similarities (0-1)
// instead of distances when using cosine space. For other spaces, returns 1-distance.
func (i *Index) SearchKSimilarity(query []float32, k int) (labels []uint64, similarities []float32, count int) {
	labels, distances, count := i.SearchK(query, k)

	if count == 0 {
		return labels, nil, 0
	}

	similarities = make([]float32, count)
	if i.normalize {
		// For cosine space: similarity = 1 - distance
		// Since cosine distance = 1 - cosine_similarity
		for j := 0; j < count; j++ {
			similarities[j] = 1.0 - distances[j]
		}
	} else {
		// For other spaces, convert distance to similarity (arbitrary but consistent)
		// Using simple 1/(1+distance) transformation
		for j := 0; j < count; j++ {
			similarities[j] = 1.0 / (1.0 + distances[j])
		}
	}
	return labels, similarities, count
}

func (i *Index) SetEf(ef int) {
	if i == nil || i.h == nil {
		return
	}
	bindings.SetEf(i.h, int32(ef))
}

func (i *Index) Resize(newMaxElements int) error {
	if i == nil || i.h == nil {
		return errors.New("index is closed")
	}
	result := bindings.ResizeIndexSafe(i.h, uint64(newMaxElements))
	if result != 0 {
		return errors.New("failed to resize index (new size may be smaller than current count or memory allocation failed)")
	}
	return nil
}

func (i *Index) Save(path string) error {
	if i == nil || i.h == nil {
		return errors.New("index is closed")
	}
	pathBytes := []byte(path + "\x00") // null terminate
	result := bindings.SaveIndexSafe(i.h, pathBytes)
	if result != 0 {
		return errors.New("failed to save index (check file permissions and disk space)")
	}
	return nil
}

// Introspection functions
func (i *Index) GetCurrentCount() int {
	if i == nil || i.h == nil {
		return 0
	}
	return int(bindings.GetCurrentElementCount(i.h))
}

func (i *Index) GetMaxElements() int {
	if i == nil || i.h == nil {
		return 0
	}
	return int(bindings.GetMaxElements(i.h))
}

func (i *Index) GetDeletedCount() int {
	if i == nil || i.h == nil {
		return 0
	}
	return int(bindings.GetDeletedCount(i.h))
}

// IsCosineSpace returns true if this index uses cosine similarity
func (i *Index) IsCosineSpace() bool {
	return i.normalize
}

// Delete management functions
func (i *Index) MarkDeleted(label uint64) error {
	if i == nil || i.h == nil {
		return errors.New("index is closed")
	}
	result := bindings.MarkDeletedSafe(i.h, label)
	if result != 0 {
		return errors.New("failed to mark label as deleted (label may not exist)")
	}
	return nil
}

func (i *Index) UnmarkDeleted(label uint64) error {
	if i == nil || i.h == nil {
		return errors.New("index is closed")
	}
	result := bindings.UnmarkDeletedSafe(i.h, label)
	if result != 0 {
		return errors.New("failed to unmark label as deleted (label may not exist)")
	}
	return nil
}

// Helper constructors for common use cases
func NewL2(dim, maxElements, M, efConstruction, seed int) *Index {
	return New(SpaceL2, dim, maxElements, M, efConstruction, seed)
}

func NewIP(dim, maxElements, M, efConstruction, seed int) *Index {
	return New(SpaceIP, dim, maxElements, M, efConstruction, seed)
}

func NewCosine(dim, maxElements, M, efConstruction, seed int) *Index {
	return New(SpaceCosine, dim, maxElements, M, efConstruction, seed)
}
