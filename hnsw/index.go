package hnsw

import (
	"errors"
	"math"
	"runtime"

	bindings "github.com/viktordanov/go-hnswlib"
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

// =============================================================================
// Vector Export API - For data migration and inspection
// =============================================================================

// GetDimension returns the vector dimension of this index
func (i *Index) GetDimension() int {
	if i == nil || i.h == nil {
		return 0
	}
	return int(bindings.GetDimension(i.h))
}

// GetVector retrieves the stored vector for a given label.
// For cosine indices, this returns the normalized vector (not the original).
// Returns error if label not found, is deleted, or index is closed.
func (i *Index) GetVector(label uint64) ([]float32, error) {
	if i == nil || i.h == nil {
		return nil, errors.New("index is closed")
	}

	dim := i.GetDimension()
	if dim <= 0 {
		return nil, errors.New("invalid dimension")
	}

	vector := make([]float32, dim)
	result := bindings.GetVectorByLabel(i.h, label, vector)
	if result < 0 {
		return nil, errors.New("label not found or deleted")
	}

	return vector, nil
}

// GetVectors retrieves multiple vectors by their labels.
// Returns a map of label -> vector for all found labels.
// Labels that are not found or deleted are omitted from the result.
// For cosine indices, vectors are normalized.
func (i *Index) GetVectors(labels []uint64) (map[uint64][]float32, error) {
	if i == nil || i.h == nil {
		return nil, errors.New("index is closed")
	}

	dim := i.GetDimension()
	if dim <= 0 {
		return nil, errors.New("invalid dimension")
	}

	result := make(map[uint64][]float32, len(labels))
	for _, label := range labels {
		vector := make([]float32, dim)
		if bindings.GetVectorByLabel(i.h, label, vector) >= 0 {
			result[label] = vector
		}
	}

	return result, nil
}

// ElementInfo contains information about a single stored element
type ElementInfo struct {
	InternalID uint64
	Label      uint64
	IsDeleted  bool
}

// VectorIterator provides memory-efficient iteration over all vectors in the index.
// The iterator is NOT thread-safe - don't modify the index while iterating.
type VectorIterator struct {
	index      *Index
	currentID  uint64
	totalCount uint64
	dimension  int
}

// NewIterator creates an iterator for all elements in the index.
// The iterator allows memory-efficient streaming of all vectors.
// WARNING: The iterator is NOT thread-safe - don't modify the index while iterating.
func (i *Index) NewIterator() (*VectorIterator, error) {
	if i == nil || i.h == nil {
		return nil, errors.New("index is closed")
	}

	return &VectorIterator{
		index:      i,
		currentID:  0,
		totalCount: uint64(i.GetCurrentCount()),
		dimension:  i.GetDimension(),
	}, nil
}

// Next returns true if there are more elements to iterate.
func (it *VectorIterator) Next() bool {
	return it.currentID < it.totalCount
}

// Element returns info about the current element (without vector data).
// Call Vector() to get the actual vector data.
func (it *VectorIterator) Element() (ElementInfo, error) {
	if it.index == nil || it.index.h == nil {
		return ElementInfo{}, errors.New("index is closed")
	}

	label := make([]uint64, 1)
	isDeleted := make([]int32, 1)

	result := bindings.GetElementByInternalId(it.index.h, it.currentID, label, isDeleted)
	if result < 0 {
		return ElementInfo{}, errors.New("element not found")
	}

	return ElementInfo{
		InternalID: it.currentID,
		Label:      label[0],
		IsDeleted:  isDeleted[0] != 0,
	}, nil
}

// Vector returns the vector data for the current element.
// For cosine indices, returns the normalized vector.
func (it *VectorIterator) Vector() ([]float32, error) {
	if it.index == nil || it.index.h == nil {
		return nil, errors.New("index is closed")
	}

	vector := make([]float32, it.dimension)
	result := bindings.GetVectorByInternalId(it.index.h, it.currentID, vector)
	if result < 0 {
		return nil, errors.New("failed to get vector")
	}
	return vector, nil
}

// Advance moves to the next element. Call after processing current element.
func (it *VectorIterator) Advance() {
	it.currentID++
}

// Progress returns the current progress as (current, total).
func (it *VectorIterator) Progress() (current, total uint64) {
	return it.currentID, it.totalCount
}

// Reset resets the iterator to the beginning.
func (it *VectorIterator) Reset() {
	it.currentID = 0
	it.totalCount = uint64(it.index.GetCurrentCount())
}

// ExportFunc is called for each element during export.
// Return an error to stop the export early.
type ExportFunc func(label uint64, vector []float32, isDeleted bool) error

// Export iterates over all elements and calls the provided function.
// This is memory-efficient - only one vector is loaded at a time.
// Set includeDeleted=true to include soft-deleted elements.
func (i *Index) Export(includeDeleted bool, fn ExportFunc) error {
	iter, err := i.NewIterator()
	if err != nil {
		return err
	}

	for iter.Next() {
		elem, err := iter.Element()
		if err != nil {
			iter.Advance()
			continue
		}

		if !includeDeleted && elem.IsDeleted {
			iter.Advance()
			continue
		}

		vector, err := iter.Vector()
		if err != nil {
			iter.Advance()
			continue
		}

		if err := fn(elem.Label, vector, elem.IsDeleted); err != nil {
			return err
		}

		iter.Advance()
	}

	return nil
}
