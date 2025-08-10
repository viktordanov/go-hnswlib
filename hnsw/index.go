package hnsw

import (
	"runtime"

	"github.com/vikimaster2/go-hnswlib/bindings/bindings"
)

type Space byte

const (
	SpaceL2 Space = 'l'
	SpaceIP Space = 'i'
)

type Index struct {
	h *bindings.HNSW
}

func New(space Space, dim, maxElements, M, efConstruction, seed int) *Index {
	idx := &Index{}
	idx.h = bindings.InitHNSW(int32(dim), uint64(maxElements), int32(M), int32(efConstruction), int32(seed), byte(space))
	runtime.SetFinalizer(idx, (*Index).Close)
	return idx
}

func Load(space Space, dim int, path string) *Index {
	pathBytes := []byte(path + "\x00") // null terminate
	h := bindings.LoadHNSW(pathBytes, int32(dim), byte(space))
	if h == nil {
		return nil
	}
	idx := &Index{h: h}
	runtime.SetFinalizer(idx, (*Index).Close)
	return idx
}

func (i *Index) Close() {
	if i == nil || i.h == nil {
		return
	}
	bindings.FreeHNSW(i.h)
	i.h = nil
	runtime.SetFinalizer(i, nil)
}

func (i *Index) Add(vec []float32, label uint64) {
	if i == nil || i.h == nil {
		return
	}
	bindings.AddPoint(i.h, vec, label)
}

func (i *Index) SearchK(query []float32, k int) (labels []uint64, distances []float32, count int) {
	if i == nil || i.h == nil {
		return nil, nil, 0
	}
	labels = make([]uint64, k)
	distances = make([]float32, k)
	count = int(bindings.SearchKnn(i.h, query, int32(k), labels, distances))
	if count < k {
		labels = labels[:count]
		distances = distances[:count]
	}
	return labels, distances, count
}

func (i *Index) SetEf(ef int) {
	if i == nil || i.h == nil {
		return
	}
	bindings.SetEf(i.h, int32(ef))
}

func (i *Index) Save(path string) *Index {
	if i == nil || i.h == nil {
		return i
	}
	pathBytes := []byte(path + "\x00") // null terminate
	bindings.SaveHNSW(i.h, pathBytes)
	return i
}

// Helper constructors for common use cases
func NewL2(dim, maxElements, M, efConstruction, seed int) *Index {
	return New(SpaceL2, dim, maxElements, M, efConstruction, seed)
}

func NewIP(dim, maxElements, M, efConstruction, seed int) *Index {
	return New(SpaceIP, dim, maxElements, M, efConstruction, seed)
}
