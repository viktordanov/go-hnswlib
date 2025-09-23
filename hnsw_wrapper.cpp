//hnsw_wrapper.cpp
#include <iostream>
#include "hnswlib/hnswlib.h"
#include "hnsw_wrapper.h"
#include <thread>
#include <atomic>
#include <cmath>

HNSW initHNSW(int dim, unsigned long long int max_elements, int M, int ef_construction, int rand_seed, char stype) {
  hnswlib::SpaceInterface<float> *space;
  if (stype == 'i' || stype == 'c') {
    // Both inner product and cosine use inner product space
    // For cosine, vectors will be normalized at the Go level
    space = new hnswlib::InnerProductSpace(dim);
  } else {
    space = new hnswlib::L2Space(dim);
  }
  hnswlib::HierarchicalNSW<float> *appr_alg = new hnswlib::HierarchicalNSW<float>(space, max_elements, M, ef_construction, rand_seed);
  return (void*)appr_alg;
}

HNSW loadHNSW(char *location, int dim, char stype) {
  hnswlib::SpaceInterface<float> *space;
  if (stype == 'i' || stype == 'c') {
    space = new hnswlib::InnerProductSpace(dim);
  } else {
    space = new hnswlib::L2Space(dim);
  }
  hnswlib::HierarchicalNSW<float> *appr_alg = new hnswlib::HierarchicalNSW<float>(space, std::string(location), false, 0);
  return (void*)appr_alg;
}

HNSW loadHNSWSafe(char *location, int dim, char stype) {
  try {
    hnswlib::SpaceInterface<float> *space;
    if (stype == 'i' || stype == 'c') {
      space = new hnswlib::InnerProductSpace(dim);
    } else {
      space = new hnswlib::L2Space(dim);
    }
    hnswlib::HierarchicalNSW<float> *appr_alg = new hnswlib::HierarchicalNSW<float>(space, std::string(location), false, 0);
    return (void*)appr_alg;
  } catch (const std::exception& e) {
    return nullptr;
  }
}

HNSW saveHNSW(HNSW index, char *location) {
  ((hnswlib::HierarchicalNSW<float>*)index)->saveIndex(location);
  return ((hnswlib::HierarchicalNSW<float>*)index);
}

void freeHNSW(HNSW index) {
  hnswlib::HierarchicalNSW<float>* ptr = (hnswlib::HierarchicalNSW<float>*) index;
  delete ptr;
}

void addPoint(HNSW index, float *vec, unsigned long long int label) {
        ((hnswlib::HierarchicalNSW<float>*)index)->addPoint(vec, label);
}

int searchKnn(HNSW index, float *vec, int N, unsigned long long int *label, float *dist) {
  std::priority_queue<std::pair<float, hnswlib::labeltype>> gt;
  try {
    gt = ((hnswlib::HierarchicalNSW<float>*)index)->searchKnn(vec, N);
  } catch (const std::exception& e) { 
    return 0;
  }

  int n = gt.size();
  std::pair<float, hnswlib::labeltype> pair;
  for (int i = n - 1; i >= 0; i--) {
    pair = gt.top();
    *(dist+i) = pair.first;
    *(label+i) = pair.second;
    gt.pop();
  }
  return n;
}

void setEf(HNSW index, int ef) {
    ((hnswlib::HierarchicalNSW<float>*)index)->ef_ = ef;
}

void resizeIndex(HNSW index, unsigned long long int new_max_elements) {
    ((hnswlib::HierarchicalNSW<float>*)index)->resizeIndex(new_max_elements);
}

// Introspection functions (safe)
unsigned long long getCurrentElementCount(HNSW index) {
    return ((hnswlib::HierarchicalNSW<float>*)index)->getCurrentElementCount();
}

unsigned long long getMaxElements(HNSW index) {
    return ((hnswlib::HierarchicalNSW<float>*)index)->getMaxElements();
}

unsigned long long getDeletedCount(HNSW index) {
    return ((hnswlib::HierarchicalNSW<float>*)index)->getDeletedCount();
}

// Delete management functions
void markDeleted(HNSW index, unsigned long long label) {
    ((hnswlib::HierarchicalNSW<float>*)index)->markDelete(label);
}

void unmarkDeleted(HNSW index, unsigned long long label) {
    ((hnswlib::HierarchicalNSW<float>*)index)->unmarkDelete(label);
}

// Safe versions with error handling
int addPointSafe(HNSW index, float *vec, unsigned long long label) {
    try {
        ((hnswlib::HierarchicalNSW<float>*)index)->addPoint(vec, label);
        return 0;
    } catch (const std::exception& e) {
        return -1;
    }
}

int resizeIndexSafe(HNSW index, unsigned long long new_max_elements) {
    try {
        ((hnswlib::HierarchicalNSW<float>*)index)->resizeIndex(new_max_elements);
        return 0;
    } catch (const std::exception& e) {
        return -1;
    }
}

int saveIndexSafe(HNSW index, char *location) {
    try {
        ((hnswlib::HierarchicalNSW<float>*)index)->saveIndex(std::string(location));
        return 0;
    } catch (const std::exception& e) {
        return -1;
    }
}

int markDeletedSafe(HNSW index, unsigned long long label) {
    try {
        ((hnswlib::HierarchicalNSW<float>*)index)->markDelete(label);
        return 0;
    } catch (const std::exception& e) {
        return -1;
    }
}

int unmarkDeletedSafe(HNSW index, unsigned long long label) {
    try {
        ((hnswlib::HierarchicalNSW<float>*)index)->unmarkDelete(label);
        return 0;
    } catch (const std::exception& e) {
        return -1;
    }
}
