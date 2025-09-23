// hnsw_wrapper.h
#ifdef __cplusplus
extern "C" {
#endif
  typedef void* HNSW;
  HNSW initHNSW(int dim, unsigned long long int max_elements, int M, int ef_construction, int rand_seed, char stype);
  HNSW loadHNSW(char *location, int dim, char stype);
  
  // Safe loading (returns NULL on failure)
  HNSW loadHNSWSafe(char *location, int dim, char stype);
  HNSW saveHNSW(HNSW index, char *location);
  void freeHNSW(HNSW index);
  void addPoint(HNSW index, float *vec, unsigned long long int label);
  int searchKnn(HNSW index, float *vec, int N, unsigned long long int *label, float *dist);
  void setEf(HNSW index, int ef);
  void resizeIndex(HNSW index, unsigned long long int new_max_elements);
  
  // Introspection functions (safe)
  unsigned long long getCurrentElementCount(HNSW index);
  unsigned long long getMaxElements(HNSW index);
  unsigned long long getDeletedCount(HNSW index);
  
  // Delete management functions
  void markDeleted(HNSW index, unsigned long long label);
  void unmarkDeleted(HNSW index, unsigned long long label);
  
  // Safe versions with error handling (return 0 on success, non-zero on error)
  int addPointSafe(HNSW index, float *vec, unsigned long long label);
  int resizeIndexSafe(HNSW index, unsigned long long new_max_elements);
  int saveIndexSafe(HNSW index, char *location);
  int markDeletedSafe(HNSW index, unsigned long long label);
  int unmarkDeletedSafe(HNSW index, unsigned long long label);
#ifdef __cplusplus
}
#endif
