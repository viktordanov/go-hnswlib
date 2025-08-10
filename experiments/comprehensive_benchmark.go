package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/viktordanov/go-hnswlib/hnsw"
)

type BenchmarkResult struct {
	Config         string
	Dimensions     int
	DatasetSize    int
	EF             int
	M              int
	EFConstruction int
	QueryTime      time.Duration
	Recall         float64
	QueriesPerSec  float64
}

type ExperimentConfig struct {
	Dimensions     []int
	DatasetSizes   []int
	EFValues       []int
	M              int
	EFConstruction int
	NumQueries     int
	K              int
	Seed           int64
}

func main() {
	fmt.Println("Comprehensive HNSW Cosine Similarity Benchmark")
	fmt.Println("==============================================")

	// Configuration for experiments
	config := ExperimentConfig{
		Dimensions:     []int{50, 100, 300, 1024, 3096},
		DatasetSizes:   []int{1000, 5000, 10000, 20000, 50000},
		EFValues:       []int{10, 20, 50, 100, 200},
		M:              32,
		EFConstruction: 400,
		NumQueries:     25, // Reduced for faster execution with larger datasets
		K:              10,
		Seed:           42,
	}

	results := runComprehensiveBenchmark(config)

	// Print summary tables
	printSummaryTables(results)

	// Save detailed results to file
	saveResultsToFile(results, "experiments/benchmark_results.txt")

	fmt.Println("\n✅ Comprehensive benchmark completed!")
	fmt.Println("📊 Results saved to experiments/benchmark_results.txt")
}

func runComprehensiveBenchmark(config ExperimentConfig) []BenchmarkResult {
	var results []BenchmarkResult
	totalRuns := len(config.Dimensions) * len(config.DatasetSizes) * len(config.EFValues)
	currentRun := 0

	for _, dim := range config.Dimensions {
		for _, size := range config.DatasetSizes {
			// Skip very large combinations that would take too long
			if dim >= 1024 && size >= 20000 {
				fmt.Printf("Skipping D%d_N%d (too large for this benchmark)\n", dim, size)
				totalRuns -= len(config.EFValues)
				continue
			}

			// Generate test data once per configuration
			fmt.Printf("Generating %d vectors of %d dimensions...\n", size, dim)
			vectors := generateRandomVectors(size, dim, config.Seed)
			queries := generateRandomVectors(config.NumQueries, dim, config.Seed+1)

			// Create naive baseline for recall calculation (sample for large datasets)
			var naive [][]uint64
			if size <= 10000 {
				naive = createNaiveBaseline(vectors, queries, config.K)
			} else {
				// For large datasets, use a sample for recall calculation
				sampleSize := 5000
				sampleVectors := vectors[:sampleSize]
				naive = createNaiveBaseline(sampleVectors, queries, config.K)
				fmt.Printf("Using sample of %d vectors for recall calculation\n", sampleSize)
			}

			for _, ef := range config.EFValues {
				currentRun++
				configName := fmt.Sprintf("D%d_N%d_EF%d", dim, size, ef)
				fmt.Printf("[%d/%d] Running %s...\n", currentRun, totalRuns, configName)

				result := runSingleBenchmark(vectors, queries, naive, BenchmarkConfig{
					Dimensions:     dim,
					DatasetSize:    size,
					EF:             ef,
					M:              config.M,
					EFConstruction: config.EFConstruction,
					K:              config.K,
					ConfigName:     configName,
				})

				results = append(results, result)
			}
		}
	}

	return results
}

type BenchmarkConfig struct {
	Dimensions     int
	DatasetSize    int
	EF             int
	M              int
	EFConstruction int
	K              int
	ConfigName     string
}

func runSingleBenchmark(vectors, queries [][]float32, naiveResults [][]uint64, config BenchmarkConfig) BenchmarkResult {
	// Setup HNSW index
	index := hnsw.NewCosine(config.Dimensions, config.DatasetSize*2, config.M, config.EFConstruction, 42)
	defer index.Close()

	// Add vectors
	for i, vec := range vectors {
		if err := index.Add(vec, uint64(i)); err != nil {
			log.Fatal(err)
		}
	}

	// Set EF parameter
	index.SetEf(config.EF)

	// Benchmark search time
	start := time.Now()
	var hnswResults [][]uint64
	for _, query := range queries {
		labels, _, _ := index.SearchKSimilarity(query, config.K)
		hnswResults = append(hnswResults, labels)
	}
	queryTime := time.Since(start)

	// Calculate recall (handle case where naive results might be from sampled data)
	recall := calculateRecallWithSampling(hnswResults, naiveResults, config.K, config.DatasetSize)

	return BenchmarkResult{
		Config:         config.ConfigName,
		Dimensions:     config.Dimensions,
		DatasetSize:    config.DatasetSize,
		EF:             config.EF,
		M:              config.M,
		EFConstruction: config.EFConstruction,
		QueryTime:      queryTime,
		Recall:         recall,
		QueriesPerSec:  float64(len(queries)) / queryTime.Seconds(),
	}
}

func createNaiveBaseline(vectors, queries [][]float32, k int) [][]uint64 {
	naive := NewNaiveCosineSimilarity()
	for i, vec := range vectors {
		naive.Add(vec, uint64(i))
	}

	var naiveResults [][]uint64
	for _, query := range queries {
		labels, _, _ := naive.SearchK(query, k)
		naiveResults = append(naiveResults, labels)
	}

	return naiveResults
}

func calculateRecall(hnswResults, naiveResults [][]uint64, k int) float64 {
	totalRecall := 0.0

	for i := 0; i < len(hnswResults); i++ {
		naiveSet := make(map[uint64]bool)
		for _, label := range naiveResults[i] {
			naiveSet[label] = true
		}

		overlap := 0
		for _, label := range hnswResults[i] {
			if naiveSet[label] {
				overlap++
			}
		}
		totalRecall += float64(overlap) / float64(k)
	}

	return totalRecall / float64(len(hnswResults))
}

func calculateRecallWithSampling(hnswResults, naiveResults [][]uint64, k int, datasetSize int) float64 {
	// For large datasets where we used sampling, we can only check overlap
	// within the sampled range (first 5000 vectors)
	sampleSize := 5000
	if datasetSize <= 10000 {
		// Use normal recall calculation for smaller datasets
		return calculateRecall(hnswResults, naiveResults, k)
	}

	totalRecall := 0.0

	for i := 0; i < len(hnswResults); i++ {
		naiveSet := make(map[uint64]bool)
		for _, label := range naiveResults[i] {
			naiveSet[label] = true
		}

		overlap := 0
		validResults := 0
		for _, label := range hnswResults[i] {
			if label < uint64(sampleSize) { // Only count results within sampled range
				validResults++
				if naiveSet[label] {
					overlap++
				}
			}
		}

		if validResults > 0 {
			totalRecall += float64(overlap) / float64(validResults)
		}
	}

	return totalRecall / float64(len(hnswResults))
}

func printSummaryTables(results []BenchmarkResult) {
	fmt.Println("\n📊 SUMMARY TABLES")
	fmt.Println("=================")

	// Group results by dimension and dataset size
	groupedResults := make(map[string][]BenchmarkResult)
	for _, result := range results {
		key := fmt.Sprintf("D%d_N%d", result.Dimensions, result.DatasetSize)
		groupedResults[key] = append(groupedResults[key], result)
	}

	// Sort keys for consistent output
	var keys []string
	for key := range groupedResults {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	for _, key := range keys {
		groupResults := groupedResults[key]

		// Sort by EF value
		sort.Slice(groupResults, func(i, j int) bool {
			return groupResults[i].EF < groupResults[j].EF
		})

		fmt.Printf("\n🎯 Configuration: %d dimensions, %d vectors\n",
			groupResults[0].Dimensions, groupResults[0].DatasetSize)
		fmt.Println("EF    | Recall | Query Time | Queries/sec | Speedup vs EF=200")
		fmt.Println("------|--------|------------|-------------|------------------")

		var baselineTime time.Duration
		for _, result := range groupResults {
			if result.EF == 200 {
				baselineTime = result.QueryTime
				break
			}
		}
		if baselineTime == 0 && len(groupResults) > 0 {
			baselineTime = groupResults[len(groupResults)-1].QueryTime
		}

		for _, result := range groupResults {
			speedup := float64(baselineTime) / float64(result.QueryTime)
			if baselineTime == 0 {
				speedup = 1.0
			}

			fmt.Printf("%-5d | %5.1f%% | %10v | %11.0f | %14.1fx\n",
				result.EF,
				result.Recall*100,
				result.QueryTime,
				result.QueriesPerSec,
				speedup)
		}
	}

	// Print overall insights
	printInsights(results)
}

func printInsights(results []BenchmarkResult) {
	fmt.Println("\n🔍 KEY INSIGHTS")
	fmt.Println("===============")

	// Find best recall by configuration
	configRecall := make(map[string]float64)
	configBestEF := make(map[string]int)

	for _, result := range results {
		key := fmt.Sprintf("D%d_N%d", result.Dimensions, result.DatasetSize)
		if result.Recall > configRecall[key] {
			configRecall[key] = result.Recall
			configBestEF[key] = result.EF
		}
	}

	fmt.Println("Configuration-wise best recall:")
	for key, recall := range configRecall {
		fmt.Printf("  %s: %.1f%% recall at EF=%d\n", key, recall*100, configBestEF[key])
	}

	// Find sweet spot (good recall + speed)
	fmt.Println("\nSweet spot analysis (>95% recall):")
	for _, result := range results {
		if result.Recall >= 0.95 {
			key := fmt.Sprintf("D%d_N%d", result.Dimensions, result.DatasetSize)
			fmt.Printf("  %s: EF=%d achieves %.1f%% recall at %.0f queries/sec\n",
				key, result.EF, result.Recall*100, result.QueriesPerSec)
			break // Take first (lowest EF) that achieves >95%
		}
	}
}

func saveResultsToFile(results []BenchmarkResult, filename string) {
	// Ensure directory exists
	dir := filepath.Dir(filename)
	if err := os.MkdirAll(dir, 0755); err != nil {
		log.Printf("Failed to create directory: %v", err)
		return
	}

	file, err := os.Create(filename)
	if err != nil {
		log.Printf("Failed to create file: %v", err)
		return
	}
	defer file.Close()

	// Write header
	fmt.Fprintf(file, "Comprehensive HNSW Cosine Similarity Benchmark Results\n")
	fmt.Fprintf(file, "Generated: %s\n", time.Now().Format("2006-01-02 15:04:05"))
	fmt.Fprintf(file, "========================================================\n\n")

	// Write detailed results
	fmt.Fprintf(file, "Detailed Results:\n")
	fmt.Fprintf(file, "Config       | Dim | Size  | EF  | M  | EFConstr | Time      | Recall | QPS    \n")
	fmt.Fprintf(file, "-------------|-----|-------|-----|----|---------|-----------|---------|---------\n")

	for _, result := range results {
		fmt.Fprintf(file, "%-12s | %-3d | %-5d | %-3d | %-2d | %-8d | %-9v | %5.1f%% | %7.0f\n",
			result.Config,
			result.Dimensions,
			result.DatasetSize,
			result.EF,
			result.M,
			result.EFConstruction,
			result.QueryTime,
			result.Recall*100,
			result.QueriesPerSec)
	}

	fmt.Fprintf(file, "\nLegend:\n")
	fmt.Fprintf(file, "- Dim: Vector dimensions\n")
	fmt.Fprintf(file, "- Size: Dataset size (number of vectors)\n")
	fmt.Fprintf(file, "- EF: Search parameter (higher = better recall, slower)\n")
	fmt.Fprintf(file, "- M: Max connections per node\n")
	fmt.Fprintf(file, "- EFConstr: Construction parameter\n")
	fmt.Fprintf(file, "- QPS: Queries per second\n")
}

func generateRandomVectors(count, dim int, seed int64) [][]float32 {
	rand.Seed(seed)
	vectors := make([][]float32, count)
	for i := 0; i < count; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float32()*2 - 1 // Range [-1, 1]
		}
		vectors[i] = vec
	}
	return vectors
}
