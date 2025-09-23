package benchmark

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"time"

	"github.com/shubhang93/tablewr"
	"github.com/viktordanov/go-hnswlib/experimental/parallel"
	"github.com/viktordanov/go-hnswlib/hnsw"
)

type BenchmarkSuite string

const (
	SuiteQuick         BenchmarkSuite = "quick"
	SuiteStandard      BenchmarkSuite = "standard"
	SuiteComprehensive BenchmarkSuite = "comprehensive"
	SuiteParallel      BenchmarkSuite = "parallel"
	SuiteCustom        BenchmarkSuite = "custom"
)

type PlatformConfig struct {
	Suite          BenchmarkSuite `json:"suite"`
	Dimensions     []int          `json:"dimensions"`
	DatasetSizes   []int          `json:"dataset_sizes"`
	EFValues       []int          `json:"ef_values"`
	M              int            `json:"m"`
	EFConstruction int            `json:"ef_construction"`
	NumQueries     int            `json:"num_queries"`
	K              int            `json:"k"`
	Seed           int64          `json:"seed"`

	TestSequential bool `json:"test_sequential"`
	TestParallel   bool `json:"test_parallel"`
	MaxWorkers     int  `json:"max_workers"`

	DetectHardware bool   `json:"detect_hardware"`
	OutputDir      string `json:"output_dir"`

	WarmupRuns      int  `json:"warmup_runs"`
	SkipLargeCombos bool `json:"skip_large_combos"`
}

type SystemInfo struct {
	OS         string `json:"os"`
	Arch       string `json:"arch"`
	CPUs       int    `json:"cpus"`
	GOMAXPROCS int    `json:"gomaxprocs"`
	GoVersion  string `json:"go_version"`
	CGOEnabled bool   `json:"cgo_enabled"`
	Timestamp  string `json:"timestamp"`
}

type BenchmarkResult struct {
	Config         string        `json:"config"`
	Dimensions     int           `json:"dimensions"`
	DatasetSize    int           `json:"dataset_size"`
	EF             int           `json:"ef"`
	M              int           `json:"m"`
	EFConstruction int           `json:"ef_construction"`
	QueryTime      time.Duration `json:"query_time"`
	Recall         float64       `json:"recall"`
	QueriesPerSec  float64       `json:"queries_per_sec"`
	ExecutionMode  string        `json:"execution_mode"`
	WorkerCount    int           `json:"worker_count"`
}

type BenchmarkReport struct {
	System    SystemInfo        `json:"system_info"`
	Config    PlatformConfig    `json:"benchmark_config"`
	Results   []BenchmarkResult `json:"results"`
	Generated time.Time         `json:"generated"`
}

func GetSystemInfo() SystemInfo {
	return SystemInfo{
		OS:         runtime.GOOS,
		Arch:       runtime.GOARCH,
		CPUs:       runtime.NumCPU(),
		GOMAXPROCS: runtime.GOMAXPROCS(0),
		GoVersion:  runtime.Version(),
		CGOEnabled: true,
		Timestamp:  fmt.Sprintf("%d", time.Now().Unix()),
	}
}

func GetDefaultConfig(suite BenchmarkSuite) PlatformConfig {
	base := PlatformConfig{
		Suite:           suite,
		M:               32,
		EFConstruction:  400,
		K:               10,
		Seed:            42,
		TestSequential:  true,
		TestParallel:    false,
		MaxWorkers:      runtime.GOMAXPROCS(0) * 2,
		DetectHardware:  true,
		OutputDir:       "results",
		WarmupRuns:      1,
		SkipLargeCombos: true,
	}

	switch suite {
	case SuiteQuick:
		base.Dimensions = []int{50, 128}
		base.DatasetSizes = []int{1000, 5000}
		base.EFValues = []int{50, 100}
		base.NumQueries = 10

	case SuiteStandard:
		base.Dimensions = []int{50, 128, 384}
		base.DatasetSizes = []int{1000, 5000, 10000}
		base.EFValues = []int{20, 50, 100, 200}
		base.NumQueries = 25

	case SuiteComprehensive:
		base.Dimensions = []int{50, 100, 300, 768, 1024}
		base.DatasetSizes = []int{1000, 5000, 10000, 20000}
		base.EFValues = []int{10, 20, 50, 100, 200, 400}
		base.NumQueries = 50

	case SuiteParallel:
		base.Dimensions = []int{128, 384}
		base.DatasetSizes = []int{5000, 10000}
		base.EFValues = []int{50, 100, 200}
		base.NumQueries = 100
		base.TestSequential = true
		base.TestParallel = true

	default:
		base.Dimensions = []int{128}
		base.DatasetSizes = []int{1000}
		base.EFValues = []int{50}
		base.NumQueries = 10
	}

	return base
}

func (c *PlatformConfig) Validate() error {
	if len(c.Dimensions) == 0 {
		return fmt.Errorf("at least one dimension must be specified")
	}
	if len(c.DatasetSizes) == 0 {
		return fmt.Errorf("at least one dataset size must be specified")
	}
	if len(c.EFValues) == 0 {
		return fmt.Errorf("at least one EF value must be specified")
	}
	if c.NumQueries <= 0 {
		return fmt.Errorf("num_queries must be positive")
	}
	if !c.TestSequential && !c.TestParallel {
		return fmt.Errorf("at least one execution mode (sequential or parallel) must be enabled")
	}
	return nil
}

func (c *PlatformConfig) EstimateRuntime() string {
	combinations := len(c.Dimensions) * len(c.DatasetSizes) * len(c.EFValues)

	if c.TestSequential && c.TestParallel {
		combinations *= 2
	}

	avgTimePerRun := 5.0
	for _, size := range c.DatasetSizes {
		if size > 10000 {
			avgTimePerRun = 15.0
			break
		}
	}

	totalMinutes := float64(combinations) * avgTimePerRun / 60.0

	if totalMinutes < 1 {
		return "< 1 minute"
	} else if totalMinutes < 60 {
		return fmt.Sprintf("~%.0f minutes", totalMinutes)
	} else {
		return fmt.Sprintf("~%.1f hours", totalMinutes/60.0)
	}
}

func RunBenchmark(config PlatformConfig) []BenchmarkResult {
	var results []BenchmarkResult
	totalRuns := calculateTotalRuns(config)
	currentRun := 0

	for _, dim := range config.Dimensions {
		for _, size := range config.DatasetSizes {
			if shouldSkipConfiguration(config, dim, size) {
				continue
			}

			vectors, queries := generateTestData(size, dim, config)
			naive := createBaseline(vectors, queries, size, config.K)

			for _, ef := range config.EFValues {
				currentRun++
				configName := fmt.Sprintf("D%d_N%d_EF%d", dim, size, ef)
				fmt.Printf("[%d/%d] Running %s...\n", currentRun, totalRuns, configName)

				benchConfig := BenchmarkConfig{
					Dimensions:     dim,
					DatasetSize:    size,
					EF:             ef,
					M:              config.M,
					EFConstruction: config.EFConstruction,
					K:              config.K,
					ConfigName:     configName,
				}

				if config.TestSequential {
					result := runSequentialBenchmark(vectors, queries, naive, benchConfig, config)
					result.ExecutionMode = "sequential"
					results = append(results, result)
				}

				if config.TestParallel {
					result := runParallelBenchmark(vectors, queries, naive, benchConfig, config)
					result.ExecutionMode = "parallel"
					results = append(results, result)
				}
			}
		}
	}

	return results
}

func SaveReport(report *BenchmarkReport, outputDir string) {
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		log.Printf("Failed to create output directory: %v", err)
		return
	}

	saveTextReport(report, outputDir)
}

func PrintConsoleSummary(results []BenchmarkResult, config PlatformConfig) {
	fmt.Println("\nBENCHMARK SUMMARY")
	fmt.Println("====================")

	totalConfigs := len(groupResultsByConfig(results))
	fmt.Printf("Tested %d configurations with %d total runs\n", totalConfigs, len(results))

	if config.TestSequential && config.TestParallel {
		fmt.Println("\nExecution modes: Sequential + Parallel")
	} else if config.TestParallel {
		fmt.Println("\nExecution mode: Parallel only")
	} else {
		fmt.Println("\nExecution mode: Sequential only")
	}

	fmt.Println("\nTop Results Preview:")
	printTopResultsTable(results)

	fmt.Printf("\nResults saved to: %s/\n", config.OutputDir)
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

func generateTestData(size, dim int, config PlatformConfig) ([][]float32, [][]float32) {
	fmt.Printf("Generating %d vectors of %d dimensions...\n", size, dim)
	vectors := GenerateRandomVectors(size, dim, config.Seed)
	queries := GenerateRandomVectors(config.NumQueries, dim, config.Seed+1)
	return vectors, queries
}

func GenerateRandomVectors(count, dim int, seed int64) [][]float32 {
	rand.Seed(seed)
	vectors := make([][]float32, count)
	for i := 0; i < count; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float32()*2 - 1
		}
		vectors[i] = vec
	}
	return vectors
}

func createBaseline(vectors, queries [][]float32, size, k int) [][]uint64 {
	if size <= 10000 {
		return createNaiveBaseline(vectors, queries, k)
	}

	sampleSize := 5000
	sampleVectors := vectors[:sampleSize]
	fmt.Printf("Using sample of %d vectors for recall calculation\n", sampleSize)
	return createNaiveBaseline(sampleVectors, queries, k)
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

func runSequentialBenchmark(vectors, queries [][]float32, naiveResults [][]uint64, config BenchmarkConfig, platformConfig PlatformConfig) BenchmarkResult {
	index := hnsw.NewCosine(config.Dimensions, config.DatasetSize*2, config.M, config.EFConstruction, 42)
	defer index.Close()

	for i, vec := range vectors {
		if err := index.Add(vec, uint64(i)); err != nil {
			log.Fatal(err)
		}
	}

	index.SetEf(config.EF)

	start := time.Now()
	var hnswResults [][]uint64
	for _, query := range queries {
		labels, _, _ := index.SearchKSimilarity(query, config.K)
		hnswResults = append(hnswResults, labels)
	}
	queryTime := time.Since(start)

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
		ExecutionMode:  "sequential",
		WorkerCount:    1,
	}
}

func runParallelBenchmark(vectors, queries [][]float32, naiveResults [][]uint64, config BenchmarkConfig, platformConfig PlatformConfig) BenchmarkResult {
	index := hnsw.NewCosine(config.Dimensions, config.DatasetSize*2, config.M, config.EFConstruction, 42)
	defer index.Close()

	vectorData := make([]parallel.VectorData, len(vectors))
	for i, vec := range vectors {
		vectorData[i] = parallel.VectorData{Vector: vec, Label: uint64(i)}
	}

	err := parallel.BatchAdd(index, vectorData, &parallel.BatchAddOptions{
		MaxWorkers: platformConfig.MaxWorkers,
	})
	if err != nil {
		log.Fatal(err)
	}

	index.SetEf(config.EF)

	start := time.Now()
	results, err := parallel.ParallelSearch(index, queries, config.K, &parallel.ParallelSearchOptions{
		MaxWorkers: platformConfig.MaxWorkers,
	})
	if err != nil {
		log.Fatal(err)
	}
	queryTime := time.Since(start)

	var hnswResults [][]uint64
	for _, result := range results {
		hnswResults = append(hnswResults, result.Labels)
	}

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
		ExecutionMode:  "parallel",
		WorkerCount:    platformConfig.MaxWorkers,
	}
}

func calculateRecallWithSampling(hnswResults, naiveResults [][]uint64, k int, datasetSize int) float64 {
	sampleSize := 5000
	if datasetSize <= 10000 {
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
			if label < uint64(sampleSize) {
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

func calculateTotalRuns(config PlatformConfig) int {
	total := 0
	for _, dim := range config.Dimensions {
		for _, size := range config.DatasetSizes {
			if shouldSkipConfiguration(config, dim, size) {
				continue
			}
			runs := len(config.EFValues)
			if config.TestSequential && config.TestParallel {
				runs *= 2
			}
			total += runs
		}
	}
	return total
}

func shouldSkipConfiguration(config PlatformConfig, dim, size int) bool {
	if !config.SkipLargeCombos {
		return false
	}

	if dim >= 1024 && size >= 20000 {
		fmt.Printf("Skipping D%d_N%d (too large)\n", dim, size)
		return true
	}
	return false
}

func groupResultsByConfig(results []BenchmarkResult) map[string][]BenchmarkResult {
	grouped := make(map[string][]BenchmarkResult)
	for _, result := range results {
		key := fmt.Sprintf("D%d_N%d", result.Dimensions, result.DatasetSize)
		grouped[key] = append(grouped[key], result)
	}
	return grouped
}

func printTopResultsTable(results []BenchmarkResult) {
	if len(results) == 0 {
		return
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].QueriesPerSec > results[j].QueriesPerSec
	})

	wr := tablewr.New(os.Stdout, 0, tablewr.WithSep())
	data := [][]string{
		{"Config", "Mode", "EF", "Recall", "QPS", "Time"},
	}

	limit := 10
	if len(results) < limit {
		limit = len(results)
	}

	for i := 0; i < limit; i++ {
		result := results[i]
		row := []string{
			result.Config,
			result.ExecutionMode,
			fmt.Sprintf("%d", result.EF),
			fmt.Sprintf("%.1f%%", result.Recall*100),
			fmt.Sprintf("%.0f", result.QueriesPerSec),
			result.QueryTime.String(),
		}
		data = append(data, row)
	}

	if err := wr.Write(data); err != nil {
		log.Printf("Failed to write console table: %v", err)
	}
}

func saveTextReport(report *BenchmarkReport, outputDir string) {
	filename := filepath.Join(outputDir, "benchmark_results.txt")
	file, err := os.Create(filename)
	if err != nil {
		log.Printf("Failed to create text report: %v", err)
		return
	}
	defer file.Close()

	fmt.Fprintf(file, "HNSW Platform Benchmark Report\n")
	fmt.Fprintf(file, "==============================\n")
	fmt.Fprintf(file, "Generated: %s\n", report.Generated.Format("2006-01-02 15:04:05"))
	fmt.Fprintf(file, "Suite: %s\n", report.Config.Suite)
	fmt.Fprintf(file, "Platform: %s/%s (%d CPUs)\n\n", report.System.OS, report.System.Arch, report.System.CPUs)

	writeDetailedResults(file, report.Results)
	writeSummaryTables(file, report.Results)
	writeInsights(file, report.Results, report.Config)
}

func writeDetailedResults(file *os.File, results []BenchmarkResult) {
	fmt.Fprintf(file, "Detailed Results:\n")

	wr := tablewr.New(file, 0, tablewr.WithSep())

	data := [][]string{
		{"Config", "Dim", "Size", "EF", "Mode", "Workers", "Time", "Recall", "QPS"},
	}

	for _, result := range results {
		row := []string{
			result.Config,
			fmt.Sprintf("%d", result.Dimensions),
			fmt.Sprintf("%d", result.DatasetSize),
			fmt.Sprintf("%d", result.EF),
			result.ExecutionMode,
			fmt.Sprintf("%d", result.WorkerCount),
			result.QueryTime.String(),
			fmt.Sprintf("%.1f%%", result.Recall*100),
			fmt.Sprintf("%.0f", result.QueriesPerSec),
		}
		data = append(data, row)
	}

	if err := wr.Write(data); err != nil {
		log.Printf("Failed to write table: %v", err)
	}
	fmt.Fprintf(file, "\n")
}

func writeSummaryTables(file *os.File, results []BenchmarkResult) {
	fmt.Fprintf(file, "Summary Tables:\n")
	fmt.Fprintf(file, "===============\n")

	groupedResults := groupResultsByConfig(results)

	var keys []string
	for key := range groupedResults {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	for _, key := range keys {
		groupResults := groupedResults[key]
		sort.Slice(groupResults, func(i, j int) bool {
			if groupResults[i].EF != groupResults[j].EF {
				return groupResults[i].EF < groupResults[j].EF
			}
			return groupResults[i].ExecutionMode < groupResults[j].ExecutionMode
		})

		fmt.Fprintf(file, "\nConfiguration: %d dimensions, %d vectors\n",
			groupResults[0].Dimensions, groupResults[0].DatasetSize)

		wr := tablewr.New(file, 0, tablewr.WithSep())
		data := [][]string{
			{"EF", "Mode", "Recall", "Query Time", "QPS", "Speedup"},
		}

		var baselineTime time.Duration
		for _, result := range groupResults {
			if result.EF == 200 && result.ExecutionMode == "sequential" {
				baselineTime = result.QueryTime
				break
			}
		}
		if baselineTime == 0 && len(groupResults) > 0 {
			baselineTime = groupResults[0].QueryTime
		}

		for _, result := range groupResults {
			speedup := float64(baselineTime) / float64(result.QueryTime)
			if baselineTime == 0 {
				speedup = 1.0
			}

			row := []string{
				fmt.Sprintf("%d", result.EF),
				result.ExecutionMode,
				fmt.Sprintf("%.1f%%", result.Recall*100),
				result.QueryTime.String(),
				fmt.Sprintf("%.0f", result.QueriesPerSec),
				fmt.Sprintf("%.1fx", speedup),
			}
			data = append(data, row)
		}

		if err := wr.Write(data); err != nil {
			log.Printf("Failed to write summary table: %v", err)
		}
		fmt.Fprintf(file, "\n")
	}
}

func writeInsights(file *os.File, results []BenchmarkResult, config PlatformConfig) {
	fmt.Fprintf(file, "\nKey Insights:\n")
	fmt.Fprintf(file, "=============\n")

	if config.TestSequential && config.TestParallel {
		writeParallelComparison(file, results)
	}

	writeBestConfigurations(file, results)
}

func writeParallelComparison(file *os.File, results []BenchmarkResult) {
	fmt.Fprintf(file, "\nSequential vs Parallel Performance:\n")

	wr := tablewr.New(file, 0, tablewr.WithSep())
	data := [][]string{
		{"Config", "Sequential QPS", "Parallel QPS", "Speedup", "Workers"},
	}

	for _, result := range results {
		if result.ExecutionMode == "parallel" {
			key := fmt.Sprintf("D%d_N%d_EF%d", result.Dimensions, result.DatasetSize, result.EF)
			var seqTime time.Duration
			var seqQPS float64
			for _, seqResult := range results {
				if seqResult.ExecutionMode == "sequential" &&
					seqResult.Dimensions == result.Dimensions &&
					seqResult.DatasetSize == result.DatasetSize &&
					seqResult.EF == result.EF {
					seqTime = seqResult.QueryTime
					seqQPS = seqResult.QueriesPerSec
					break
				}
			}
			if seqTime > 0 {
				speedup := float64(seqTime) / float64(result.QueryTime)

				row := []string{
					key,
					fmt.Sprintf("%.0f", seqQPS),
					fmt.Sprintf("%.0f", result.QueriesPerSec),
					fmt.Sprintf("%.1fx", speedup),
					fmt.Sprintf("%d", result.WorkerCount),
				}
				data = append(data, row)
			}
		}
	}

	if len(data) > 1 {
		if err := wr.Write(data); err != nil {
			log.Printf("Failed to write parallel comparison table: %v", err)
		}
	}
}

func writeBestConfigurations(file *os.File, results []BenchmarkResult) {
	fmt.Fprintf(file, "\nBest configurations (>90%% recall):\n")

	wr := tablewr.New(file, 0, tablewr.WithSep())
	data := [][]string{
		{"Config", "Mode", "EF", "Recall", "QPS", "Time"},
	}

	for _, result := range results {
		if result.Recall >= 0.90 {
			row := []string{
				result.Config,
				result.ExecutionMode,
				fmt.Sprintf("%d", result.EF),
				fmt.Sprintf("%.1f%%", result.Recall*100),
				fmt.Sprintf("%.0f", result.QueriesPerSec),
				result.QueryTime.String(),
			}
			data = append(data, row)
		}
	}

	if len(data) > 1 {
		if err := wr.Write(data); err != nil {
			log.Printf("Failed to write best configurations table: %v", err)
		}
	}
}
