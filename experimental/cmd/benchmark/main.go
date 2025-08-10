package main

import (
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/viktordanov/go-hnswlib/experimental/benchmark"
)

func main() {
	var suite = flag.String("suite", "standard", "Benchmark suite: quick, standard, comprehensive, parallel, custom")
	flag.Parse()

	fmt.Printf("HNSW Platform Benchmark Suite: %s\n", *suite)
	fmt.Println("==================================")

	config := benchmark.GetDefaultConfig(benchmark.BenchmarkSuite(*suite))

	if err := config.Validate(); err != nil {
		log.Fatalf("Invalid configuration: %v", err)
	}

	fmt.Printf("Estimated runtime: %s\n", config.EstimateRuntime())
	fmt.Printf("Output format: TXT\n")
	fmt.Println()

	system := benchmark.GetSystemInfo()
	results := benchmark.RunBenchmark(config)

	report := &benchmark.BenchmarkReport{
		System:    system,
		Config:    config,
		Results:   results,
		Generated: time.Now(),
	}

	benchmark.PrintConsoleSummary(results, config)
	benchmark.SaveReport(report, config.OutputDir)

	fmt.Println("\n✅ Benchmark completed!")
}
