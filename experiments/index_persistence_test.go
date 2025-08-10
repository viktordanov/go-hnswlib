package main

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/viktordanov/go-hnswlib/hnsw"
)

// Test configurations for build time, save time, and index size analysis
var testConfigs = []struct {
	name       string
	dimensions int
	size       int
	samples    int // Number of test runs for averaging
}{
	{"Small: 100D, 1K vectors", 100, 1000, 5},
	{"Medium: 300D, 5K vectors", 300, 5000, 3},
	{"Large: 1024D, 10K vectors", 1024, 10000, 2},
	{"XLarge: 300D, 50K vectors", 300, 50000, 1},
	{"High-Dim: 3096D, 1K vectors", 3096, 1000, 3},
}

func TestIndexBuildTime(t *testing.T) {
	fmt.Println("HNSW Index Build Time Benchmark")
	fmt.Println("===============================")
	fmt.Printf("%-30s | %-12s | %-12s | %-10s\n", "Configuration", "Avg Build", "Per Vector", "Rate (vec/s)")
	fmt.Println(fmt.Sprintf("%s|%s|%s|%s",
		"------------------------------",
		"------------",
		"------------",
		"----------"))

	for _, config := range testConfigs {
		var totalBuildTime time.Duration

		for sample := 0; sample < config.samples; sample++ {
			vectors := generateRandomVectors(config.size, config.dimensions, int64(42+sample))

			start := time.Now()
			index := hnsw.NewCosine(config.dimensions, config.size*2, 32, 400, 42)

			for i, vec := range vectors {
				if err := index.Add(vec, uint64(i)); err != nil {
					t.Fatalf("Failed to add vector: %v", err)
				}
			}

			buildTime := time.Since(start)
			totalBuildTime += buildTime
			index.Close()
		}

		avgBuildTime := totalBuildTime / time.Duration(config.samples)
		perVectorTime := avgBuildTime / time.Duration(config.size)
		vectorsPerSec := float64(config.size) / avgBuildTime.Seconds()

		fmt.Printf("%-30s | %12v | %12v | %10.0f\n",
			config.name, avgBuildTime, perVectorTime, vectorsPerSec)
	}
}

func TestIndexSaveTime(t *testing.T) {
	fmt.Println("\nHNSW Index Save Time Benchmark")
	fmt.Println("==============================")
	fmt.Printf("%-30s | %-12s | %-12s | %-12s\n", "Configuration", "Save Time", "File Size", "Save Rate")
	fmt.Println(fmt.Sprintf("%s|%s|%s|%s",
		"------------------------------",
		"------------",
		"------------",
		"------------"))

	tempDir, err := os.MkdirTemp("", "hnsw_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	for _, config := range testConfigs {
		var totalSaveTime time.Duration
		var totalFileSize int64

		for sample := 0; sample < config.samples; sample++ {
			vectors := generateRandomVectors(config.size, config.dimensions, int64(42+sample))

			// Build index
			index := hnsw.NewCosine(config.dimensions, config.size*2, 32, 400, 42)
			for i, vec := range vectors {
				if err := index.Add(vec, uint64(i)); err != nil {
					t.Fatalf("Failed to add vector: %v", err)
				}
			}

			// Benchmark save time
			filename := filepath.Join(tempDir, fmt.Sprintf("test_%d_%d_%d.hnsw",
				config.dimensions, config.size, sample))

			start := time.Now()
			if err := index.Save(filename); err != nil {
				t.Fatalf("Failed to save index: %v", err)
			}
			saveTime := time.Since(start)
			totalSaveTime += saveTime

			// Measure file size
			fileInfo, err := os.Stat(filename)
			if err != nil {
				t.Fatalf("Failed to stat file: %v", err)
			}
			totalFileSize += fileInfo.Size()

			index.Close()
			os.Remove(filename) // Clean up immediately
		}

		avgSaveTime := totalSaveTime / time.Duration(config.samples)
		avgFileSize := totalFileSize / int64(config.samples)
		saveRateMBps := float64(avgFileSize) / (1024 * 1024) / avgSaveTime.Seconds()

		fmt.Printf("%-30s | %12v | %12s | %10.1f MB/s\n",
			config.name, avgSaveTime, formatBytes(avgFileSize), saveRateMBps)
	}
}

func TestIndexLoadTime(t *testing.T) {
	fmt.Println("\nHNSW Index Load Time Benchmark")
	fmt.Println("==============================")
	fmt.Printf("%-30s | %-12s | %-12s | %-12s\n", "Configuration", "Load Time", "File Size", "Load Rate")
	fmt.Println(fmt.Sprintf("%s|%s|%s|%s",
		"------------------------------",
		"------------",
		"------------",
		"------------"))

	tempDir, err := os.MkdirTemp("", "hnsw_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	for _, config := range testConfigs {
		var totalLoadTime time.Duration
		var totalFileSize int64

		for sample := 0; sample < config.samples; sample++ {
			vectors := generateRandomVectors(config.size, config.dimensions, int64(42+sample))

			// Build and save index
			index := hnsw.NewCosine(config.dimensions, config.size*2, 32, 400, 42)
			for i, vec := range vectors {
				if err := index.Add(vec, uint64(i)); err != nil {
					t.Fatalf("Failed to add vector: %v", err)
				}
			}

			filename := filepath.Join(tempDir, fmt.Sprintf("test_%d_%d_%d.hnsw",
				config.dimensions, config.size, sample))

			if err := index.Save(filename); err != nil {
				t.Fatalf("Failed to save index: %v", err)
			}
			index.Close()

			// Measure file size
			fileInfo, err := os.Stat(filename)
			if err != nil {
				t.Fatalf("Failed to stat file: %v", err)
			}
			totalFileSize += fileInfo.Size()

			// Benchmark load time
			start := time.Now()
			loadedIndex, err := hnsw.Load(hnsw.SpaceCosine, config.dimensions, filename)
			if err != nil {
				t.Fatalf("Failed to load index: %v", err)
			}
			loadTime := time.Since(start)
			totalLoadTime += loadTime

			loadedIndex.Close()
			os.Remove(filename)
		}

		avgLoadTime := totalLoadTime / time.Duration(config.samples)
		avgFileSize := totalFileSize / int64(config.samples)
		loadRateMBps := float64(avgFileSize) / (1024 * 1024) / avgLoadTime.Seconds()

		fmt.Printf("%-30s | %12v | %12s | %10.1f MB/s\n",
			config.name, avgLoadTime, formatBytes(avgFileSize), loadRateMBps)
	}
}

func TestIndexSizeAnalysis(t *testing.T) {
	fmt.Println("\nHNSW Index Size Analysis")
	fmt.Println("========================")
	fmt.Printf("%-30s | %-12s | %-12s | %-12s | %-8s\n",
		"Configuration", "Raw Data", "Index Size", "Overhead", "Ratio")
	fmt.Println(fmt.Sprintf("%s|%s|%s|%s|%s",
		"------------------------------",
		"------------",
		"------------",
		"------------",
		"--------"))

	tempDir, err := os.MkdirTemp("", "hnsw_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	for _, config := range testConfigs {
		vectors := generateRandomVectors(config.size, config.dimensions, 42)

		// Calculate raw data size
		rawDataSize := int64(config.size * config.dimensions * 4) // 4 bytes per float32

		// Build and save index
		index := hnsw.NewCosine(config.dimensions, config.size*2, 32, 400, 42)
		for i, vec := range vectors {
			if err := index.Add(vec, uint64(i)); err != nil {
				t.Fatalf("Failed to add vector: %v", err)
			}
		}

		filename := filepath.Join(tempDir, fmt.Sprintf("size_test_%d_%d.hnsw",
			config.dimensions, config.size))

		if err := index.Save(filename); err != nil {
			t.Fatalf("Failed to save index: %v", err)
		}
		index.Close()

		// Measure file size
		fileInfo, err := os.Stat(filename)
		if err != nil {
			t.Fatalf("Failed to stat file: %v", err)
		}
		indexSize := fileInfo.Size()

		overhead := indexSize - rawDataSize
		ratio := float64(indexSize) / float64(rawDataSize)

		fmt.Printf("%-30s | %12s | %12s | %12s | %7.1fx\n",
			config.name,
			formatBytes(rawDataSize),
			formatBytes(indexSize),
			formatBytes(overhead),
			ratio)

		os.Remove(filename)
	}
}

func BenchmarkIndexConstruction(b *testing.B) {
	configs := []struct {
		name       string
		dimensions int
		size       int
	}{
		{"D100_N1K", 100, 1000},
		{"D300_N5K", 300, 5000},
		{"D1024_N10K", 1024, 10000},
	}

	for _, config := range configs {
		b.Run(config.name, func(b *testing.B) {
			vectors := generateRandomVectors(config.size, config.dimensions, 42)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				index := hnsw.NewCosine(config.dimensions, config.size*2, 32, 400, 42)

				for j, vec := range vectors {
					if err := index.Add(vec, uint64(j)); err != nil {
						b.Fatal(err)
					}
				}

				index.Close()
			}
		})
	}
}

// generateRandomVectors is defined in comprehensive_benchmark.go

func formatBytes(bytes int64) string {
	if bytes < 1024 {
		return fmt.Sprintf("%d B", bytes)
	} else if bytes < 1024*1024 {
		return fmt.Sprintf("%.1f KB", float64(bytes)/1024)
	} else if bytes < 1024*1024*1024 {
		return fmt.Sprintf("%.1f MB", float64(bytes)/(1024*1024))
	} else {
		return fmt.Sprintf("%.1f GB", float64(bytes)/(1024*1024*1024))
	}
}
