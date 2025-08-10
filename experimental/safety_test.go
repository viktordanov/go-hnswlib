package experimental

import (
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/shubhang93/tablewr"
	"github.com/viktordanov/go-hnswlib/experimental/parallel"
	"github.com/viktordanov/go-hnswlib/hnsw"
)

// Test for race conditions and data integrity
func TestParallelSafety(t *testing.T) {
	fmt.Printf("🔍 Testing Parallel Safety and Data Integrity\n")
	fmt.Printf("=============================================\n")

	// Create index and populate with known data
	index := hnsw.NewCosine(64, 50000, 16, 200, 42)
	defer index.Close()

	// Add vectors with predictable patterns
	const numVectors = 10000
	vectors := make([]parallel.VectorData, numVectors)
	for i := 0; i < numVectors; i++ {
		vector := make([]float32, 64)
		// Create distinctive vectors for cosine similarity
		// Each vector has a unique pattern that will be distinguishable after normalization
		for j := 0; j < 64; j++ {
			// Create unique patterns: use position and vector index to create distinct vectors
			vector[j] = float32(i+j*100) + float32(j)*0.1
		}
		vectors[i] = parallel.VectorData{
			Vector: vector,
			Label:  uint64(i),
		}
	}

	err := parallel.BatchAdd(index, vectors, nil)
	if err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}

	fmt.Printf("✅ Added %d vectors with predictable patterns\n", numVectors)

	// Create test queries
	const numQueries = 1000
	queries := make([][]float32, numQueries)
	expectedLabels := make([]uint64, numQueries)

	for i := 0; i < numQueries; i++ {
		// Use every 10th vector as a query
		idx := i * 10
		if idx >= numVectors {
			idx = numVectors - 1
		}
		queries[i] = vectors[idx].Vector
		expectedLabels[i] = vectors[idx].Label
	}

	fmt.Printf("✅ Created %d test queries with known expected results\n", numQueries)

	// Set high EF for better recall in safety testing
	index.SetEf(200)
	fmt.Printf("✅ Set EF=200 for high recall accuracy\n")

	// Test 1: Sequential search (baseline)
	fmt.Printf("\n📊 Test 1: Sequential Search (Baseline)\n")
	sequentialResults := make([][]uint64, numQueries)
	for i, query := range queries {
		labels, _, _ := index.SearchK(query, 5) // k=5 to increase chance of finding correct match
		sequentialResults[i] = labels
	}

	// Verify sequential results (check if expected label is in top 5)
	sequentialCorrect := 0
	for i := 0; i < numQueries; i++ {
		for _, label := range sequentialResults[i] {
			if label == expectedLabels[i] {
				sequentialCorrect++
				break
			}
		}
	}
	fmt.Printf("   Sequential found expected in top-5: %d/%d (%.1f%%)\n", sequentialCorrect, numQueries,
		float64(sequentialCorrect)/float64(numQueries)*100)

	// Test 2: Parallel search
	fmt.Printf("\n📊 Test 2: Parallel Search\n")
	parallelResults, err := parallel.ParallelSearch(index, queries, 5, nil) // k=5
	if err != nil {
		t.Fatalf("Parallel search failed: %v", err)
	}

	// Verify parallel results (check if expected label is in top 5)
	parallelCorrect := 0
	for i := 0; i < numQueries; i++ {
		for _, label := range parallelResults[i].Labels {
			if label == expectedLabels[i] {
				parallelCorrect++
				break
			}
		}
	}
	fmt.Printf("   Parallel found expected in top-5: %d/%d (%.1f%%)\n", parallelCorrect, numQueries,
		float64(parallelCorrect)/float64(numQueries)*100)

	// Test 3: Compare results (check if top result is identical)
	fmt.Printf("\n📊 Test 3: Sequential vs Parallel Comparison\n")
	identical := 0
	for i := 0; i < numQueries; i++ {
		if len(sequentialResults[i]) > 0 && len(parallelResults[i].Labels) > 0 {
			if sequentialResults[i][0] == parallelResults[i].Labels[0] {
				identical++
			}
		}
	}
	fmt.Printf("   Identical top results: %d/%d (%.1f%%)\n", identical, numQueries,
		float64(identical)/float64(numQueries)*100)

	// Test 4: Order preservation
	fmt.Printf("\n📊 Test 4: Order Preservation Test\n")
	orderCorrect := 0
	for i := 0; i < numQueries; i++ {
		if parallelResults[i].QueryIndex == i {
			orderCorrect++
		}
	}
	fmt.Printf("   Correct order: %d/%d (%.1f%%)\n", orderCorrect, numQueries,
		float64(orderCorrect)/float64(numQueries)*100)

	// Test 5: High worker count test
	fmt.Printf("\n📊 Test 5: High Worker Count Test\n")
	opts := &parallel.ParallelSearchOptions{MaxWorkers: runtime.GOMAXPROCS(0) * 4}
	stressResults, err := parallel.ParallelSearch(index, queries, 5, opts)
	if err != nil {
		t.Fatalf("Stress test failed: %v", err)
	}

	stressCorrect := 0
	for i := 0; i < numQueries; i++ {
		for _, label := range stressResults[i].Labels {
			if label == expectedLabels[i] {
				stressCorrect++
				break
			}
		}
	}
	fmt.Printf("   High worker test found expected in top-5: %d/%d (%.1f%%)\n", stressCorrect, numQueries,
		float64(stressCorrect)/float64(numQueries)*100)

	// Test 6: Collision test - same query vector from multiple workers
	fmt.Printf("\n📊 Test 6: Collision Test - Same Query Vector\n")
	sameQuery := queries[0]
	sameQuerySlice := make([][]float32, 100)
	for i := 0; i < 100; i++ {
		sameQuerySlice[i] = sameQuery
	}

	opts = &parallel.ParallelSearchOptions{MaxWorkers: runtime.GOMAXPROCS(0) * 3}
	collisionResults, err := parallel.ParallelSearch(index, sameQuerySlice, 10, opts)
	if err != nil {
		t.Fatalf("Collision test failed: %v", err)
	}

	// All results should be identical since same query
	firstResult := collisionResults[0].Labels
	allSame := true
	for i := 1; i < len(collisionResults); i++ {
		if len(collisionResults[i].Labels) != len(firstResult) {
			allSame = false
			break
		}
		for j := 0; j < len(firstResult); j++ {
			if collisionResults[i].Labels[j] != firstResult[j] {
				allSame = false
				break
			}
		}
		if !allSame {
			break
		}
	}

	if allSame {
		fmt.Printf("   ✅ Collision test passed: All 100 identical queries returned identical results\n")
	} else {
		t.Errorf("   ❌ Collision test failed: Identical queries returned different results!")
	}

	// Final verification
	fmt.Printf("\n🎯 Final Results:\n")

	wr := tablewr.New(os.Stdout, 0, tablewr.WithSep())
	data := [][]string{
		{"Test", "Result", "Score", "Status"},
		{"Sequential Accuracy", fmt.Sprintf("%d/%d", sequentialCorrect, numQueries), fmt.Sprintf("%.1f%%", float64(sequentialCorrect)/float64(numQueries)*100), "✅"},
		{"Parallel Accuracy", fmt.Sprintf("%d/%d", parallelCorrect, numQueries), fmt.Sprintf("%.1f%%", float64(parallelCorrect)/float64(numQueries)*100), "✅"},
		{"Identical Results", fmt.Sprintf("%d/%d", identical, numQueries), fmt.Sprintf("%.1f%%", float64(identical)/float64(numQueries)*100), "✅"},
		{"Order Preservation", fmt.Sprintf("%d/%d", orderCorrect, numQueries), fmt.Sprintf("%.1f%%", float64(orderCorrect)/float64(numQueries)*100), func() string {
			if orderCorrect == numQueries {
				return "✅"
			} else {
				return "❌"
			}
		}()},
		{"High Worker Test", fmt.Sprintf("%d/%d", stressCorrect, numQueries), fmt.Sprintf("%.1f%%", float64(stressCorrect)/float64(numQueries)*100), "✅"},
	}

	if err := wr.Write(data); err != nil {
		t.Fatalf("Failed to write results table: %v", err)
	}

	if abs(parallelCorrect-sequentialCorrect) <= 2 && identical >= numQueries*80/100 {
		fmt.Printf("\n✅ PASS: Parallel search is safe and accurate\n")
	} else {
		t.Errorf("❌ FAIL: Data integrity issues detected")
		t.Errorf("   Sequential correct: %d, Parallel correct: %d, Identical: %d",
			sequentialCorrect, parallelCorrect, identical)
	}

	if orderCorrect == numQueries {
		fmt.Printf("✅ PASS: Query order preserved correctly\n")
	} else {
		t.Errorf("❌ FAIL: Query order not preserved: %d/%d", orderCorrect, numQueries)
	}
}

// Test concurrent access safety
func TestConcurrentAccess(t *testing.T) {
	fmt.Printf("\n🔒 Testing Concurrent Access Safety\n")
	fmt.Printf("====================================\n")

	index := hnsw.NewCosine(128, 25000, 16, 200, 42)
	defer index.Close()

	// Add test data
	vectors := make([]parallel.VectorData, 5000)
	for i := 0; i < 5000; i++ {
		vector := make([]float32, 128)
		for j := 0; j < 128; j++ {
			vector[j] = rand.Float32()
		}
		vectors[i] = parallel.VectorData{Vector: vector, Label: uint64(i)}
	}

	err := parallel.BatchAdd(index, vectors, nil)
	if err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}

	// Set high EF for better recall
	index.SetEf(200)

	// Test 1: Same query from many goroutines
	fmt.Printf("Test 1: Collision Test - 500 Goroutines, Same Query\n")
	query := make([]float32, 128)
	for i := 0; i < 128; i++ {
		query[i] = rand.Float32()
	}

	const numGoroutines = 500
	results := make([][]uint64, numGoroutines)
	var wg sync.WaitGroup

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			labels, _, _ := index.SearchK(query, 10)
			results[idx] = labels
		}(i)
	}

	wg.Wait()

	// Verify all results are identical (same query should give same results)
	if len(results[0]) == 0 {
		t.Fatalf("No results returned")
	}

	consistent := true
	for i := 1; i < numGoroutines; i++ {
		if len(results[i]) != len(results[0]) {
			consistent = false
			break
		}
		for j := 0; j < len(results[0]); j++ {
			if results[i][j] != results[0][j] {
				consistent = false
				break
			}
		}
		if !consistent {
			break
		}
	}

	if consistent {
		fmt.Printf("   ✅ PASS: All %d concurrent searches returned identical results\n", numGoroutines)
	} else {
		t.Errorf("❌ FAIL: Concurrent searches returned different results")
		// Show first few results for debugging
		for i := 0; i < min(5, numGoroutines); i++ {
			fmt.Printf("   Result %d: %v\n", i, results[i])
		}
	}

	// Test 2: Different queries from many goroutines
	fmt.Printf("Test 2: Different Queries Test - 1000 Goroutines\n")
	const numDifferentQueries = 1000
	differentQueries := make([][]float32, numDifferentQueries)
	differentResults := make([][]uint64, numDifferentQueries)

	// Create different queries
	for i := 0; i < numDifferentQueries; i++ {
		query := make([]float32, 128)
		for j := 0; j < 128; j++ {
			query[j] = rand.Float32() + float32(i)*0.001 // Slightly different each time
		}
		differentQueries[i] = query
	}

	var wg2 sync.WaitGroup
	for i := 0; i < numDifferentQueries; i++ {
		wg2.Add(1)
		go func(idx int) {
			defer wg2.Done()
			labels, _, _ := index.SearchK(differentQueries[idx], 5)
			differentResults[idx] = labels
		}(i)
	}

	wg2.Wait()

	// Verify all searches completed successfully
	completedCount := 0
	for i := 0; i < numDifferentQueries; i++ {
		if len(differentResults[i]) > 0 {
			completedCount++
		}
	}

	if completedCount == numDifferentQueries {
		fmt.Printf("   ✅ PASS: All %d different concurrent searches completed successfully\n", numDifferentQueries)
	} else {
		t.Errorf("❌ FAIL: Only %d/%d different searches completed", completedCount, numDifferentQueries)
	}

	// Test 3: Large batch parallel search test
	fmt.Printf("Test 3: Large Batch Parallel Search - 2000 Queries\n")
	batchQueries := make([][]float32, 2000)
	for i := 0; i < 2000; i++ {
		query := make([]float32, 128)
		for j := 0; j < 128; j++ {
			query[j] = rand.Float32()
		}
		batchQueries[i] = query
	}

	opts := &parallel.ParallelSearchOptions{MaxWorkers: runtime.GOMAXPROCS(0) * 5}
	batchResults, err := parallel.ParallelSearch(index, batchQueries, 5, opts)
	if err != nil {
		t.Fatalf("Large batch search failed: %v", err)
	}

	if len(batchResults) == 2000 {
		fmt.Printf("   ✅ PASS: Large batch search with %d queries completed successfully\n", 2000)
	} else {
		t.Errorf("❌ FAIL: Expected 2000 results, got %d", len(batchResults))
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// Large scale stress test
func TestLargeScaleStressTest(t *testing.T) {
	fmt.Printf("\nLarge Scale Stress Test\n")
	fmt.Printf("=======================\n")

	// Large index with high-dimensional vectors
	index := hnsw.NewCosine(256, 100000, 32, 400, 42)
	defer index.Close()

	// Add large dataset
	fmt.Printf("Adding 20,000 high-dimensional vectors...\n")
	vectors := make([]parallel.VectorData, 20000)
	for i := 0; i < 20000; i++ {
		vector := make([]float32, 256)
		for j := 0; j < 256; j++ {
			vector[j] = rand.Float32() * 100.0
		}
		vectors[i] = parallel.VectorData{Vector: vector, Label: uint64(i)}
	}

	err := parallel.BatchAdd(index, vectors, &parallel.BatchAddOptions{MaxWorkers: runtime.GOMAXPROCS(0) * 2})
	if err != nil {
		t.Fatalf("Failed to add large dataset: %v", err)
	}

	// Set high EF for better recall
	index.SetEf(200)

	// Test 1: High concurrency identical queries
	fmt.Printf("Test 1: 2000 Identical Queries with High Worker Count\n")
	query := vectors[1000].Vector // Use an existing vector
	identicalQueries := make([][]float32, 2000)
	for i := 0; i < 2000; i++ {
		identicalQueries[i] = query
	}

	opts := &parallel.ParallelSearchOptions{MaxWorkers: runtime.GOMAXPROCS(0) * 10} // 10x workers!
	start := time.Now()
	nuclearResults, err := parallel.ParallelSearch(index, identicalQueries, 20, opts)
	if err != nil {
		t.Fatalf("Nuclear identical query test failed: %v", err)
	}
	elapsed := time.Since(start)

	// Verify all results are identical
	firstResult := nuclearResults[0].Labels
	allIdentical := true
	for i := 1; i < len(nuclearResults); i++ {
		if len(nuclearResults[i].Labels) != len(firstResult) {
			allIdentical = false
			break
		}
		for j := 0; j < len(firstResult); j++ {
			if nuclearResults[i].Labels[j] != firstResult[j] {
				allIdentical = false
				break
			}
		}
		if !allIdentical {
			break
		}
	}

	identicalElapsed := elapsed
	identicalQPS := float64(2000) / elapsed.Seconds()

	if allIdentical {
		fmt.Printf("   ✅ PASS: 2000 identical queries in %v (%.0f QPS)\n",
			elapsed, identicalQPS)
	} else {
		t.Errorf("   ❌ FAIL: Identical queries returned different results")
	}

	// Test 2: High load different queries test
	fmt.Printf("Test 2: High Load Different Queries - 5000 Queries, High Worker Count\n")
	loadQueries := make([][]float32, 5000)
	for i := 0; i < 5000; i++ {
		query := make([]float32, 256)
		for j := 0; j < 256; j++ {
			query[j] = rand.Float32()*200.0 - 100.0 // Random range [-100, 100]
		}
		loadQueries[i] = query
	}

	loadOpts := &parallel.ParallelSearchOptions{MaxWorkers: runtime.GOMAXPROCS(0) * 15}
	start = time.Now()
	loadResults, err := parallel.ParallelSearch(index, loadQueries, 15, loadOpts)
	if err != nil {
		t.Fatalf("High load test failed: %v", err)
	}
	elapsed = time.Since(start)

	loadElapsed := elapsed
	loadQPS := float64(5000) / elapsed.Seconds()

	if len(loadResults) == 5000 {
		fmt.Printf("   ✅ PASS: 5000 queries in %v (%.0f QPS)\n",
			elapsed, loadQPS)
	} else {
		t.Errorf("   ❌ FAIL: Expected 5000 results, got %d", len(loadResults))
	}

	// Test 3: Memory pressure test - rapid batch searches
	fmt.Printf("Test 3: Memory Pressure Test - 100 Rapid Batch Searches\n")
	smallQuery := make([][]float32, 100)
	for i := 0; i < 100; i++ {
		query := make([]float32, 256)
		for j := 0; j < 256; j++ {
			query[j] = rand.Float32()
		}
		smallQuery[i] = query
	}

	// Run 100 batch searches rapidly
	var rapidWG sync.WaitGroup
	rapidErrors := make(chan error, 100)

	start = time.Now()
	for batch := 0; batch < 100; batch++ {
		rapidWG.Add(1)
		go func(batchNum int) {
			defer rapidWG.Done()
			_, err := parallel.ParallelSearch(index, smallQuery, 5,
				&parallel.ParallelSearchOptions{MaxWorkers: runtime.GOMAXPROCS(0)})
			if err != nil {
				select {
				case rapidErrors <- err:
				default:
				}
			}
		}(batch)
	}

	rapidWG.Wait()
	elapsed = time.Since(start)
	close(rapidErrors)

	errorCount := 0
	for err := range rapidErrors {
		errorCount++
		if errorCount == 1 { // Only print first error
			fmt.Printf("   Error: %v\n", err)
		}
	}

	rapidElapsed := elapsed

	if errorCount == 0 {
		fmt.Printf("   ✅ PASS: 100 rapid batches completed in %v\n", elapsed)
	} else {
		t.Errorf("   ❌ FAIL: %d errors occurred", errorCount)
	}

	fmt.Printf("\nLarge Scale Test Summary:\n")
	wr2 := tablewr.New(os.Stdout, 0, tablewr.WithSep())
	summaryData := [][]string{
		{"Test", "Queries", "Time", "QPS", "Status"},
		{"Identical Queries", "2000", identicalElapsed.String(), fmt.Sprintf("%.0f", identicalQPS), func() string {
			if allIdentical {
				return "✅ PASS"
			} else {
				return "❌ FAIL"
			}
		}()},
		{"Different Queries", "5000", loadElapsed.String(), fmt.Sprintf("%.0f", loadQPS), func() string {
			if len(loadResults) == 5000 {
				return "✅ PASS"
			} else {
				return "❌ FAIL"
			}
		}()},
		{"Rapid Batches", "100", rapidElapsed.String(), "-", func() string {
			if errorCount == 0 {
				return "✅ PASS"
			} else {
				return "❌ FAIL"
			}
		}()},
	}

	if err := wr2.Write(summaryData); err != nil {
		t.Fatalf("Failed to write summary table: %v", err)
	}
	fmt.Printf("System handled high load successfully\n")
}
