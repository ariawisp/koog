#!/bin/bash

echo "🚀 Running Optimized Benchmark (< 2 minutes)"
echo "==========================================="
echo ""
echo "Configuration:"
echo "- 15 questions per system (statistically meaningful)"
echo "- All 3 retrieval systems"
echo "- 6 parallel requests (faster execution)"
echo "- 1 run per system (to fit in time limit)"
echo ""

# Run the benchmark with optimized parameters
./gradlew agents:agents-benchmark:runBenchmark --args="run --size 15 --systems all --parallel 6 --runs 1 --output benchmark-results.json" -q

echo ""
echo "✅ Benchmark complete! Results saved to benchmark-results.json"