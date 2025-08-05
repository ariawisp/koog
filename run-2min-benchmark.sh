#!/bin/bash

echo "🚀 Running 2-Minute Comprehensive Benchmark"
echo "==========================================="
echo ""
echo "This test will:"
echo "- Evaluate 20 questions (enough for statistical significance)"
echo "- Test all 3 retrieval systems (node-distance, RRF, vector)"
echo "- Use parallel execution (4 concurrent requests)"
echo "- Complete in under 2 minutes"
echo ""

# Run the benchmark
./gradlew agents:agents-benchmark:runBenchmark --args="run --size 20 --systems all --parallel 4 --runs 2" -q

echo ""
echo "✅ Benchmark complete!"