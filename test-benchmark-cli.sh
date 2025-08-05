#!/bin/bash

echo "Testing Koog Benchmark CLI"
echo "========================="
echo ""

# Test help command
echo "1. Testing help command:"
./gradlew agents:agents-benchmark:runBenchmark --args="--help"

echo ""
echo "2. Testing list command:"
./gradlew agents:agents-benchmark:runBenchmark --args="list"

echo ""
echo "3. Testing quick command help:"
./gradlew agents:agents-benchmark:runBenchmark --args="quick --help"

echo ""
echo "Note: To run actual benchmarks, you need LMStudio running at localhost:1234"
echo "Example commands:"
echo "  ./gradlew agents:agents-benchmark:runBenchmark --args='quick'"
echo "  ./gradlew agents:agents-benchmark:runBenchmark --args='run --size 10 --quick'"
echo "  ./gradlew agents:agents-benchmark:runBenchmark --args='run --size 50 --parallel 4'"