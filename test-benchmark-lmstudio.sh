#!/bin/bash

echo "Testing Koog Benchmark with LMStudio"
echo "===================================="
echo ""

# First check if LMStudio is accessible
echo "1. Checking LMStudio connection..."
curl -s http://localhost:1234/v1/models > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ LMStudio is accessible at localhost:1234"
    echo ""
    
    # Show available models
    echo "Available models:"
    curl -s http://localhost:1234/v1/models | jq -r '.data[].id' 2>/dev/null || echo "Could not list models"
    echo ""
else
    echo "❌ Cannot connect to LMStudio at localhost:1234"
    echo "Please ensure LMStudio is running and the local server is started"
    exit 1
fi

# Run a quick benchmark test
echo "2. Running quick benchmark test (5 questions)..."
echo ""
./gradlew agents:agents-benchmark:runBenchmark --args="run --size 5 --quick --systems node-distance" -q

echo ""
echo "3. To run more comprehensive tests, try:"
echo "   ./gradlew agents:agents-benchmark:runBenchmark --args='run --size 25 --systems all' -q"
echo "   ./gradlew agents:agents-benchmark:runBenchmark --args='run --size 50 --parallel 4' -q"