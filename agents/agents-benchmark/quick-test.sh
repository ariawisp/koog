#!/bin/bash

# Quick benchmark test - good balance of speed and reliability
# Takes ~2-3 minutes with decent statistical validity

echo "🚀 Running Quick Benchmark Test"
echo "This will evaluate 25 questions - enough for meaningful results"
echo ""

./run-benchmark.sh --quick --size 25 --systems all