#!/bin/bash

echo "🚀 Koog Benchmark Runner"
echo "========================"
echo ""
echo "This script runs the Koog memory system benchmark"
echo ""

# Default values
SIZE=10
OUTPUT=""
DATASET=""
LMSTUDIO_URL="http://localhost:1234"
RUNS=3
SYSTEMS="all"
QUICK=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --size)
      SIZE="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --lmstudio-url)
      LMSTUDIO_URL="$2"
      shift 2
      ;;
    --runs)
      RUNS="$2"
      shift 2
      ;;
    --systems)
      SYSTEMS="$2"
      shift 2
      ;;
    --quick)
      QUICK=true
      shift
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --help)
      echo "Usage: ./run-benchmark.sh [options]"
      echo ""
      echo "Options:"
      echo "  --size <number>       Number of questions to evaluate (default: 10)"
      echo "  --output <path>       Output file for results (default: console only)"
      echo "  --dataset <path>      Path to dataset file (JSONL or LettaBench format)"
      echo "  --lmstudio-url <url>  LMStudio URL (default: http://localhost:1234)"
      echo "  --runs <number>       Number of evaluation runs (default: 3)"
      echo "  --systems <list>      Comma-separated systems: all,node-distance,rrf,vector (default: all)"
      echo "  --quick               Quick mode - single run, no warmup, faster testing"
      echo "  --help                Show this help message"
      echo ""
      echo "Examples:"
      echo "  ./run-benchmark.sh --size 50 --output results.json"
      echo "  ./run-benchmark.sh --quick --size 3              # Quick test with 3 questions"
      echo "  ./run-benchmark.sh --systems node-distance --runs 1"
      echo "  ./run-benchmark.sh --dataset lettabench.jsonl --size 5"
      echo "  ./run-benchmark.sh --lmstudio-url http://192.168.1.100:1234"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Build the program arguments
PROGRAM_ARGS="--size=$SIZE --lmstudio-url=$LMSTUDIO_URL --runs=$RUNS --systems=$SYSTEMS"

if [ -n "$OUTPUT" ]; then
  PROGRAM_ARGS="$PROGRAM_ARGS --output=$OUTPUT"
fi

if [ -n "$DATASET" ]; then
  PROGRAM_ARGS="$PROGRAM_ARGS --dataset=$DATASET"  
fi

if [ "$QUICK" = true ]; then
  PROGRAM_ARGS="$PROGRAM_ARGS --quick"
fi

echo "Configuration:"
echo "  LMStudio URL: $LMSTUDIO_URL"
echo "  Dataset: ${DATASET:-synthetic}"
echo "  Dataset size: $SIZE questions"
echo "  Output: ${OUTPUT:-console only}"
echo "  Runs: $RUNS"
echo "  Systems: $SYSTEMS"
if [ "$QUICK" = true ]; then
  echo "  Mode: QUICK (minimal evaluation)"
fi
echo ""

echo ""
echo "⚠️  Make sure LMStudio is running at $LMSTUDIO_URL"
echo ""

echo "Running benchmark..."
echo ""

# Run the benchmark
cd ../.. && ./gradlew :agents:agents-benchmark:runBenchmark -Pargs="$PROGRAM_ARGS"