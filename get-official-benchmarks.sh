#!/bin/bash

echo "📥 Downloading Official Benchmark Datasets"
echo "========================================"
echo ""

# Create datasets directory
mkdir -p agents/agents-benchmark/datasets/official

# 1. LongMemEval Dataset
echo "1. Downloading LongMemEval dataset..."
cd agents/agents-benchmark/datasets/official
wget -q https://drive.google.com/uc?export=download&id=1zJgtYRFhOh5zDQzzatiddfjYhFSnyQ80 -O longmemeval_data.tar.gz
if [ -f longmemeval_data.tar.gz ]; then
    tar -xzf longmemeval_data.tar.gz
    echo "✅ LongMemEval dataset downloaded"
else
    echo "❌ Failed to download LongMemEval"
fi

# 2. Check for LettaBench
echo ""
echo "2. LettaBench information:"
echo "   - We have the data generator in: letta-leaderboard-main/"
echo "   - Can generate questions using: run_letta_file_bench.py"
echo "   - Our existing data in: agents/agents-benchmark/datasets/letta_file_bench/"

# 3. MemGPT DMR Dataset
echo ""
echo "3. Deep Memory Retrieval (DMR) dataset:"
echo "   - Original from MemGPT paper"
echo "   - Check: https://github.com/cpacker/MemGPT"

echo ""
echo "📊 To run apples-to-apples benchmarks:"
echo ""
echo "1. With GPT-4 (via OpenAI API):"
echo "   export OPENAI_API_KEY=your-key"
echo "   ./gradlew agents:agents-benchmark:runBenchmark --args='run --dataset ./longmemeval_data/questions.jsonl --size 100 --systems all'"
echo ""
echo "2. With local LLM (current setup):"
echo "   ./gradlew agents:agents-benchmark:runBenchmark --args='run --dataset ./longmemeval_data/questions.jsonl --size 100 --systems all --parallel 4'"