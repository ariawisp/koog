# Koog Memory System Benchmarking Guide

## Overview

The Koog benchmark system provides rigorous evaluation of the memory retrieval system against published academic baselines. It requires a real LLM connection (via LMStudio) to produce meaningful results.

## Prerequisites

1. **LMStudio** - Download and install from [lmstudio.ai](https://lmstudio.ai)
2. **Local LLM Model** - Load a model in LMStudio (e.g., Llama 3, Mistral, etc.)
3. **LMStudio Server** - Start the local server (usually on `http://localhost:1234`)

## Quick Start

### Basic Benchmark Run

```bash
# Run with synthetic dataset (10 questions)
./agents/agents-benchmark/run-benchmark.sh --size 10

# Run with more questions
./agents/agents-benchmark/run-benchmark.sh --size 100

# Save results to file
./agents/agents-benchmark/run-benchmark.sh --size 50 --output results.json
```

### Using Real Datasets

```bash
# Run with LettaBench dataset
./agents/agents-benchmark/run-benchmark.sh --dataset /path/to/lettabench.jsonl

# Run with custom JSONL dataset
./agents/agents-benchmark/run-benchmark.sh --dataset /path/to/custom.jsonl --size 200
```

### Custom LMStudio URL

```bash
# If LMStudio is running on a different machine/port
./agents/agents-benchmark/run-benchmark.sh --lmstudio-url http://192.168.1.100:1234
```

## Benchmark Datasets

### Supported Formats

1. **LettaBench** - Multi-hop reasoning questions
   - Format: JSONL with complex QA structure
   - Download: [github.com/letta-ai/letta-benchmark](https://github.com/letta-ai/letta-benchmark)

2. **JSONL** - Standard question-answer pairs
   ```json
   {"id": "q1", "question": "What is...", "goldAnswer": "The answer is..."}
   {"id": "q2", "question": "Who did...", "goldAnswer": "Person X did..."}
   ```

3. **JSON** - Full dataset object
   ```json
   {
     "name": "MyDataset",
     "items": [
       {"id": "q1", "question": "...", "goldAnswer": "..."},
       {"id": "q2", "question": "...", "goldAnswer": "..."}
     ]
   }
   ```

## Understanding Results

### Performance Metrics

- **Accuracy**: Percentage of correct answers (with confidence intervals)
- **Latency**: Average response time in milliseconds
- **Tokens**: Average tokens used per question

### Literature Comparison

The system compares against published results from:

1. **Zep** (arXiv:2501.13956v1)
   - Deep Memory Retrieval: 94.8% accuracy
   - LongMemEval: 71.2% accuracy (gpt-4o)

2. **MemGPT** (arXiv:2310.08560)
   - Deep Memory Retrieval: 93.4% accuracy

3. **Other Systems**
   - Full-conversation baseline: 94.4%
   - Conversation summaries: 78.6%

### Interpreting Rankings

- **#1 Ranking**: State-of-the-art performance, ready for publication
- **Top 3**: Competitive performance, suitable for production
- **Top 50%**: Solid performance with room for optimization
- **Bottom 50%**: Needs improvement in retrieval or prompting

## Advanced Usage

### Running Specific Tests

```bash
# Test simple benchmark
./gradlew :agents:agents-benchmark:jvmTest --tests "SimpleBenchmarkTest"

# Test with real LMStudio (if available)
./gradlew :agents:agents-benchmark:jvmTest --tests "RealLLMBenchmarkTest"

# Check submission readiness
./gradlew :agents:agents-benchmark:jvmTest --tests "FullScaleEvaluationTest.testSubmissionReadiness"
```

### Configuring Retrieval Strategies

The benchmark evaluates three retrieval strategies:
1. `HYBRID_NODE_DISTANCE` - Graph-based with distance ranking
2. `HYBRID_RRF` - Reciprocal Rank Fusion
3. `VECTOR_SIMILARITY` - Pure embedding similarity

## Cost Considerations

- **Token Usage**: ~1,500 tokens per question
- **Estimated Cost**: $0.002 per question (GPT-4 mini pricing)
- **Full Benchmark**: 200 questions ≈ $1.20 per run

## Troubleshooting

### LMStudio Connection Issues

```
❌ Failed to connect to LMStudio at http://localhost:1234
```

**Solutions:**
1. Ensure LMStudio is running
2. Check the server is started (look for "Server" tab in LMStudio)
3. Verify the URL matches LMStudio's settings
4. Try `curl http://localhost:1234/v1/models` to test connection

### Out of Memory

For large datasets, increase JVM heap:
```bash
export JAVA_OPTS="-Xmx4g"
./agents/agents-benchmark/run-benchmark.sh --size 1000
```

### Slow Performance

1. Use a smaller/faster model in LMStudio
2. Reduce token generation limits
3. Enable GPU acceleration in LMStudio
4. Run with smaller batch sizes

## Submission to Academic Benchmarks

Once you achieve good results:

1. **Prepare Submission Package**
   - Run full dataset (200+ questions)
   - Ensure 3+ runs for statistical significance
   - Document methodology

2. **Submit to Leaderboards**
   - [Letta Memory Leaderboard](https://github.com/letta-ai/letta-leaderboard)
   - [LongMemEval Benchmark](https://github.com/Buki2/LongMemEval)

3. **Publication Requirements**
   - Include confidence intervals
   - Report hardware specifications
   - Provide reproducible configurations
   - Share code/model details

## Contributing

To add new benchmarks:
1. Implement dataset loader in `io/DatasetLoader.kt`
2. Add published baselines to `literature/LiteratureRepository.kt`
3. Create tests for validation
4. Update this documentation