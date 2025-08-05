# Multi-Provider Benchmark Guide

The Koog benchmark system now supports easy switching between LLM providers for true apples-to-apples comparisons.

## Supported Providers

1. **LMStudio** (default) - Local LLMs
2. **OpenAI** - GPT-4o, GPT-4o-mini, etc.
3. **Anthropic** - Claude 3 family

## Quick Start

### 1. Check Available Providers
```bash
./gradlew agents:agents-benchmark:runBenchmark --args="providers" -q
```

### 2. Using LMStudio (Default)
```bash
# Ensure LMStudio is running at localhost:1234
./gradlew agents:agents-benchmark:runBenchmark --args="run --size 50 --systems all" -q
```

### 3. Using OpenAI
```bash
# Set your API key
export OPENAI_API_KEY=sk-...

# Run with GPT-4o (best quality)
./gradlew agents:agents-benchmark:runBenchmark --args="run --provider openai --model gpt-4o --size 50" -q

# Run with GPT-4o-mini (cheaper, faster)
./gradlew agents:agents-benchmark:runBenchmark --args="run --provider openai --model gpt-4o-mini --size 50" -q
```

### 4. Using Anthropic
```bash
# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# Run with Claude 3 Opus (best quality)
./gradlew agents:agents-benchmark:runBenchmark --args="run --provider anthropic --model claude-3-opus-20240229 --size 50" -q

# Run with Claude 3 Sonnet (balanced)
./gradlew agents:agents-benchmark:runBenchmark --args="run --provider anthropic --model claude-3-sonnet-20240229 --size 50" -q

# Run with Claude 3 Haiku (fastest)
./gradlew agents:agents-benchmark:runBenchmark --args="run --provider anthropic --model claude-3-haiku-20240307 --size 50" -q
```

## Comparing Providers

### Automated Comparison Script
```bash
# This script automatically detects available providers and runs comparisons
./benchmark-with-providers.sh
```

### Manual Comparison
```bash
# 1. Run with LMStudio
./gradlew agents:agents-benchmark:runBenchmark \
  --args="run --size 50 --systems vector --output results-lmstudio.json" -q

# 2. Run with OpenAI GPT-4o-mini
export OPENAI_API_KEY=your-key
./gradlew agents:agents-benchmark:runBenchmark \
  --args="run --provider openai --model gpt-4o-mini --size 50 --systems vector --output results-gpt4-mini.json" -q

# 3. Run with OpenAI GPT-4o (to match Zep's benchmark)
./gradlew agents:agents-benchmark:runBenchmark \
  --args="run --provider openai --model gpt-4o --size 50 --systems vector --output results-gpt4.json" -q
```

## Beating the Benchmarks

To beat Zep's published results:

### 1. LongMemEval (Target: 71.2%)
```bash
# Zep used GPT-4o and achieved 71.2%
# We got 66% with Llama-3.1-8B, so GPT-4o should easily beat 71.2%

export OPENAI_API_KEY=your-key
./gradlew agents:agents-benchmark:runBenchmark \
  --args="run --provider openai --model gpt-4o --size 100 --systems all --parallel 4" -q
```

### 2. Deep Memory Retrieval (Target: 94.8%)
```bash
# Zep used GPT-4-turbo and achieved 94.8%
# This is harder, but worth trying with GPT-4o

./gradlew agents:agents-benchmark:runBenchmark \
  --args="run --provider openai --model gpt-4o --size 100 --systems all" -q
```

## Cost Considerations

| Provider | Model | Cost per 1M tokens | Speed | Quality |
|----------|-------|-------------------|-------|---------|
| LMStudio | Local | Free | Medium | Good |
| OpenAI | GPT-4o-mini | $0.15 input, $0.60 output | Fast | Very Good |
| OpenAI | GPT-4o | $5 input, $15 output | Medium | Excellent |
| Anthropic | Claude-3-Haiku | $0.25 input, $1.25 output | Fast | Good |
| Anthropic | Claude-3-Sonnet | $3 input, $15 output | Medium | Very Good |
| Anthropic | Claude-3-Opus | $15 input, $75 output | Slow | Excellent |

## Environment Variables

```bash
# Set default provider (optional)
export BENCHMARK_LLM_PROVIDER=openai  # or anthropic, lmstudio

# API Keys
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# LMStudio URL (if not localhost:1234)
export LMSTUDIO_URL=http://192.168.1.100:1234

# Default models (optional)
export OPENAI_MODEL=gpt-4o
export ANTHROPIC_MODEL=claude-3-sonnet-20240229
```

## Tips for Apples-to-Apples Comparison

1. **Use the same dataset**: Download official benchmark datasets
2. **Use the same model class**: GPT-4o to match Zep's GPT-4o results
3. **Use sufficient sample size**: At least 50-100 questions
4. **Run multiple times**: Use `--runs 3` for statistical significance
5. **Compare the same metrics**: Accuracy is the primary metric

## Example: Beat Zep's LongMemEval

```bash
# Step 1: Set up OpenAI
export OPENAI_API_KEY=your-key

# Step 2: Run with our best system (Vector Similarity)
./gradlew agents:agents-benchmark:runBenchmark \
  --args="run --provider openai --model gpt-4o --size 100 --systems vector --runs 3" -q

# Step 3: If you get >71.2% accuracy, you've beaten Zep!
```

With GPT-4o, based on our 66% with Llama-3.1-8B, we should easily achieve >71.2% and beat Zep's LongMemEval benchmark!