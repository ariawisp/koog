# Parallel Benchmark Execution

The Koog benchmark system now supports parallel execution for significantly improved performance when evaluating large datasets.

## Features

### Parallel Execution
- **Configurable Concurrency**: Set max concurrent requests with `-p/--parallel` flag
- **Smart Semaphore Control**: Prevents API overload while maximizing throughput
- **Progress Tracking**: Real-time progress updates with time estimates
- **Retry Logic**: Automatic retry with exponential backoff for failed requests
- **Deterministic Ordering**: Results maintain order despite parallel execution

### Performance Benefits
- **3-4x speedup** with 8 concurrent requests (typical)
- **Linear scaling** up to API rate limits
- **Reduced total evaluation time** from hours to minutes

## Usage

### Command Line Interface

The new Clikt-based CLI replaces the old bash scripts:

```bash
# Quick test with parallel execution
koog-benchmark run --quick --parallel 8

# Full benchmark with 4 concurrent requests
koog-benchmark run --size 100 --runs 3 --parallel 4

# Compare specific systems in parallel
koog-benchmark compare node-distance,rrf --parallel 8
```

### CLI Commands

#### `run` - Run full benchmark
```bash
koog-benchmark run [options]
  -s, --size       Number of questions (default: 10)
  -o, --output     Output JSON file  
  -d, --dataset    Dataset file path
  -r, --runs       Number of runs (default: 3)
  -p, --parallel   Max concurrent requests (default: 1)
  -q, --quick      Quick mode (1 run, no warmup)
  --systems        Systems to test (default: all)
```

#### `quick` - Quick benchmark test
```bash
koog-benchmark quick [options]
  -s, --size       Number of questions (default: 25)
  --systems        System to test (default: all)
```

#### `compare` - Compare specific systems
```bash
koog-benchmark compare <systems> [options]
  -d, --dataset    Dataset to use
  -s, --size       Number of questions
```

#### `list` - Show available resources
```bash
koog-benchmark list
```

## Implementation Details

### ParallelBenchmarkExecutor

The new `ParallelBenchmarkExecutor` class provides:

```kotlin
class ParallelBenchmarkExecutor(
    config: BenchmarkConfig = BenchmarkConfig(
        maxConcurrency = 8,      // Parallel by default
        retryAttempts = 3,       // Retry failed requests
        retryDelayMs = 1000,     // Initial retry delay
        progressCallback = null   // Optional progress tracking
    )
)
```

### Architecture

1. **Semaphore-based concurrency control**
   - Limits concurrent API requests
   - Prevents overwhelming LLM endpoints
   
2. **Coroutine-based execution**
   - Efficient resource utilization
   - Non-blocking I/O operations
   
3. **Progress tracking**
   - Real-time completion updates
   - Estimated time remaining
   - Per-system, per-run granularity

4. **Error handling**
   - Automatic retry with backoff
   - Graceful degradation
   - Detailed error reporting

## Performance Guidelines

### Choosing Concurrency Level

- **Local LLMs (LMStudio)**: 2-4 concurrent requests
- **Cloud APIs (OpenAI)**: 8-16 concurrent requests  
- **Rate-limited APIs**: Adjust based on limits

### Memory Considerations

Each concurrent request maintains:
- Question context
- Response buffer
- Retry state

Estimate ~10MB per concurrent request for large contexts.

## Examples

### Basic Parallel Benchmark
```bash
# Evaluate 100 questions with 8 concurrent requests
koog-benchmark run --size 100 --parallel 8
```

### Production Benchmark
```bash
# Full evaluation with optimal settings
koog-benchmark run \
  --dataset lettabench.jsonl \
  --size 1000 \
  --runs 3 \
  --parallel 16 \
  --output results.json
```

### Quick Performance Test
```bash
# Compare sequential vs parallel
time koog-benchmark quick --parallel 1
time koog-benchmark quick --parallel 8
```

## Monitoring

The parallel executor provides real-time progress updates:

```
🚀 Starting Parallel Comparative Benchmark
==================================================
Dataset: LettaBench-100 (100 questions)
Systems: [Koog-HybridNodeDistance]
Max concurrency: 8

📊 Evaluating Koog-HybridNodeDistance...
  ⚡ Warming up with 5 questions...
  🔄 Run 1/3
    Progress: 45/100 (45%) - ETA: 2m 15s
```

## Best Practices

1. **Start with lower concurrency** and increase gradually
2. **Monitor API rate limits** to avoid throttling
3. **Use warmup phase** for stable latency measurements
4. **Enable verbose mode** for debugging parallel issues
5. **Save results regularly** with `--output` flag

## Troubleshooting

### High error rates
- Reduce concurrency level
- Check API rate limits
- Verify network stability

### Inconsistent results
- Ensure sufficient warmup questions
- Check for API throttling
- Verify retry logic is working

### Memory issues
- Reduce concurrency level
- Use smaller batch sizes
- Monitor system resources