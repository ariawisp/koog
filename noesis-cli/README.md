# Noesis CLI - Unified Command Line Interface

The official CLI for Noesis Runtime, providing comprehensive benchmarking, streaming, and interactive capabilities.

## Installation

```bash
./gradlew :noesis-cli:build
```

## Commands

### Performance Benchmarking

#### `noesis benchmark` - Comprehensive Performance Test
Replaces the Rust benchmark tests (previously in `noesis-runtime/tests/`).

```bash
# Run comprehensive benchmark with scenarios
noesis benchmark --scenarios --runs 3

# Single benchmark 
noesis benchmark --length 500 --runs 5

# With specific model
noesis benchmark --model /path/to/model.bin
```

**Features:**
- 200/500/1000/2000 token scenarios
- Weighted throughput calculation (30%/40%/20%/10%)
- Target validation against 150 tok/s
- Performance consistency metrics
- Live streaming during benchmarks

**Previously validated**: 191.94 tok/s weighted average ✅

#### `noesis tau-bench` - τ-Bench++ World Record Attempt
Ported from `NoesisFirstLightTest.kt` - comprehensive AI accuracy benchmark.

```bash
# Run core τ-Bench++ scenarios
noesis tau-bench

# Extended benchmark with more scenarios
noesis tau-bench --extended

# Quiet mode (results only)
noesis tau-bench --verbose false
```

**Scenarios:**
- Math calculations
- Logical reasoning
- Tool usage chains
- Memory tests
- Planning tasks
- Coding generation
- Analysis tasks
- Translation (extended)

**World Record Criteria:**
- Accuracy ≥ 75%
- Speed ≥ 150 tok/s
- All channels preserved
- Native tool execution

### Interactive Mode

#### `noesis repl` - Interactive REPL
```bash
noesis repl
noesis repl --model /path/to/model.bin --temperature 0.8
```

**Features:**
- Real-time streaming with channel visualization
- Conversation history management
- Commands: `/help`, `/stats`, `/clear`, `/model <path>`
- Performance metrics display

### Streaming

#### `noesis stream` - Single Response Streaming
```bash
noesis stream "Explain quantum computing"
noesis stream "Write a poem" --max-tokens 500 --temperature 0.9
```

### Demo

#### `noesis demo` - Comprehensive Feature Showcase
```bash
noesis demo
```

Demonstrates:
1. Basic real-time streaming
2. Channel-aware streaming
3. Tool call detection
4. Performance showcase

## Migration Notes

### From Rust Tests
All Rust benchmark tests have been migrated to this CLI:
- `noesis-runtime/tests/realistic_performance_benchmark.rs` → `noesis benchmark`
- `noesis-runtime/benches/` → Removed (use CLI instead)

### From JVM Tests
- `NoesisFirstLightTest.kt` → `noesis tau-bench`
  - `testFirstCognition()` → Part of demo command
  - `testFirstToolExecution()` → Tool scenarios in tau-bench
  - `testTauBenchPlusPlus()` → Full tau-bench command

## Performance Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| Throughput | 150+ tok/s | 191.94 tok/s ✅ |
| τ-Bench Accuracy | 75% | TBD |
| Latency | <500ms | TBD |

## Architecture

The CLI uses the NoesisRuntime via JNI, providing:
- Direct access to Metal GPU acceleration
- Real-time token streaming with Harmony channel detection
- Tool execution capabilities
- Performance metrics collection

All benchmarking is now consolidated in this single CLI tool, providing a consistent interface for performance validation and testing.

## Development

To add new benchmarks or scenarios:
1. Extend `TauBenchCommand` for accuracy tests
2. Modify `BenchmarkCommand` for performance tests
3. Add scenarios to `getCoreScenarios()` or `getExtendedScenarios()`

## References

- Original Rust benchmarks: [Deleted - see git history]
- Original test: `prompt/prompt-executor/prompt-executor-clients/noesis-executor/src/jvmTest/`
- Validated performance: 191.94 tok/s (December 2024)