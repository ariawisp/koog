# Koog Memory Benchmark Suite

This module provides a comprehensive benchmark suite for evaluating Koog's memory system against industry standards.

## Structure

```
agents-benchmark/
├── src/                      # Benchmark implementation
│   ├── model/               # Data models (QAItem, Dataset, etc.)
│   ├── judge/               # Answer evaluation (SimpleJudge, LlmJudge)
│   ├── retrieval/           # Retrieval adapters (Graph, Hybrid, Vector)
│   └── runner/              # Execution framework
├── datasets/
│   ├── raw/                 # Original dataset formats
│   └── koog/                # Normalized JSONL format
└── tools/
    ├── koog_ds_convert.py   # Dataset converter
    └── koog_ds_check.py     # Format validator
```

## Quick Start

### 1. Import Datasets

Place raw datasets in `datasets/raw/`:
- Letta: JSONL files with questions/answers
- LongMemEval: conversations.json + qas.json
- Mem0: JSON files with Q&A items
- KG-LM: questions.json with KPI queries

### 2. Convert to Koog Format

```bash
cd agents/agents-benchmark
python tools/koog_ds_convert.py --src datasets/raw/letta --dst datasets/koog --dataset letta
python tools/koog_ds_convert.py --src datasets/raw/longmemeval --dst datasets/koog --dataset longmemeval
python tools/koog_ds_convert.py --src datasets/raw/mem0 --dst datasets/koog --dataset mem0
python tools/koog_ds_convert.py --src datasets/raw/kg-lm --dst datasets/koog --dataset kg-lm
```

### 3. Verify Datasets

```bash
python tools/koog_ds_check.py datasets/koog/letta.test.jsonl
```

### 4. Run Benchmarks

```kotlin
// In your Kotlin code
val dataset = DatasetLoader.loadFromJsonl(
    File("datasets/koog/letta.test.jsonl").readText(),
    "letta"
)

val executor = BenchmarkExecutor(
    tokenizer = yourTokenizer,
    graphProvider = yourGraphProvider,
    smartRouter = yourSmartRouter
)

val report = executor.runAllModes(dataset)
executor.printReport(report)
```

## Dataset Format

All datasets are normalized to JSONL with this schema:

```json
{
  "id": "unique-id",
  "dataset": "letta|longmemeval|mem0|kg-lm",
  "split": "train|dev|test",
  "type": "qa|temporal|multi_session|update",
  "question": "The question text",
  "gold_answer": "Expected answer",
  "evidence": [
    {"text": "Supporting text", "source": "file.txt", "meta": {}}
  ],
  "meta": {
    "session_id": "optional-session",
    "reasoning_steps": ["step1", "step2"],
    "contradiction": {...}
  }
}
```

## Metrics

The benchmark suite tracks:
- **Accuracy**: LLM-as-judge or exact match
- **Token Usage**: Average tokens per query
- **Latency**: p50, p95, p99
- **Token Efficiency**: Accuracy per token ratio

## Extending

To add a new dataset:
1. Create a mapper function in `koog_ds_convert.py`
2. Add the dataset choice to argparse
3. Implement the conversion logic
4. Test with `koog_ds_check.py`