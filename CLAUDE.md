# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains the Koan Agents framework, a Kotlin multiplatform library for building AI agents. The framework enables creating intelligent agents that interact with tools, handle complex workflows, and maintain context across conversations.

## Building and Testing

### Basic Commands

```bash
# Build the project
./gradlew assemble

# Compile test classes
./gradlew jvmTestClasses jsTestClasses

# Run all JVM tests
./gradlew jvmTest

# Run all JS tests
./gradlew jsTest

# Run a specific test class
./gradlew jvmTest --tests "fully.qualified.TestClassName"
# Example:
./gradlew jvmTest --tests "ai.koog.agents.test.SimpleAgentIntegrationTest"

# Run a specific test method
./gradlew jvmTest --tests "fully.qualified.TestClassName.testMethodName"
# Example:
./gradlew jvmTest --tests "ai.koog.agents.test.SimpleAgentIntegrationTest.integration_simpleSingleRunAgentShouldNotCallToolsByDefault"
```

## Architecture

### Key Modules

1. **agents-core**: Core abstractions and interfaces
   - AIAgent, AIAgentStrategy, event handling system, AIAgent, execution strategies, session management

2. **agents-tools**: Tool infrastructure
   - Tool, ToolRegistry, ToolDescriptor

3. **agents-features**: Extensible agent capabilities
   - Memory, tracing, and other features 

4. **prompt**: LLM interaction layer
   - LLM executors, prompt construction, structured data processing

### Core Concepts

- **Agents**: State-machine graphs with nodes that process inputs and produce outputs
- **Tools**: Encapsulated actions with standardized interfaces
- **Strategies**: Define agent behavior and execution flow
- **Features**: Installable extensions that enhance agent capabilities
- **Event Handling**: System for intercepting and processing agent lifecycle events

### Implementation Pattern

1. Define tools that agents can use
2. Register tools in the ToolRegistry
3. Configure agent with strategy
4. Set up communication (if integrating with external systems)

## Testing

The project has extensive testing support:

- **Mocking LLM responses**:
  ```kotlin
  val mockLLMApi = getMockExecutor(toolRegistry, eventHandler) {
      mockLLMAnswer("Hello!") onRequestContains "Hello"
      mockLLMToolCall(CreateTool, CreateTool.Args("solve")) onRequestEquals "Solve task"
  }
  ```

- **Mocking tool calls**:
  ```kotlin
  mockTool(PositiveToneTool) alwaysReturns "The text has a positive tone."
  ```

- **Testing agent graph structure**:
  ```kotlin
  testGraph {
      assertStagesOrder("first", "second")
      // ...
  }
  ```

For detailed testing guidelines, refer to `agents/agents-test/TESTING.md`.

---

# Noesis Runtime Development Guidelines

## Current Transformation Status

The Koog framework is being transformed into **Noesis Runtime** - a Harmony-native, Metal-only reasoning engine. This is an in-place transformation, not a fork.

**Key Achievements:**
- ✅ FlatBuffers multiplatform support (JVM, JS, WASM)
- ✅ Harmony-native message format established
- ✅ Metal GPU inference infrastructure (100+ tok/s)
- ✅ Binary serialization via custom FlatBuffers fork

## Core Development Principles

### 1. Harmony-Native Only
- **HarmonyCore** is THE message type - no alternatives
- All communication uses FlatBuffers binary serialization
- Multi-channel reasoning (Analysis, Commentary, Final) is fundamental
- No JSON APIs or text-based serialization

### 2. Transform In-Place
- Keep class names (AIAgent stays AIAgent, not NoesisAgent)
- Modify existing files rather than creating new ones
- Let compilation errors guide the transformation
- No compatibility layers or migration shims

### 3. Binary-First Architecture
- Token checkpoints use raw ByteArrays, not text
- FlatBuffers for all serialization (already implemented)
- Zero-copy deserialization across platforms
- Direct Metal GPU memory access

### 4. No Provider Abstractions
- Delete all LLM client implementations except Metal/Harmony
- Remove all downsamplers and compatibility layers
- Eliminate provider-specific configurations
- Focus exclusively on GPT-OSS models

## Technical Stack

### Serialization
- **FlatBuffers**: Custom fork at `github.com/ariawisp/flatbuffers`
- Automatic integration via Gradle composite builds
- Supports JVM, JS, WASM/JS, and Native targets

### Inference
- **Metal GPU**: Local inference on Apple Silicon
- **Harmony JNI**: Rust bridge for token processing
- **Target**: 100+ tokens/sec on M2 Ultra

### Key Files to Preserve
- `HarmonyCore.kt` - Core message format
- `HarmonyNativeLLMClient.kt` - Metal executor
- `MetalInferenceJNI.kt` - GPU inference bridge
- FlatBuffers schemas (`harmony_core.fbs`, `token_checkpoint.fbs`)

## Development Workflow

### When Adding Features
1. Use HarmonyCore types exclusively
2. Implement with FlatBuffers serialization
3. Design for multi-channel processing
4. Test across JVM, JS, and WASM targets

### When Refactoring
1. Delete provider-specific code immediately
2. Transform to token-native operations
3. Replace JSON with FlatBuffers
4. Maintain channel isolation

### Testing Multiplatform Code
```bash
./gradlew :module:compileKotlinJvm     # JVM compilation
./gradlew :module:compileKotlinJs      # JavaScript
./gradlew :module:compileKotlinWasmJs  # WebAssembly
```

## Forbidden Patterns

❌ **Never**:
- Create compatibility layers or adapters
- Maintain backwards compatibility with old message formats
- Add new provider clients or API integrations
- Use JSON for internal serialization
- Introduce type aliases for migration

✅ **Always**:
- Break things that need breaking
- Use compilation errors to find transformation points
- Prioritize binary efficiency over readability
- Design for parallel channel processing
- Keep the transformation aggressive and complete

## Harmony Response Format (GPT-OSS Models)

### Special Tokens
The GPT-OSS models use special tokens for message structure (o200k_harmony encoding):

| Token | Purpose | Token ID |
|-------|---------|----------|
| `<\|start\|>` | Message beginning with role header | 200006 |
| `<\|end\|>` | Message end | 200007 |
| `<\|message\|>` | Header to content transition | 200008 |
| `<\|channel\|>` | Channel information | 200005 |
| `<\|constrain\|>` | Tool call data type | 200003 |
| `<\|return\|>` | Completion stop token | 200002 |
| `<\|call\|>` | Tool call stop token | 200012 |

### Channel Architecture
Harmony channels separate reasoning from output:

1. **analysis** - Model's internal chain-of-thought (CoT)
   - NEVER shown to users (safety not guaranteed)
   - Contains raw reasoning process
   - Used for browser/python built-in tools

2. **commentary** - Tool interactions and preambles
   - Function tool calls must use this channel
   - May contain user-visible action plans
   - Used for multi-tool execution sequences

3. **final** - User-facing responses only
   - Safety-aligned output
   - The actual answer to user queries
   - Should be the only channel shown to users

### Role Hierarchy
Information priority when conflicts arise:
`system` > `developer` > `user` > `assistant` > `tool`

### Message Format
```
<|start|>{role}<|channel|>{channel}<|message|>{content}<|end|>
```

Tool calls include recipient:
```
<|start|>assistant<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{...}<|call|>
```

### Reasoning Configuration
Control reasoning effort in system message:
- `Reasoning: high` - Extensive CoT
- `Reasoning: medium` - Balanced (default)
- `Reasoning: low` - Minimal CoT

### Built-in Tools
GPT-OSS models have native support for:
- **browser** - Web search, open links, find patterns
- **python** - Code execution in Jupyter environment

These go in system message, not developer message.

### Implementation Notes

1. **CoT Handling**: Drop analysis channel content between turns unless tool calling is involved
2. **Tool Namespaces**: Use `functions` namespace to avoid conflicts with built-in tools
3. **TypeScript Syntax**: Define tools using TypeScript-like type definitions
4. **Streaming**: Use StreamableParser for incremental token processing

### Rust/Python Libraries
- Python: `openai-harmony` (PyPI)
- Rust: `openai_harmony` (crates.io)
- Both provide rendering and parsing with proper token handling

## Reference Documentation

For detailed transformation plans and architecture decisions, see:
- `KOOG_PROVIDER_REFACTOR.md` - Complete transformation strategy
- `prompt/prompt-model/src/commonMain/flatbuffers/` - FlatBuffers schemas
- `examples/.../harmony/HarmonyFirstExample.kt` - Usage patterns
- OpenAI Harmony docs at `../openai-harmony.md` for full format specification

---

**Remember**: This is not evolution, it's metamorphosis. The old Koog architecture must die completely for Noesis to emerge.
