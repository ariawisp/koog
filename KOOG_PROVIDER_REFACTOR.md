# Noesis Runtime - Transforming Koog into a Harmony-Native Reasoning Engine

## 🧠 Strategic Pivot: Transform Koog INTO Noesis Runtime

**Decision Date**: 2025-08-06

We are transforming Koog directly into the **Noesis Runtime** - a Harmony-native reasoning engine designed exclusively for OpenAI's GPT-OSS models. This is not a fork or parallel development, but a complete metamorphosis of the existing codebase.

## 🎯 Core Philosophy

> "This isn't about prompts. It's about thought."

The Noesis Runtime treats Harmony not as a message format, but as the **substrate for cognition**. We're building an agentic operating system where:

- **Models reason in Harmony**, not just generate text
- **Memory is structural**, not chat history  
- **Tools are executed as thoughts**, not bolted-on APIs
- **Checkpoints are binary tokens**, not lossy text

## 🏗️ Architecture Vision

```
┌─────────────────────────────────────────────────────────────┐
│                    Noesis Runtime                           │
│  "Cognition as a Service"                                   │
├─────────────────────────────────────────────────────────────┤
│  • Harmony-native from ground up                            │
│  • Binary token checkpoints                                 │
│  • Multi-channel reasoning streams                          │
│  • Native Metal GPU inference                               │
│  • Zero provider abstractions                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 Harmony IR (The Substrate)                  │
│  • Not a format - a reasoning language                      │
│  • Channels: analysis, commentary, final                    │
│  • Native tool calling via <|call|> tokens                  │
│  • Role hierarchy with semantic boundaries                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              GPT-OSS 20B/120B (Metal Backend)               │
│  • Local inference on Apple Silicon                         │
│  • 100+ tokens/sec on M2 Ultra                              │
│  • No API dependencies, no rate limits                      │
│  • Full ownership of the inference stack                    │
└─────────────────────────────────────────────────────────────┘
```

## 🔥 What We're Building

### 1. Harmony-Native Agent System
```kotlin
// Transform existing AIAgent.kt to be Harmony-native
class AIAgent {  // KEEP THE NAME - no unnecessary renames
    private val reasoning: HarmonyReasoningEngine
    private val memory: TokenStreamGraph
    private val tools: StreamIntegratedTools
    
    suspend fun think(context: HarmonyContext): ThoughtStream {
        // Direct token manipulation, no text conversion
        return reasoning.process(context.tokens)
            .routeChannels()
            .executeTools()
            .updateMemory()
    }
}
```

### 2. Binary Token Checkpoints
```kotlin
@Serializable
data class TokenCheckpoint(
    val modelId: String = "gpt-oss-20b",
    val tokens: ByteArray,           // Raw token IDs, not text
    val channels: Map<Channel, IntRange>,
    val timestamp: Long,
    val parentCheckpoint: String?,   // For branching/forking
    val memoryGraph: TokenMemoryGraph
)

// Enables:
// - Perfect state reconstruction
// - Instant context forking for parallel exploration
// - Token-level debugging and replay
// - Efficient storage (10x smaller than text)
```

### 3. Stream-Integrated Tools
```kotlin
// Tools operate directly on token streams, not as callbacks
class StreamTool {
    suspend fun execute(
        tokenStream: Flow<Token>,
        insertionPoint: Channel
    ): Flow<Token> {
        return tokenStream.transform { token ->
            when {
                token.isToolCall() -> {
                    val result = executeToolLogic(token)
                    emit(token)
                    emit(result.asTokens())
                }
                else -> emit(token)
            }
        }
    }
}
```

### 4. Multi-Channel Reasoning
```kotlin
sealed class ReasoningChannel {
    // NEVER exposed to users - model's internal thoughts
    object Analysis : ReasoningChannel()
    
    // Tool interactions and structured outputs
    object Commentary : ReasoningChannel()
    
    // User-facing responses only
    object Final : ReasoningChannel()
}

// True parallel processing of channels
fun processChannels(tokens: Flow<Token>): ChannelStreams {
    return tokens.split { token ->
        when (token.channel) {
            ANALYSIS -> processAnalysis(token)    // Route to reasoning engine
            COMMENTARY -> processTools(token)     // Route to tool executor
            FINAL -> processResponse(token)       // Route to user interface
        }
    }
}
```

## 🚀 Repository Structure (Transform In-Place)

```
koog/ (becomes Noesis Runtime - NO NEW FILES, TRANSFORM EXISTING)
├── agents/
│   ├── agents-core/              
│   │   ├── AIAgent.kt            # TRANSFORM: Make token-native (keep name)
│   │   ├── AIAgentStrategy.kt    # TRANSFORM: Simplify to reasoning graphs
│   │   └── AIAgentContext.kt     # TRANSFORM: Use HarmonyCore instead of Messages
│   │
│   ├── agents-tools/             
│   │   ├── Tool.kt               # TRANSFORM: Stream-integrated (keep name)
│   │   └── ToolRegistry.kt       # TRANSFORM: Token-based tool calls
│   │
│   ├── agents-features/          # TRANSFORM: Update to use HarmonyCore
│   │   ├── agents-features-memory/      # Convert to TokenMemoryGraph
│   │   ├── agents-features-snapshot/    # Becomes binary TokenCheckpoint
│   │   ├── agents-features-tokenizer/   # Move to Rust, thin Kotlin wrapper
│   │   ├── agents-features-opentelemetry/ # Adapt to Harmony events
│   │   └── agents-features-trace/       # Channel-aware logging
│   │
│   └── agents-test/              # KEEP: Testing framework
│
├── prompt/
│   ├── prompt-model/             
│   │   ├── harmony/              # KEEP & EXPAND: Already excellent
│   │   │   ├── HarmonyCore.kt    # Already THE message type
│   │   │   ├── ChanneledMessage.kt
│   │   │   └── HarmonyJNI.kt     
│   │   ├── native/               # EXPAND: Move tokenization here
│   │   │   ├── harmony-jni/      # Existing Rust bridge
│   │   │   └── harmony-tokenizer/ # NEW: All tokenization in Rust
│   │   └── [DELETE: Message.kt, UnifiedModel.kt, LLMParams.kt]
│   │
│   └── prompt-executor/          
│       ├── prompt-executor-clients/
│       │   ├── prompt-executor-harmony-client/  # RENAME to noesis-executor
│       │   │   ├── HarmonyNativeLLMClient.kt   # KEEP: Core executor
│       │   │   ├── MetalInferenceJNI.kt        # KEEP: GPU inference
│       │   │   └── native/                     # KEEP: Rust/Metal code
│       │   │
│       │   └── [DELETE: ALL other client directories]
│       │
│       └── prompt-executor-model/
│           └── PromptExecutor.kt  # TRANSFORM: Harmony-only interface
│
└── [DELETE: All provider-specific code, downsamplers, compatibility layers]
```

**CRITICAL**: No new files. No renames like AIAgent → NoesisAgent. Transform existing code in-place.

## 📊 Why This Architecture Wins

### Performance Comparison

| Metric | Current Koog | Noesis Runtime | Improvement |
|--------|-------------|----------------|-------------|
| **Inference Latency** | 500-2000ms (API) | 50-150ms (Metal) | **10-40x faster** |
| **Token Throughput** | 20-50 tok/s | 100-200 tok/s | **4-10x faster** |
| **Memory Footprint** | Text-based | Binary tokens | **10x smaller** |
| **Context Recovery** | Re-tokenize | Direct load | **Instant** |
| **Tool Execution** | Callback-based | Stream-integrated | **Zero overhead** |
| **Reasoning Transparency** | None | Full (channels) | **Complete** |

### Unique Capabilities

| Feature | Why It Matters |
|---------|---------------|
| **Token-level checkpoints** | Perfect reproducibility, debugging, and branching |
| **Native Harmony** | No lossy conversions, full semantic preservation |
| **Channel isolation** | Safety boundaries, parallel processing |
| **Metal inference** | Local, fast, unlimited usage |
| **Stream tools** | Real-time tool execution without interruption |

## 🛠️ Git Strategy & Transformation Plan

### Recommended Git Strategy: **In-Place Transformation**

```bash
# Option 1: Transform Current Feature Branch (RECOMMENDED)
# Continue on feature/enhance-llm-params, complete the transformation
git checkout feature/enhance-llm-params
git commit -am "PIVOT: Begin Noesis Runtime transformation"

# Option 2: New Transformation Branch
# Start fresh but cherry-pick valuable work
git checkout develop
git checkout -b noesis-transformation
git cherry-pick 78a2372a  # Harmony-first architecture

# Option 3: Squash and Rebase (CLEANEST HISTORY)
# Complete transformation, then squash into one commit
git checkout feature/enhance-llm-params
# ... do all transformation work ...
git rebase -i develop  # Squash to single "Transform Koog to Noesis Runtime" commit
```

### Why Option 1 (Transform Current Branch) is Best:
1. **Preserves valuable work** already done on Harmony/Metal
2. **Shows evolution** from extension to transformation
3. **Easier to track** what changed and why
4. **Can always squash later** for clean history

### Phase 1: Mass Deletion (Day 1)
```bash
# Delete all provider infrastructure
rm -rf prompt/prompt-executor/prompt-executor-clients/prompt-executor-openai-client/
rm -rf prompt/prompt-executor/prompt-executor-clients/prompt-executor-anthropic-client/
rm -rf prompt/prompt-executor/prompt-executor-clients/prompt-executor-google-client/
rm -rf prompt/prompt-executor/prompt-executor-clients/prompt-executor-ollama-client/

# Delete all downsamplers
find . -name "*Downsampler*.kt" -delete

# Delete legacy message types
rm prompt/prompt-model/src/commonMain/kotlin/ai/koog/prompt/model/Message.kt
rm prompt/prompt-model/src/commonMain/kotlin/ai/koog/prompt/model/UnifiedModel.kt

git add -A
git commit -m "refactor: Remove all provider abstractions and legacy message types"
```

### Phase 2: Rename & Restructure (Day 2)
```bash
# Rename harmony client to be THE executor
mv prompt/prompt-executor/prompt-executor-clients/prompt-executor-harmony-client \
   prompt/prompt-executor/noesis-executor

# Update all imports and references
find . -name "*.kt" -exec sed -i '' 's/harmony.client/noesis.executor/g' {} \;

# Update build.gradle.kts files
find . -name "build.gradle.kts" -exec sed -i '' 's/harmony-client/noesis-executor/g' {} \;

git add -A
git commit -m "refactor: Rename harmony-client to noesis-executor"
```

### Phase 3: Transform Core Classes (Week 1)
- Transform `AIAgent` → `NoesisAgent` (token-native)
- Transform `Tool` → `StreamTool` (stream-integrated)
- Transform `Prompt` → `TokenContext` (binary tokens)
- Delete all `LLMParams`, provider configs

### Phase 4: Implement Token Systems (Week 2)
- Implement `TokenCheckpoint` with binary serialization
- Build `TokenMemoryGraph` for semantic memory
- Create `ChannelRouter` for multi-stream processing
- Integrate Metal inference directly

## 🔬 Technical Decisions

### Why Transform Instead of Fork?
1. **Philosophical**: Koog BECOMES Noesis, not a parallel project
2. **Technical**: In-place transformation avoids duplication
3. **Strategic**: Clean diffs show exactly what changed
4. **Clarity**: One codebase, one vision

### CRITICAL: No Compatibility Layers
- **NEVER** introduce adapters between old and new code (no typealias!)
- **NEVER** maintain backwards compatibility
- **NEVER** create migration shims
- **Let things break** - compilation errors show what needs transformation
- **This is a complete reimagining** - break everything that needs breaking

### Why Binary Tokens?
1. **Lossless**: Perfect state reconstruction
2. **Fast**: No tokenization overhead on resume
3. **Compact**: 10x smaller than text
4. **Precise**: Token-level debugging and analysis

### Why Metal Only?
1. **Performance**: 100+ tok/s on consumer hardware
2. **Control**: Full ownership of inference stack
3. **Cost**: Zero API costs, unlimited usage
4. **Privacy**: Completely local, no data leaves device

## 🎮 Mitra Companion Vision

With Noesis Runtime, each Minecraft server runs its own GPT-OSS 20B instance:

```kotlin
class MitraCompanion(
    private val runtime: NoesisRuntime,
    private val world: MinecraftWorld
) {
    suspend fun perceive() {
        val worldState = world.encodeState()
        val thought = runtime.think(worldState)
        
        // Parallel channel processing
        launch { processAnalysis(thought.analysis) }     // Internal reasoning
        launch { executeTools(thought.commentary) }       // Game actions
        launch { respondToPlayer(thought.final) }         // Chat responses
    }
}
```

**Result**: True AI companions with:
- Sub-second response times
- Perfect memory across sessions
- Complex multi-step planning
- Natural conversation flow
- Complete privacy

## 📈 Success Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| **Inference Speed** | >100 tok/s | Metal GPU acceleration |
| **Memory Coherence** | 95%+ | Graph-based, not chunks |
| **Tool Accuracy** | 95%+ | Native token-based calls |
| **Checkpoint Size** | <100KB | Binary, not text |
| **Recovery Time** | <100ms | Direct token loading |
| **Channel Isolation** | 100% | Safety guaranteed |

## 🚨 Breaking Changes from Koog

### Deleted Completely
- ❌ ALL provider clients (OpenAI, Anthropic, Google, etc.)
- ❌ UnifiedModel and all legacy message types
- ❌ LLMParams and provider-specific configurations
- ❌ Text-based prompt building
- ❌ JSON-based tool calling
- ❌ Compatibility layers

### Transformed In-Place
- ✅ AIAgent → Harmony-native (same class name)
- ✅ Messages → HarmonyCore (already exists)
- ✅ Tool → Stream-integrated (same class name)
- ✅ Memory → TokenGraphs (new capability)
- ✅ Checkpoints → Binary tokens (new capability)

## 🔮 Long-Term Vision

The Noesis Runtime enables:

1. **Distributed Cognition**: Agents that think across multiple instances
2. **Temporal Reasoning**: Time-travel through checkpoint history
3. **Parallel Exploration**: Fork contexts for what-if scenarios
4. **Semantic Memory**: Graph-based understanding, not retrieval
5. **Tool Synthesis**: Agents that create their own tools

## 📦 Feature Modules Transformation

### Impact Assessment (46 files use Message/LLMParams)

| Feature Module | Files to Change | Priority | Transformation |
|----------------|-----------------|----------|---------------|
| **memory** | ~15 files | HIGH | Convert to TokenMemoryGraph |
| **opentelemetry** | ~20 files | HIGH | Adapt events to HarmonyCore |
| **snapshot** | ~8 files | HIGH | Implement binary checkpoints |
| **trace** | ~10 files | MEDIUM | Add channel-aware logging |
| **event-handler** | ~5 files | MEDIUM | Process token streams |
| **tokenizer** | ~3 files | LOW | Move to Rust, thin wrapper |
| **common** | ~2 files | LOW | Update utilities |

### Feature-Specific Changes

#### agents-features-memory
- Replace `Message` history with token streams
- Convert `AgentMemory.kt` to work with `HarmonyCore`
- Transform history compression to token-based
- Align with `TokenMemoryGraph` vision

#### agents-features-opentelemetry
- Convert `UserMessageEvent`, `SystemMessageEvent`, etc. to use `HarmonyCore`
- Remove provider-specific telemetry
- Add channel-aware event tracking
- Focus on Metal/GPT-OSS metrics only

#### agents-features-snapshot
- Replace text-based checkpoints with binary tokens
- Natural evolution to `TokenCheckpoint` system
- Add checkpoint branching/forking
- Enable instant state recovery

#### agents-features-tokenizer
- Move core logic to Rust (`native/harmony-tokenizer/`)
- Keep thin Kotlin wrapper for integration
- Share tokenizer with Metal inference
- Direct binary token operations

## 🦀 Tokenizer Migration to Rust

### Current → Future Architecture

```rust
// native/harmony-tokenizer/src/lib.rs
pub struct HarmonyTokenizer {
    vocab: HashMap<String, TokenId>,
    merges: Vec<(TokenId, TokenId)>,
}

impl HarmonyTokenizer {
    // Core tokenization
    pub fn encode(&self, text: &str) -> Vec<TokenId>
    pub fn decode(&self, tokens: &[TokenId]) -> String
    
    // Harmony-specific
    pub fn encode_harmony(&self, harmony: &HarmonyCore) -> TokenStream
    pub fn extract_channels(&self, stream: &TokenStream) -> ChannelMap
    
    // Binary operations
    pub fn serialize_tokens(&self, tokens: &[TokenId]) -> Vec<u8>
    pub fn deserialize_tokens(&self, data: &[u8]) -> Vec<TokenId>
}
```

### Kotlin Wrapper (Minimal)

```kotlin
// agents-features-tokenizer/MessageTokenizer.kt becomes:
class HarmonyTokenizer {
    @JvmStatic external fun encodeToTokens(text: String): IntArray
    @JvmStatic external fun decodeTokens(tokens: IntArray): String
    @JvmStatic external fun countTokens(harmony: HarmonyCore): Int
    
    // High-level utilities only
    fun estimateTokens(harmony: HarmonyCore): Int = countTokens(harmony)
}
```

### Benefits of Rust Tokenization
1. **Shared with inference**: Same tokenizer for Metal GPU
2. **Zero-copy operations**: Direct binary manipulation
3. **10x faster**: Native performance vs JVM
4. **Consistent vocabulary**: Single source of truth
5. **Channel-aware**: Native understanding of Harmony structure

## 📊 Branch Analysis: What to Keep vs Reset

### Current Branch Status
- **Branch**: `feature/enhance-llm-params` 
- **Changes**: 100 files, +10,557 lines, -3,559 lines
- **Commit**: `78a2372a feat: Complete Harmony-first architecture with Metal GPU inference support`

### ✅ KEEP - Core Harmony Infrastructure

#### 1. **HarmonyCore Types** (748 lines)
- `prompt/prompt-model/src/commonMain/kotlin/ai/koog/prompt/harmony/HarmonyCore.kt`
- **Value**: Complete Harmony IR implementation with channels, roles, tools
- **Action**: Keep in place, expand as THE message type

#### 2. **Metal Inference Infrastructure** (~1,500 lines)
- `prompt-executor-harmony-client/native/metal-inference-jni/` (Rust)
- `MetalInferenceJNI.kt`, `HarmonyNativeLLMClient.kt`
- **Value**: Working Metal GPU integration, 100+ tok/s verified
- **Action**: Extract and simplify - remove JNI complexity if possible

#### 3. **Harmony JNI Bridge** (435 lines)
- `prompt/prompt-model/native/harmony-jni/src/lib.rs`
- `HarmonyEncoding.kt`, `HarmonyJNI.kt`
- **Value**: Token rendering/parsing infrastructure
- **Action**: Keep but consolidate with Metal inference

#### 4. **HarmonyFirstExample** (496 lines)
- `examples/.../harmony/HarmonyFirstExample.kt`
- **Value**: Shows complete Harmony usage patterns
- **Action**: Use as template for Noesis examples

### ❌ DELETE - Provider Infrastructure (Don't Transfer)

#### 1. **All Downsamplers** (~2,000 lines)
- `Harmony*Downsampler.kt` files (9 total)
- `HarmonyDownsamplerBase.kt`
- **Reason**: Noesis is Harmony-native, no downsampling needed
- **Action**: DELETE ENTIRELY

#### 2. **Provider Clients** (~3,000 lines)
- All `*LLMClient.kt` files
- All provider-specific request/response models
- **Reason**: Only GPT-OSS/Metal inference
- **Action**: DELETE ENTIRELY

#### 3. **Provider-Specific Serializers** (~1,000 lines)
- `*Serializers.kt`, `*JsonConfig.kt`
- **Reason**: No JSON APIs in Noesis
- **Action**: DELETE ENTIRELY

### 🔄 TRANSFORM - Core Abstractions (In-Place)

#### 1. **AIAgent.kt**
- **Current**: Message-based with complex strategies
- **Transform**: Token-native with HarmonyCore
- **Action**: Modify existing file, don't rename class

#### 2. **Tool.kt** 
- **Current**: Callback-based with JSON schemas
- **Transform**: Stream-integrated with <|call|> tokens
- **Action**: Modify existing file, don't rename class

#### 3. **PromptExecutor.kt**
- **Current**: Multi-provider abstraction
- **Transform**: Harmony-only executor
- **Action**: Strip to single implementation

### 📈 In-Place Transformation Path

#### Current State → Noesis Runtime
```
feature/enhance-llm-params (current branch)
    ↓
[Mass Deletion Commit]
    ↓
[Restructure Commit]
    ↓
[Transform Core Commit]
    ↓
[Token Systems Commit]
    ↓
noesis-runtime-v1.0 (tag)
```

#### Transformation Commits Sequence:

##### Commit 1: "refactor: Remove all provider infrastructure"
```bash
# What gets deleted:
- All provider clients (OpenAI, Anthropic, Google, Ollama, etc.)
- All downsamplers
- Legacy message types
- Provider-specific configurations
- JSON serializers for APIs
```

##### Commit 2: "refactor: Establish Harmony as sole message type"
```bash
# What changes:
- HarmonyCore becomes THE message type
- prompt-executor-harmony-client → noesis-executor
- Remove all Message/UnifiedModel references
- Update all imports
```

##### Commit 3: "feat: Transform agents to token-native"
```bash
# What changes IN-PLACE:
- AIAgent.kt transformed to use HarmonyCore
- Tool.kt transformed to stream-integrated
- AIAgentContext.kt uses tokens instead of messages
- Add TokenMemoryGraph capability
```

##### Commit 4: "feat: Binary token checkpoints"
```bash
# What's added:
- TokenCheckpoint serialization
- Checkpoint branching/forking
- Token-level state management
- Fast checkpoint recovery
```

##### Commit 5: "feat: Complete Metal inference integration"
```bash
# What's finalized:
- Direct GPT-OSS model loading
- Streaming token generation
- Channel-aware inference
- GPU memory management
```

### 📊 Code Distribution Analysis

| Component | Lines | Keep? | Reason |
|-----------|-------|-------|---------|
| **HarmonyCore + Types** | ~1,200 | ✅ YES | Foundation of Noesis |
| **Metal Inference** | ~1,500 | ✅ YES | Core inference engine |
| **Harmony JNI** | ~750 | ✅ YES | Token processing |
| **Downsamplers** | ~2,000 | ❌ NO | No providers in Noesis |
| **Provider Clients** | ~3,000 | ❌ NO | Only Metal inference |
| **Serializers** | ~1,000 | ❌ NO | No JSON APIs |
| **Agent Core** | ~500 | 🔄 PARTIAL | Extract graph concepts |
| **Examples** | ~500 | ✅ YES | Learning material |

### 🎯 Key Decisions

1. **Transform in Place**: Convert Koog directly into Noesis Runtime, not parallel development
2. **Current Branch**: Continue on `feature/enhance-llm-params` to preserve Harmony/Metal work
3. **Binary First**: Design around tokens from day 1, not messages
4. **No Compatibility**: Delete ALL provider abstractions immediately
5. **Metal Only**: Commit fully to local inference, remove all API clients
6. **Channel Native**: Make multi-channel reasoning the default
7. **Aggressive Deletion**: Remove legacy code immediately, not gradually
8. **Tokenizer to Rust**: Move all tokenization to native code for performance

## 🏁 Next Steps

### Immediate Actions (Today)
1. Commit current work: `git commit -am "docs: Define Noesis Runtime transformation strategy"`
2. Begin mass deletion of provider infrastructure
3. Remove all downsamplers and compatibility layers
4. Establish Harmony as the sole message type

### Day 1-2: Demolition Phase
```bash
# Delete ALL provider clients (6 total)
rm -rf prompt/prompt-executor/prompt-executor-clients/prompt-executor-openai-client/
rm -rf prompt/prompt-executor/prompt-executor-clients/prompt-executor-anthropic-client/
rm -rf prompt/prompt-executor/prompt-executor-clients/prompt-executor-google-client/
rm -rf prompt/prompt-executor/prompt-executor-clients/prompt-executor-ollama-client/
rm -rf prompt/prompt-executor/prompt-executor-clients/prompt-executor-bedrock-client/
rm -rf prompt/prompt-executor/prompt-executor-clients/prompt-executor-openrouter-client/

# Delete ALL downsamplers (17 files)
find . -name "*Downsampler*.kt" -delete

# Delete legacy message types
rm prompt/prompt-model/src/commonMain/kotlin/ai/koog/prompt/model/Message.kt
rm prompt/prompt-model/src/commonMain/kotlin/ai/koog/prompt/model/UnifiedModel.kt
rm prompt/prompt-model/src/commonMain/kotlin/ai/koog/prompt/dsl/LLMParams.kt

git add -A
git commit -m "refactor: Remove all provider infrastructure (~17,000 lines)"
```

### Week 1: Core Transformation
- [ ] Transform AIAgent.kt to be Harmony-native (keep class name)
- [ ] Transform Tool.kt to be stream-integrated (keep class name)  
- [ ] Replace Message/UnifiedModel with HarmonyCore everywhere
- [ ] Rename harmony-client directory → noesis-executor
- [ ] Features will break - this is intentional to see what needs fixing

### Week 2: Token Systems & Features
- [ ] Implement TokenCheckpoint with binary serialization
- [ ] Transform agents-features-memory to TokenMemoryGraph
- [ ] Transform agents-features-snapshot to binary checkpoints
- [ ] Move tokenizer logic to Rust (native/harmony-tokenizer/)
- [ ] Create ChannelRouter for multi-stream processing

### Week 3: Feature Transformation
- [ ] Update agents-features-opentelemetry for Harmony events
- [ ] Transform agents-features-trace for channel-aware logging
- [ ] Update agents-features-event-handler for token streams
- [ ] Fix all compilation errors from Message → HarmonyCore change
- [ ] Integrate Metal inference directly (optimize JNI)

### Week 4: Mitra Integration
- [ ] Minecraft world state encoder
- [ ] Token-based action executor
- [ ] Companion behavior system
- [ ] Server deployment

---

**Status**: 🔥 READY TO TRANSFORM  
**Strategy**: In-place transformation on `feature/enhance-llm-params`  
**Impact**: Delete ~7,000 lines (providers), Transform ~3,500 lines (core), Keep ~1,500 lines (Harmony/Metal)  
**Philosophy**: "Koog dies, Noesis rises"  

*This is not evolution, it's metamorphosis. The caterpillar becomes the butterfly.*