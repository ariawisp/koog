# Noesis Runtime Quick Start Guide

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3) or Intel Mac
- Xcode Command Line Tools (`xcode-select --install`)
- Rust toolchain (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- CMake (`brew install cmake`)
- Java 17+ 

## Setup (One-Time)

### 1. Configure Your Model Path

```bash
# Copy the template configuration
cp noesis.properties.template noesis.properties

# Edit noesis.properties and set your model path
# Example: gptoss.model.path=/Users/yourname/models/gpt-oss-20b/model.bin
```

### 2. Download a GPT-OSS Model

Download from [OpenAI GPT-OSS Models](https://huggingface.co/collections/openai/gpt-oss-models)

Recommended models:
- `gpt-oss-20b` - Best quality/speed balance (13.7 GB)
- `gpt-oss-7b` - Faster, lower quality (4.5 GB)
- `gpt-oss-120b` - Highest quality, slower (80 GB)

### 3. Build Everything

```bash
# This will automatically:
# - Clone GPT-OSS from GitHub
# - Build GPT-OSS Metal libraries
# - Build Noesis Runtime (Rust)
# - Copy Metal shaders to correct locations
./gradlew buildNoesisRuntime
```

## Running

```bash
# Run the Noesis Runtime with your configured model
./gradlew runNoesis
```

## Testing

```bash
# Run tests with real inference
./gradlew testNoesis
```

## Troubleshooting

### Check Your Setup
```bash
# Run the setup wizard
./gradlew -b setup-noesis.gradle.kts setupNoesis

# Check configuration and build status
./gradlew -b setup-noesis.gradle.kts checkNoesis
```

### Common Issues

#### "Model file not found"
- Ensure your model path in `noesis.properties` is absolute
- Check the file exists: `ls -la /path/to/your/model.bin`

#### "Unsupported system" error
- This usually means the Metal shaders weren't found
- Run: `./gradlew clean buildNoesisRuntime`
- The build process will copy `default.metallib` to the correct locations

#### Build fails with "libgptoss.a not found"
- Clean and rebuild: `./gradlew cleanGptOss cloneGptOss buildGptOss`

#### Rust compilation errors
- Update Rust: `rustup update`
- Clean Rust build: `cd noesis-runtime && cargo clean`
- Rebuild: `./gradlew buildNoesisRuntime`

## Configuration Options

Edit `noesis.properties` to customize:

```properties
# Required: Path to your model
gptoss.model.path=/path/to/model.bin

# Optional: Max tokens to generate (default: 100)
gptoss.test.max_tokens=200

# Optional: Temperature for generation (0.0-1.0, default: 0.7)
gptoss.test.temperature=0.8
```

## Performance

Expected performance on Apple Silicon:
- M1: 80-100 tokens/second
- M2: 100-120 tokens/second  
- M2 Ultra: 150-200 tokens/second
- M3 Max: 200+ tokens/second

## Architecture

```
Your Code (Kotlin/Java)
    ↓ JNI
Noesis Runtime (Rust)
    ↓ FFI
GPT-OSS C API
    ↓ Metal
Apple GPU
```

## Development

### Clean Everything
```bash
./gradlew clean cleanGptOss
cd noesis-runtime && cargo clean
```

### Rebuild After Changes
```bash
# After Rust changes
./gradlew buildNoesisRuntime

# After Kotlin changes
./gradlew build

# Full rebuild
./gradlew clean buildNoesisRuntime build
```

### Working with Multiple Models

You can switch models by editing `noesis.properties` or creating multiple property files:

```bash
# Create profiles for different models
cp noesis.properties noesis-7b.properties
cp noesis.properties noesis-20b.properties
cp noesis.properties noesis-120b.properties

# Switch by copying
cp noesis-20b.properties noesis.properties
./gradlew runNoesis
```

## Git Workflow

The following files are git-ignored and won't affect your commits:
- `noesis.properties` - Your local configuration
- `local.properties` - Android/Gradle local settings  
- `gradle.local.properties` - Additional local Gradle settings
- `build/` directories - All build artifacts
- `*.metallib` - Compiled Metal shaders
- GPT-OSS source (auto-cloned)

Only `noesis.properties.template` is tracked in git as a reference.