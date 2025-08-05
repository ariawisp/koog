#!/bin/bash

echo "🤖 Koog Benchmark - Multi-Provider Test Script"
echo "============================================="
echo ""

# Function to check if API key is set
check_api_key() {
    local key_name=$1
    local key_value=${!key_name}
    if [ -z "$key_value" ]; then
        echo "❌ $key_name not set"
        return 1
    else
        echo "✅ $key_name is set"
        return 0
    fi
}

# Check available providers
echo "Checking available providers..."
echo ""

LMSTUDIO_AVAILABLE=false
OPENAI_AVAILABLE=false
ANTHROPIC_AVAILABLE=false

# Check LMStudio
if curl -s http://localhost:1234/v1/models > /dev/null 2>&1; then
    echo "✅ LMStudio is running at localhost:1234"
    LMSTUDIO_AVAILABLE=true
else
    echo "❌ LMStudio not detected at localhost:1234"
fi

# Check API keys
check_api_key "OPENAI_API_KEY" && OPENAI_AVAILABLE=true
check_api_key "ANTHROPIC_API_KEY" && ANTHROPIC_AVAILABLE=true

echo ""
echo "Available providers:"
./gradlew agents:agents-benchmark:runBenchmark --args="providers" -q

echo ""
echo "===================="
echo "📊 Example Commands:"
echo "===================="
echo ""

if [ "$LMSTUDIO_AVAILABLE" = true ]; then
    echo "1. Test with LMStudio (default):"
    echo "   ./gradlew agents:agents-benchmark:runBenchmark --args='run --size 10 --quick' -q"
    echo ""
fi

if [ "$OPENAI_AVAILABLE" = true ]; then
    echo "2. Test with OpenAI GPT-4:"
    echo "   ./gradlew agents:agents-benchmark:runBenchmark --args='run --provider openai --model gpt-4o --size 10' -q"
    echo ""
    echo "3. Test with OpenAI GPT-4-mini (cheaper):"
    echo "   ./gradlew agents:agents-benchmark:runBenchmark --args='run --provider openai --model gpt-4o-mini --size 10' -q"
    echo ""
fi

if [ "$ANTHROPIC_AVAILABLE" = true ]; then
    echo "4. Test with Anthropic Claude:"
    echo "   ./gradlew agents:agents-benchmark:runBenchmark --args='run --provider anthropic --model claude-3-sonnet-20240229 --size 10' -q"
    echo ""
fi

echo "================================"
echo "🏃 Quick Comparison Test"
echo "================================"
echo ""
echo "Would you like to run a quick comparison between available providers? (y/n)"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    DATASET_SIZE=5
    echo ""
    echo "Running comparison with $DATASET_SIZE questions..."
    echo ""
    
    # Test with LMStudio
    if [ "$LMSTUDIO_AVAILABLE" = true ]; then
        echo "📱 Testing LMStudio..."
        ./gradlew agents:agents-benchmark:runBenchmark --args="run --size $DATASET_SIZE --systems vector --output results-lmstudio.json" -q
        echo ""
    fi
    
    # Test with OpenAI
    if [ "$OPENAI_AVAILABLE" = true ]; then
        echo "🌐 Testing OpenAI GPT-4o-mini..."
        ./gradlew agents:agents-benchmark:runBenchmark --args="run --provider openai --model gpt-4o-mini --size $DATASET_SIZE --systems vector --output results-openai.json" -q
        echo ""
    fi
    
    # Test with Anthropic
    if [ "$ANTHROPIC_AVAILABLE" = true ]; then
        echo "🧠 Testing Anthropic Claude..."
        ./gradlew agents:agents-benchmark:runBenchmark --args="run --provider anthropic --model claude-3-haiku-20240307 --size $DATASET_SIZE --systems vector --output results-anthropic.json" -q
        echo ""
    fi
    
    echo "✅ Comparison complete! Check results-*.json files for details."
fi