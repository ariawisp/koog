#!/bin/bash

echo "🔍 Detailed Parallelism Test"
echo "============================"
echo ""
echo "This test will show exactly when each question is processed"
echo ""

# Create a simple monitoring script
cat > monitor_lmstudio.sh << 'EOF'
#!/bin/bash
echo "Monitoring LMStudio requests..."
while true; do
    TIMESTAMP=$(date +"%H:%M:%S")
    RESPONSE=$(curl -s http://localhost:1234/v1/models 2>/dev/null)
    if [ $? -eq 0 ]; then
        # Check if server is busy by trying to get response time
        START=$(date +%s%N)
        curl -s http://localhost:1234/v1/models > /dev/null 2>&1
        END=$(date +%s%N)
        LATENCY=$((($END - $START) / 1000000))
        echo "[$TIMESTAMP] LMStudio response time: ${LATENCY}ms"
    fi
    sleep 0.5
done
EOF

chmod +x monitor_lmstudio.sh

echo "Starting LMStudio monitor in background..."
./monitor_lmstudio.sh > lmstudio_monitor.log 2>&1 &
MONITOR_PID=$!

echo ""
echo "Running benchmark with 8 questions, 4 parallel..."
echo "Watch the pattern of processing to see if parallelism works"
echo ""

# Run benchmark with timing output
time ./gradlew agents:agents-benchmark:runBenchmark --args="run --size 8 --systems vector --parallel 4 --quick -v" -q 2>&1 | grep -E "(Running benchmark:|Progress:|Duration:|Processing|Evaluating|🔄)"

# Kill the monitor
kill $MONITOR_PID 2>/dev/null
rm monitor_lmstudio.sh

echo ""
echo "Check lmstudio_monitor.log to see request patterns"
echo "If parallelism works, you should see bursts of activity"