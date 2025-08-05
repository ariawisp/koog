package ai.koog.agents.benchmark

import ai.koog.agents.benchmark.cli.main as benchmarkMain

/**
 * Main entry point for running benchmarks
 * 
 * This delegates to the Clikt-based CLI for better command parsing and help generation.
 * 
 * Usage:
 *   koog-benchmark run --size 50 --quick
 *   koog-benchmark quick
 *   koog-benchmark compare node-distance,rrf
 *   koog-benchmark list
 */
fun main(args: Array<String>) {
    // Delegate to the CLI main function
    benchmarkMain(args)
}