package ai.koog.agents.benchmark.cli

import com.github.ajalt.clikt.core.subcommands
import com.github.ajalt.clikt.testing.test
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals

class BenchmarkCliTest {
    
    @Test
    fun `help command shows usage`() {
        val result = BenchmarkCli().test("--help")
        
        assertEquals(0, result.statusCode)
        assertContains(result.output, "Koog Memory System Benchmark Tool")
        assertContains(result.output, "Commands:")
    }
    
    @Test
    fun `list command shows available resources`() {
        val result = BenchmarkCli()
            .subcommands(ListCommand())
            .test("list")
        
        assertEquals(0, result.statusCode)
        assertContains(result.output, "Available Benchmark Resources")
        assertContains(result.output, "Datasets:")
        assertContains(result.output, "Retrieval Systems:")
    }
    
    @Test
    fun `run command help shows options`() {
        val result = BenchmarkCli()
            .subcommands(RunCommand())
            .test("run --help")
        
        assertEquals(0, result.statusCode)
        assertContains(result.output, "Run benchmark evaluation")
        assertContains(result.output, "--size")
        assertContains(result.output, "--output")
    }
    
    @Test
    fun `quick command help shows description`() {
        val result = BenchmarkCli()
            .subcommands(QuickCommand())
            .test("quick --help")
        
        assertEquals(0, result.statusCode)
        assertContains(result.output, "Run a quick benchmark test")
    }
    
    @Test
    fun `compare command validates systems`() {
        val result = BenchmarkCli()
            .subcommands(CompareCommand())
            .test("compare invalid-system")
        
        // Should fail because invalid-system is not valid
        assertEquals(1, result.statusCode)
        assertContains(result.stderr, "Invalid systems: invalid-system")
        assertContains(result.stderr, "Valid options:")
    }
}