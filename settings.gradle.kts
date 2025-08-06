rootProject.name = "koog-agents"

pluginManagement {
    repositories {
        gradlePluginPortal()
    }
}

// Clone and include FlatBuffers Kotlin multiplatform library
val flatbuffersDir = file("build/flatbuffers-kmp")
val flatbuffersKotlinDir = file("build/flatbuffers-kmp/kotlin")

if (!flatbuffersDir.exists()) {
    println("Cloning FlatBuffers Kotlin multiplatform library...")
    val cloneResult = exec {
        commandLine("git", "clone", 
            "-b", "kotlin-kmp-build-improvements",
            "--depth", "1",
            "https://github.com/ariawisp/flatbuffers.git",
            flatbuffersDir.absolutePath)
    }
    
    if (cloneResult.exitValue == 0) {
        println("FlatBuffers cloned successfully")
        
        // Copy our flatc binary to where the FlatBuffers fork expects it
        val ourFlatc = file("prompt/prompt-model/build/tools/flatc/flatc")
        val targetFlatc = file("build/flatbuffers-kmp/flatc")
        
        if (ourFlatc.exists()) {
            ourFlatc.copyTo(targetFlatc, overwrite = true)
            targetFlatc.setExecutable(true)
            println("Copied flatc binary to FlatBuffers directory")
        } else {
            // If flatc doesn't exist yet, we'll download it first time build runs
            println("Note: flatc binary not found yet, will be downloaded on first build")
        }
    }
}

// Include FlatBuffers Kotlin as a composite build only if it exists
if (flatbuffersKotlinDir.exists()) {
    includeBuild(flatbuffersKotlinDir) {
        dependencySubstitution {
            substitute(module("com.google.flatbuffers:flatbuffers-kotlin"))
                .using(project(":flatbuffers-kotlin"))
        }
    }
}

include(":agents:agents-core")
include(":agents:agents-ext")

include(":agents:agents-features:agents-features-common")
include(":agents:agents-features:agents-features-event-handler")
include(":agents:agents-features:agents-features-memory")
include(":agents:agents-features:agents-features-opentelemetry")
include(":agents:agents-features:agents-features-trace")
include(":agents:agents-features:agents-features-tokenizer")
include(":agents:agents-features:agents-features-snapshot")

include(":agents:agents-mcp")
include(":agents:agents-test")
include(":agents:agents-tools")
include(":agents:agents-utils")

include(":examples")

include(":integration-tests")

include(":koog-agents")

include(":prompt:prompt-cache:prompt-cache-files")
include(":prompt:prompt-cache:prompt-cache-model")
include(":prompt:prompt-cache:prompt-cache-redis")

include(":prompt:prompt-executor:prompt-executor-cached")

include(":prompt:prompt-executor:prompt-executor-clients")
include(":prompt:prompt-executor:prompt-executor-clients:noesis-executor")

include(":prompt:prompt-executor:prompt-executor-llms")
include(":prompt:prompt-executor:prompt-executor-llms-all")
include(":prompt:prompt-executor:prompt-executor-model")

include(":prompt:prompt-llm")
include(":prompt:prompt-markdown")
include(":prompt:prompt-model")
include(":prompt:prompt-structure")
include(":prompt:prompt-tokenizer")
include(":prompt:prompt-xml")

include(":embeddings:embeddings-base")
include(":embeddings:embeddings-llm")

include(":rag:rag-base")
include(":rag:vector-storage")

include(":koog-spring-boot-starter")

include(":koog-ktor")
