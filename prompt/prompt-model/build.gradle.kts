import ai.koog.gradle.publish.maven.Publishing.publishToMaven
import org.gradle.internal.os.OperatingSystem

group = rootProject.group
version = rootProject.version

plugins {
    id("ai.kotlin.multiplatform") // This already configures jvm, js, and wasmJs targets
    alias(libs.plugins.kotlin.serialization)
}

// FlatBuffers configuration
val flatbuffersDir = file("src/commonMain/flatbuffers")
val generatedFlatbuffersDir = layout.buildDirectory.dir("generated/flatbuffers")

// Task to download flatc compiler if not present
val downloadFlatc = tasks.register<Exec>("downloadFlatc") {
    val flatcDir = layout.buildDirectory.dir("tools/flatc")
    val flatcBinary = flatcDir.map { 
        it.file(if (OperatingSystem.current().isWindows) "flatc.exe" else "flatc")
    }
    
    onlyIf {
        !flatcBinary.get().asFile.exists()
    }
    
    doFirst {
        flatcDir.get().asFile.mkdirs()
        logger.lifecycle("Downloading FlatBuffers compiler...")
    }
    
    // Determine OS-specific download URL
    val flatcUrl = when {
        OperatingSystem.current().isMacOsX -> "https://github.com/google/flatbuffers/releases/download/v24.3.25/Mac.flatc.binary.zip"
        OperatingSystem.current().isLinux -> "https://github.com/google/flatbuffers/releases/download/v24.3.25/Linux.flatc.binary.clang++-15.zip"
        OperatingSystem.current().isWindows -> "https://github.com/google/flatbuffers/releases/download/v24.3.25/Windows.flatc.binary.zip"
        else -> throw GradleException("Unsupported OS for FlatBuffers")
    }
    
    commandLine("bash", "-c", """
        cd "${flatcDir.get().asFile}" &&
        curl -L -o flatc.zip "$flatcUrl" &&
        unzip -o flatc.zip &&
        chmod +x flatc &&
        rm flatc.zip
    """.trimIndent())
    
    doLast {
        logger.lifecycle("FlatBuffers compiler downloaded successfully")
    }
}

// Task to compile FlatBuffers schemas
val compileFlatBuffers = tasks.register<Exec>("compileFlatBuffers") {
    dependsOn(downloadFlatc)
    
    val flatcBinary = layout.buildDirectory.file(
        "tools/flatc/${if (OperatingSystem.current().isWindows) "flatc.exe" else "flatc"}"
    )
    
    val outputDir = generatedFlatbuffersDir.get().asFile
    
    inputs.dir(flatbuffersDir)
    outputs.dir(outputDir)
    
    doFirst {
        outputDir.mkdirs()
        logger.lifecycle("Compiling FlatBuffers schemas...")
    }
    
    // Compile all .fbs files with Kotlin multiplatform support
    commandLine("bash", "-c", """
        for fbs in ${flatbuffersDir}/*.fbs; do
            if [ -f "${'$'}fbs" ]; then
                echo "Compiling ${'$'}fbs for Kotlin Multiplatform..."
                ${flatcBinary.get().asFile} --kotlin-kmp -o ${outputDir} "${'$'}fbs"
            fi
        done
    """.trimIndent())
    
    doLast {
        logger.lifecycle("FlatBuffers schemas compiled successfully")
    }
}

kotlin {
    // The ai.kotlin.multiplatform plugin already configures jvm, js, and wasmJs targets
    
    sourceSets {
        commonMain {
            // Generated FlatBuffers KMP code goes in commonMain
            // TODO: Fix FlatBuffers enum generation before re-enabling
            // kotlin.srcDir(generatedFlatbuffersDir)
            dependencies {
                api(project(":prompt:prompt-llm"))
                api(project(":agents:agents-tools"))
                api(libs.kotlinx.serialization.json)
                api(libs.kotlinx.datetime)
                api(libs.kotlinx.io.core)
                api(libs.kotlinx.coroutines.core)
                // FlatBuffers Kotlin multiplatform runtime
                // TODO: Re-enable when FlatBuffers generation is fixed
                // implementation("com.google.flatbuffers:flatbuffers-kotlin:25.2.10")
            }
        }
        
        jvmMain {
            dependencies {
                // JVM-specific dependencies if needed
            }
        }
        
        jsMain {
            dependencies {
                // JS-specific dependencies if needed
            }
        }

        commonTest {
            dependencies {
                implementation(kotlin("test"))
            }
        }

        jsTest {
            dependencies {
                implementation(kotlin("test-js"))
            }
        }

        jvmTest {
            dependencies {
                implementation(kotlin("test-junit5"))
                implementation(libs.junit.jupiter.params)
            }
        }
    }

    // Note: Disabling explicitApi because FlatBuffers generates code without explicit visibility modifiers
    // explicitApi()
}

// Make Kotlin compilation depend on FlatBuffers generation
tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile>().configureEach {
    dependsOn(compileFlatBuffers)
}

// Include generated FlatBuffers in source JAR
tasks.withType<Jar>().configureEach {
    if (name == "jvmJar" || name == "jsJar") {
        dependsOn(compileFlatBuffers)
    }
}

publishToMaven()
