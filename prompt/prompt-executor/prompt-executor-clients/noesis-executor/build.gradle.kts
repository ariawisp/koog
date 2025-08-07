import ai.koog.gradle.publish.maven.Publishing.publishToMaven
import org.gradle.internal.os.OperatingSystem
import java.io.ByteArrayOutputStream
import java.io.FileOutputStream
import java.time.Duration
import java.util.Properties

plugins {
    id("ai.kotlin.multiplatform")
    alias(libs.plugins.kotlin.serialization)
}


// Load local properties for user-specific configuration
val localProperties = Properties()
val localPropertiesFile = rootProject.file("noesis.properties")
if (localPropertiesFile.exists()) {
    localPropertiesFile.inputStream().use { localProperties.load(it) }
    logger.lifecycle("Loaded noesis.properties with ${localProperties.size} settings")
} else {
    logger.lifecycle("No noesis.properties found. Copy noesis.properties.template to noesis.properties and configure your model path.")
}

// Configuration for GPT-OSS Metal integration
val gptOssBuildDir = layout.buildDirectory.dir("gpt-oss")
val gptOssSourceDir = gptOssBuildDir.map { it.dir("source") }
val gptOssBuildPath = gptOssBuildDir.map { it.dir("build") }
val gptOssRepo = "https://github.com/openai/gpt-oss.git"
val gptOssCommit = "main" // You can pin to a specific commit/tag for reproducibility

// Task to clone FlatBuffers from GitHub
val cloneFlatBuffers = tasks.register<Exec>("cloneFlatBuffers") {
    val flatBuffersDir = file("${layout.buildDirectory.get()}/flatbuffers-kmp")
    val flatBuffersRepo = "https://github.com/ariawisp/flatbuffers.git"
    val flatBuffersBranch = "kotlin-kmp-build-improvements"
    
    onlyIf {
        !flatBuffersDir.exists() || !file("${flatBuffersDir}/.git").exists()
    }
    
    doFirst {
        flatBuffersDir.parentFile.mkdirs()
        logger.lifecycle("Cloning FlatBuffers from ${flatBuffersRepo} (branch: ${flatBuffersBranch})")
    }
    
    commandLine("bash", "-c", """
        if [ -d "${flatBuffersDir}" ]; then
            cd "${flatBuffersDir}" && git fetch origin && git checkout ${flatBuffersBranch} && git pull origin ${flatBuffersBranch}
        else
            git clone -b ${flatBuffersBranch} ${flatBuffersRepo} "${flatBuffersDir}"
        fi
    """.trimIndent())
    
    doLast {
        logger.lifecycle("FlatBuffers source cloned/updated successfully")
    }
}

// Task to apply patches to FlatBuffers
val applyFlatBuffersPatches = tasks.register<Exec>("applyFlatBuffersPatches") {
    dependsOn(cloneFlatBuffers)
    
    val flatBuffersDir = file("${layout.buildDirectory.get()}/flatbuffers-kmp")
    val patchFile = file("patches/flatbuffers-disable-signing.patch")
    
    onlyIf {
        patchFile.exists() && flatBuffersDir.exists()
    }
    
    workingDir = flatBuffersDir
    
    doFirst {
        logger.lifecycle("Applying FlatBuffers patches...")
    }
    
    commandLine("bash", "-c", """
        # Check if patch is already applied
        if git apply --check "${patchFile.absolutePath}" 2>/dev/null; then
            echo "Applying patch: ${patchFile.name}"
            git apply "${patchFile.absolutePath}"
            echo "✅ Patch applied successfully"
        else
            echo "Patch already applied or not needed: ${patchFile.name}"
        fi
    """.trimIndent())
    
    isIgnoreExitValue = true // Don't fail if patch is already applied
    
    doLast {
        logger.lifecycle("✅ FlatBuffers patches processed")
    }
}

// Task to build FlatBuffers compiler AND Java/Kotlin libraries
val buildFlatBuffers = tasks.register<Exec>("buildFlatBuffers") {
    dependsOn(applyFlatBuffersPatches)
    
    val flatBuffersDir = file("${layout.buildDirectory.get()}/flatbuffers-kmp")
    val buildDir = file("${flatBuffersDir}/build")
    
    onlyIf {
        !file("${flatBuffersDir}/flatc").exists() || 
        !file("${flatBuffersDir}/java/target/flatbuffers-java-25.2.10.jar").exists()
    }
    
    workingDir = flatBuffersDir
    
    doFirst {
        buildDir.mkdirs()
        logger.lifecycle("Building FlatBuffers compiler...")
    }
    
    commandLine("bash", "-c", """
        mkdir -p build &&
        cd build &&
        cmake .. -DCMAKE_BUILD_TYPE=Release &&
        make -j`sysctl -n hw.ncpu` &&
        cp flatc ../
    """.trimIndent())
    
    doLast {
        val flatc = file("${flatBuffersDir}/flatc")
        if (flatc.exists()) {
            logger.lifecycle("FlatBuffers compiler built successfully at ${flatc.absolutePath}")
        } else {
            throw GradleException("Failed to build FlatBuffers compiler")
        }
    }
}

// Task to build and publish FlatBuffers Java library to Maven local
val publishFlatBuffersJava = tasks.register<Exec>("publishFlatBuffersJava") {
    dependsOn(buildFlatBuffers)
    
    val flatBuffersDir = file("${layout.buildDirectory.get()}/flatbuffers-kmp")
    val javaDir = file("${flatBuffersDir}/java")
    
    onlyIf {
        javaDir.exists()
    }
    
    workingDir = javaDir
    
    doFirst {
        logger.lifecycle("Building and publishing FlatBuffers Java library to Maven local...")
        // Compile Java sources
        file("${javaDir}/target/classes").mkdirs()
    }
    
    commandLine("bash", "-c", """
        javac -d target/classes src/main/java/com/google/flatbuffers/*.java &&
        cd target/classes &&
        jar cf ../flatbuffers-java-25.2.10-LOCAL.jar com &&
        mkdir -p ~/.m2/repository/com/google/flatbuffers/flatbuffers-java/25.2.10-LOCAL &&
        cp ../flatbuffers-java-25.2.10-LOCAL.jar ~/.m2/repository/com/google/flatbuffers/flatbuffers-java/25.2.10-LOCAL/ &&
        echo "✅ FlatBuffers Java library published to Maven local"
    """.trimIndent())
    
    doLast {
        logger.lifecycle("✅ FlatBuffers Java library published to Maven local")
    }
}

// Task to build and publish FlatBuffers Kotlin library to Maven local
val publishFlatBuffersKotlin = tasks.register<Exec>("publishFlatBuffersKotlin") {
    dependsOn(applyFlatBuffersPatches)
    
    val flatBuffersDir = file("${layout.buildDirectory.get()}/flatbuffers-kmp")
    val kotlinDir = file("${flatBuffersDir}/kotlin")
    
    onlyIf {
        kotlinDir.exists()
    }
    
    workingDir = kotlinDir
    
    doFirst {
        logger.lifecycle("Building and publishing FlatBuffers Kotlin library to Maven local...")
    }
    
    commandLine("./gradlew", ":flatbuffers-kotlin:publishToMavenLocal", "-x", "test", "-x", "generateFBTestClassesKt")
    
    doLast {
        logger.lifecycle("✅ FlatBuffers Kotlin library published to Maven local")
    }
}

// Task to publish all FlatBuffers libraries
val publishFlatBuffersLibraries = tasks.register("publishFlatBuffersLibraries") {
    dependsOn(publishFlatBuffersJava, publishFlatBuffersKotlin)
    
    doLast {
        logger.lifecycle("✅ All FlatBuffers libraries published to Maven local")
    }
}

// Task to build FlatBuffers Kotlin runtime
val buildFlatBuffersKotlin = tasks.register<GradleBuild>("buildFlatBuffersKotlin") {
    dependsOn(buildFlatBuffers)
    
    val flatBuffersDir = file("${layout.buildDirectory.get()}/flatbuffers-kmp")
    
    dir = file("${flatBuffersDir}/kotlin")
    tasks = listOf("publishToMavenLocal")
    
    doFirst {
        logger.lifecycle("Building FlatBuffers Kotlin runtime...")
    }
    
    doLast {
        logger.lifecycle("FlatBuffers Kotlin runtime published to Maven Local")
    }
}

// Task to clone GPT-OSS from GitHub
val cloneGptOss = tasks.register<Exec>("cloneGptOss") {
    enabled = OperatingSystem.current().isMacOsX
    
    val sourceDir = gptOssSourceDir.get().asFile
    
    // Only clone if not already present
    onlyIf {
        !sourceDir.exists() || !file("${sourceDir}/.git").exists()
    }
    
    doFirst {
        sourceDir.parentFile.mkdirs()
        logger.lifecycle("Cloning GPT-OSS from GitHub to ${sourceDir}")
    }
    
    commandLine("bash", "-c", """
        if [ -d "${sourceDir}" ]; then
            cd "${sourceDir}" && git fetch origin && git checkout ${gptOssCommit} && git pull origin ${gptOssCommit}
        else
            git clone ${gptOssRepo} "${sourceDir}" && cd "${sourceDir}" && git checkout ${gptOssCommit}
        fi
    """.trimIndent())
    
    doLast {
        logger.lifecycle("GPT-OSS source cloned/updated successfully")
    }
}

// Task to apply patches to GPT-OSS after cloning
val applyGptOssPatches = tasks.register<Exec>("applyGptOssPatches") {
    description = "Apply patches to GPT-OSS for Metal backend fixes"
    group = "gpt-oss"
    
    dependsOn(cloneGptOss)
    
    val sourceDir = gptOssSourceDir.get().asFile
    val patchFile = file("patches/gptoss-metal-backend-fix.patch")
    
    workingDir = sourceDir
    
    // Only run if patch exists and hasn't been applied
    onlyIf {
        if (!patchFile.exists() || !sourceDir.exists()) {
            return@onlyIf false
        }
        
        // Check if patch has already been applied
        val backendFile = sourceDir.resolve("_build/gpt_oss_build_backend/backend.py")
        if (backendFile.exists()) {
            val content = backendFile.readText()
            if (content.contains("\"True\"")) {
                logger.lifecycle("GPT-OSS Metal backend patch already applied, skipping")
                return@onlyIf false
            }
        }
        return@onlyIf true
    }
    
    doFirst {
        logger.lifecycle("Applying GPT-OSS Metal backend fix patch...")
    }
    
    // Apply patch using git apply (works even outside git repos)
    commandLine("git", "apply", "--ignore-whitespace", patchFile.absolutePath)
    
    doLast {
        logger.lifecycle("✅ Applied GPT-OSS Metal backend fix")
        logger.lifecycle("   Now 'GPTOSS_BUILD_METAL=True' will work correctly")
    }
}

// Task to build GPT-OSS Metal library
val buildGptOss = tasks.register<Exec>("buildGptOss") {
    enabled = OperatingSystem.current().isMacOsX
    
    dependsOn(cloneGptOss, applyGptOssPatches)
    
    val sourceDir = gptOssSourceDir.get().asFile
    val buildDir = gptOssBuildPath.get().asFile
    val metalSourceDir = file("${sourceDir}/gpt_oss/metal")
    
    // Only build if library doesn't exist
    onlyIf {
        !file("${buildDir}/libgptoss.a").exists() || 
        !file("${buildDir}/default.metallib").exists()
    }
    
    workingDir = metalSourceDir
    
    doFirst {
        if (!metalSourceDir.exists()) {
            throw GradleException("GPT-OSS Metal source not found at ${metalSourceDir.absolutePath}")
        }
        
        // Create build directory
        buildDir.mkdirs()
        
        logger.lifecycle("Building GPT-OSS Metal library in ${buildDir}")
    }
    
    commandLine("bash", "-c", """
        mkdir -p "${buildDir}" &&
        cd "${buildDir}" &&
        cmake "${metalSourceDir}" -DCMAKE_BUILD_TYPE=Release &&
        make -j`sysctl -n hw.ncpu`
    """.trimIndent())
    
    doLast {
        // Verify the build artifacts exist
        val libFile = file("${buildDir}/libgptoss.a")
        val metalLib = file("${buildDir}/default.metallib")
        
        if (!libFile.exists()) {
            throw GradleException("GPT-OSS build failed: libgptoss.a not found")
        }
        if (!metalLib.exists()) {
            throw GradleException("GPT-OSS build failed: default.metallib not found")
        }
        
        logger.lifecycle("GPT-OSS Metal library built successfully")
        logger.lifecycle("  Library: ${libFile.absolutePath}")
        logger.lifecycle("  Shaders: ${metalLib.absolutePath}")
    }
}

// Task to build the unified Noesis Runtime native library
val buildNoesisRuntime = tasks.register<Exec>("buildNoesisRuntime") {
    val nativeDir = file("../../../../noesis-runtime")
    
    workingDir = nativeDir
    
    // Only build on macOS
    enabled = OperatingSystem.current().isMacOsX
    
    if (enabled) {
        // Always depend on GPT-OSS build
        dependsOn(buildGptOss)
        
        val buildDir = gptOssBuildPath.get().asFile
        
        doFirst {
            // Set environment variables for the Rust build
            environment("GPTOSS_PATH", buildDir.absolutePath)
            environment("GPT_OSS_METAL_PATH", buildDir.absolutePath)
            
            logger.lifecycle("Building unified Noesis Runtime library")
            logger.lifecycle("  GPT-OSS path: ${buildDir.absolutePath}")
        }
        
        commandLine("cargo", "build", "--release")
        
        doLast {
            val libFile = nativeDir.resolve("target/release/libnoesis_runtime.dylib")
            if (libFile.exists()) {
                // Copy to multiple architecture directories for compatibility
                val architectures = listOf("aarch64", "arm64", System.getProperty("os.arch"))
                architectures.forEach { arch ->
                    val resourceDir = file("src/jvmMain/resources/native/$arch")
                    resourceDir.mkdirs()
                    libFile.copyTo(resourceDir.resolve("libnoesis_runtime.dylib"), overwrite = true)
                }
                
                // Also copy to build resources for immediate testing
                val buildResourceDir = file("build/processedResources/jvm/main/native/${System.getProperty("os.arch")}")
                buildResourceDir.mkdirs()
                libFile.copyTo(buildResourceDir.resolve("libnoesis_runtime.dylib"), overwrite = true)
                
                // Metal shaders are loaded directly from GPT-OSS build directory
                // No need to copy them - GPT-OSS handles Metal library loading
                
                logger.lifecycle("Noesis Runtime library built and copied to resources")
            } else {
                throw GradleException("Failed to build Noesis Runtime library")
            }
        }
    } else {
        doFirst {
            logger.lifecycle("Skipping Noesis Runtime build on non-macOS platform")
        }
    }
}

// Simplified: No need to embed shaders - GPT-OSS handles Metal library loading
// The default.metallib is loaded directly from the GPT-OSS build directory

// Task to copy native library to resources
val copyNativeLibraryToResources = tasks.register<Copy>("copyNativeLibraryToResources") {
    dependsOn(buildNoesisRuntime)
    
    val osArch = System.getProperty("os.arch").lowercase()
    // Copy to src resources so it's included in the JAR
    val resourceDir = file("src/jvmMain/resources/native/$osArch")
    
    from(file("../../../../noesis-runtime/target/release")) {
        include("*.dylib", "*.so", "*.dll")
        // Metal libraries are loaded from GPT-OSS build directory
    }
    into(resourceDir)
    
    doFirst {
        resourceDir.mkdirs()
    }
    
    doLast {
        logger.lifecycle("Copied native library and shaders to: $resourceDir")
    }
}

// Clean task for native library
val cleanNoesisRuntime = tasks.register<Delete>("cleanNoesisRuntime") {
    delete(file("../../../../noesis-runtime/target"))
    delete(fileTree("src/jvmMain/resources/native") {
        include("**/*.dylib", "**/*.so", "**/*.dll")
    })
    delete(fileTree("build/processedResources") {
        include("**/*.dylib", "**/*.so", "**/*.dll")
    })
}

// Clean task for GPT-OSS
val cleanGptOss = tasks.register<Delete>("cleanGptOss") {
    delete(gptOssBuildDir)
    doLast {
        logger.lifecycle("GPT-OSS build directory cleaned")
    }
}

// Task to test Metal inference
val testMetalInference = tasks.register<JavaExec>("testMetalInference") {
    enabled = OperatingSystem.current().isMacOsX
    
    dependsOn(buildNoesisRuntime)
    dependsOn(tasks.named("compileKotlinJvm"))
    dependsOn(tasks.named("jvmJar"))
    
    mainClass.set("ai.koog.prompt.executor.clients.harmony.MetalInferenceJNI")
    classpath = sourceSets["jvmMain"].runtimeClasspath
    
    // Set library path for native library loading
    systemProperty("java.library.path", file("../../../../noesis-runtime/target/release").absolutePath)
    
    // Set working directory to where the Metal shaders are
    workingDir = gptOssBuildPath.get().asFile
    
    doFirst {
        logger.lifecycle("Testing Metal inference JNI integration...")
    }
}

// Task to run Noesis Runtime
val runNoesis = tasks.register<JavaExec>("runNoesis") {
    group = "application"
    description = "Run Noesis Runtime with GPT-OSS model"
    
    dependsOn(buildNoesisRuntime)
    dependsOn(tasks.named("jvmJar"))
    dependsOn(tasks.named("compileKotlinJvm"))
    
    mainClass.set("ai.koog.noesis.NoesisMainKt")
    classpath = sourceSets["jvmMain"].runtimeClasspath
    
    // Set library path for native library loading
    val rustTarget = file("../../../../noesis-runtime/target/release")
    systemProperty("java.library.path", rustTarget.absolutePath)
    
    // Set working directory to where Metal shaders are
    workingDir = gptOssBuildPath.get().asFile
    
    // Pass model path from properties or use default
    val modelPath = localProperties.getProperty("gptoss.model.path")
        ?: "/Users/aria/gpt-oss-20b/metal/metal/model.bin"
    systemProperty("gptoss.model.path", modelPath)
    
    // Pass other configuration
    systemProperty("gptoss.test.max_tokens", localProperties.getProperty("gptoss.test.max_tokens", "100"))
    systemProperty("gptoss.test.temperature", localProperties.getProperty("gptoss.test.temperature", "0.7"))
    
    // Set environment for GPT-OSS
    environment("GPTOSS_PATH", gptOssBuildPath.get().asFile.absolutePath)
    environment("GPT_OSS_METAL_PATH", gptOssBuildPath.get().asFile.absolutePath)
    environment("RUST_LOG", "debug")
    environment("RUST_BACKTRACE", "full")
    
    // Enable JVM debugging output
    jvmArgs = listOf(
        "-Xmx8g",
        "-XX:+HeapDumpOnOutOfMemoryError",
        "-XX:HeapDumpPath=${layout.buildDirectory.get().asFile}/heap-dump.hprof",
        "-XX:ErrorFile=${layout.buildDirectory.get().asFile}/hs_err_pid.log",
        "-XX:+PrintCommandLineFlags",
        "-Djava.util.logging.level=ALL"
    )
    
    // Redirect output to file as well as console
    val timestamp = System.currentTimeMillis()
    val logFile = file("${layout.buildDirectory.get().asFile}/noesis-runtime-${timestamp}.log")
    val latestLogFile = file("${layout.buildDirectory.get().asFile}/noesis-runtime-latest.log")
    
    // Set a timeout of 30 seconds
    timeout.set(Duration.ofSeconds(30))
    
    // Redirect both stdout and stderr to file AND console
    standardOutput = org.apache.tools.ant.util.TeeOutputStream(
        System.out,
        FileOutputStream(logFile, true)
    )
    errorOutput = org.apache.tools.ant.util.TeeOutputStream(
        System.err,
        FileOutputStream(logFile, true)
    )
    
    doFirst {
        // Create log file
        logFile.parentFile.mkdirs()
        logFile.writeText("=== NOESIS RUNTIME LOG ===\n")
        logFile.appendText("Started at: ${System.currentTimeMillis()}\n")
        logFile.appendText("Model: ${modelPath}\n")
        logFile.appendText("Working directory: ${workingDir.absolutePath}\n\n")
        logger.lifecycle("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.lifecycle("          RUNNING NOESIS RUNTIME")
        logger.lifecycle("          Kotlin → JNI → Rust → Metal → GPT-OSS")
        logger.lifecycle("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.lifecycle("")
        logger.lifecycle("Library path: ${rustTarget.absolutePath}")
        logger.lifecycle("GPT-OSS path: ${gptOssBuildPath.get().asFile.absolutePath}")
        logger.lifecycle("Model path: ${modelPath}")
        logger.lifecycle("Working dir: ${workingDir.absolutePath}")
        logger.lifecycle("")
        
        // Check if model exists
        if (!file(modelPath).exists()) {
            logger.error("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logger.error("ERROR: Model file not found: ${modelPath}")
            logger.error("")
            logger.error("Please configure your model path:")
            logger.error("1. Copy noesis.properties.template to noesis.properties")
            logger.error("2. Set gptoss.model.path to your model file location")
            logger.error("3. Download models from: https://huggingface.co/collections/openai/gpt-oss-models")
            logger.error("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            throw GradleException("Model file not found. See instructions above.")
        }
        
        logger.lifecycle("Log file will be saved to: ${logFile.absolutePath}")
        logger.lifecycle("Latest log will be at: ${latestLogFile.absolutePath}")
        logger.lifecycle("Test will timeout after 30 seconds")
    }
    
    doLast {
        // Copy to latest log file for easy access
        if (logFile.exists()) {
            logFile.copyTo(latestLogFile, overwrite = true)
        }
        
        logger.lifecycle("")
        logger.lifecycle("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.lifecycle("Runtime log saved to: ${logFile.absolutePath}")
        logger.lifecycle("Latest log saved to: ${latestLogFile.absolutePath}")
        
        // Check for crash logs
        val crashLogPattern = file("${layout.buildDirectory.get().asFile}").listFiles { _, name ->
            name.startsWith("hs_err_pid") && name.endsWith(".log")
        }
        
        if (crashLogPattern != null && crashLogPattern.isNotEmpty()) {
            logger.error("JVM crash logs found:")
            crashLogPattern.forEach { crashLog ->
                logger.error("  - ${crashLog.absolutePath}")
            }
        }
        logger.lifecycle("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    }
}

// Task for running tests
val testNoesis = tasks.register<Test>("testNoesis") {
    group = "verification"
    description = "Run Noesis Runtime tests"
    
    dependsOn(buildNoesisRuntime)
    useJUnitPlatform()
    testClassesDirs = sourceSets["jvmTest"].output.classesDirs
    classpath = sourceSets["jvmTest"].runtimeClasspath
    
    // Set library path for native library loading
    val rustTarget = file("../../../../noesis-runtime/target/release")
    systemProperty("java.library.path", rustTarget.absolutePath)
    
    // Keep default working directory (project root) - don't set workingDir
    
    // Pass model path from properties
    val modelPath = localProperties.getProperty("gptoss.model.path")
    if (modelPath != null) {
        systemProperty("gptoss.model.path", modelPath)
    }
    
    // Set environment for GPT-OSS
    environment("GPTOSS_PATH", gptOssBuildPath.get().asFile.absolutePath)
    environment("GPT_OSS_METAL_PATH", gptOssBuildPath.get().asFile.absolutePath)
    
    testLogging {
        events("passed", "skipped", "failed")
        showStandardStreams = true
    }
}

// Task to show GPT-OSS status
val gptOssStatus = tasks.register("gptOssStatus") {
    doLast {
        val sourceDir = gptOssSourceDir.get().asFile
        val buildDir = gptOssBuildPath.get().asFile
        
        println("GPT-OSS Status:")
        println("  Repository: ${gptOssRepo}")
        println("  Branch/Tag: ${gptOssCommit}")
        println("  Source directory: ${sourceDir.absolutePath}")
        println("  Source exists: ${sourceDir.exists()}")
        
        if (sourceDir.exists() && file("${sourceDir}/.git").exists()) {
            try {
                val output = ByteArrayOutputStream()
                project.exec {
                    workingDir = sourceDir
                    commandLine("git", "rev-parse", "HEAD")
                    standardOutput = output
                    isIgnoreExitValue = true
                }
                val commitHash = output.toString().trim()
                if (commitHash.isNotEmpty()) {
                    println("  Current commit: ${commitHash}")
                }
            } catch (e: Exception) {
                println("  Could not get git commit info: ${e.message}")
            }
        }
        
        println("  Build directory: ${buildDir.absolutePath}")
        println("  Library built: ${file("${buildDir}/libgptoss.a").exists()}")
        println("  Shaders built: ${file("${buildDir}/default.metallib").exists()}")
    }
}

tasks.named("clean") {
    dependsOn(cleanNoesisRuntime)
    // Optionally clean GPT-OSS as well (uncomment if desired)
    // dependsOn(cleanGptOss)
}

// Task to generate FlatBuffer classes for Noesis events
val generateNoesisEventsFB = tasks.register<Exec>("generateNoesisEventsFB") {
    description = "Generate FlatBuffer classes for Noesis events"
    group = "build"
    
    dependsOn(buildFlatBuffers)
    
    // Use the flatc built from our fork
    val flatc = file("${layout.buildDirectory.get()}/flatbuffers-kmp/flatc")
    val schemaFile = file("schemas/noesis_events.fbs")
    // Generate to jvmMain since the generated files import Java classes
    val outputDir = file("src/jvmMain/kotlin")
    
    inputs.file(schemaFile)
    outputs.dir(outputDir.resolve("ai/koog/noesis/events"))
    
    onlyIf {
        schemaFile.exists()
    }
    
    doFirst {
        outputDir.resolve("ai/koog/noesis/events").mkdirs()
        logger.lifecycle("Generating FlatBuffer classes from ${schemaFile.name} to jvmMain")
    }
    
    commandLine(
        flatc.absolutePath,
        "--kotlin",
        "--gen-mutable", 
        "--gen-object-api",
        "-o", outputDir.absolutePath,
        schemaFile.absolutePath
    )
    
    doLast {
        logger.lifecycle("Generated FlatBuffer classes for Noesis events in jvmMain")
        
        // Fix union method return type issue in generated NoesisEvent.kt
        val noesisEventFile = outputDir.resolve("ai/koog/noesis/events/NoesisEvent.kt")
        if (noesisEventFile.exists()) {
            val content = noesisEventFile.readText()
            val fixedContent = content.replace(
                "fun data(obj: Table) : Table {",
                "fun data(obj: Table) : Table? {"
            )
            if (fixedContent != content) {
                noesisEventFile.writeText(fixedContent)
                logger.lifecycle("Applied union type fix to NoesisEvent.kt")
            }
        }
    }
}

// Configure repositories to include Maven Local for FlatBuffers
repositories {
    mavenLocal() // For locally built FlatBuffers
    mavenCentral()
}

kotlin {
    jvm()
    
    sourceSets {
        val commonMain by getting {
            dependencies {
                api(project(":prompt:prompt-executor:prompt-executor-clients"))
                api(project(":prompt:prompt-model"))
                implementation(libs.kotlinx.serialization.json)
                implementation(libs.kotlinx.datetime)
                implementation(libs.oshai.kotlin.logging)
                // Use locally-built FlatBuffers libraries from Maven local (built from our fork)
                implementation("com.google.flatbuffers:flatbuffers-java:25.2.10-LOCAL")
                implementation("com.google.flatbuffers.kotlin:flatbuffers-kotlin:2.0.0-SNAPSHOT")
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")
            }
        }
        
        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
            }
        }
        
        val jvmMain by getting {
            dependencies {
                // JNI and native library support is in prompt-model
            }
            
            // Ensure resources are processed
            resources.srcDir("src/jvmMain/resources")
        }
        
        val jvmTest by getting {
            dependencies {
                implementation(kotlin("test-junit5"))
            }
        }
    }
}

// Make JVM compilation depend on native library build and FlatBuffer generation
tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile>().configureEach {
    dependsOn(generateNoesisEventsFB)
    // Ensure FlatBuffers Kotlin is available from Maven Local
    if (name.contains("Common") || name.contains("Main")) {
        dependsOn(buildFlatBuffersKotlin)
    }
    if (name.contains("Jvm")) {
        dependsOn(buildNoesisRuntime)
        if (OperatingSystem.current().isMacOsX) {
            // Metal shaders are loaded directly from GPT-OSS build
        }
    }
}

// Include native library in JAR
tasks.withType<Jar>().configureEach {
    if (name == "jvmJar") {
        dependsOn(buildNoesisRuntime, copyNativeLibraryToResources)
        duplicatesStrategy = DuplicatesStrategy.EXCLUDE
        if (OperatingSystem.current().isMacOsX) {
            // Metal shaders are loaded directly from GPT-OSS build
        }
        // Resources are automatically included from src/jvmMain/resources
    }
}

// Configure test tasks to use the native library
tasks.withType<Test>().configureEach {
    if (OperatingSystem.current().isMacOsX) {
        dependsOn(buildNoesisRuntime, copyNativeLibraryToResources)
        
        // The library will be loaded from resources, so we just need the working directory
        // for Metal shaders to be found at runtime
        workingDir = gptOssBuildPath.get().asFile
        
        // Enable more detailed error messages
        jvmArgs("-Djava.awt.headless=true")
        testLogging {
            events("passed", "skipped", "failed", "standardOut", "standardError")
            showExceptions = true
            showCauses = true
            showStackTraces = true
        }
    }
}

// Ensure native library is copied to resources before processing
tasks.named("jvmProcessResources") {
    if (OperatingSystem.current().isMacOsX) {
        dependsOn(copyNativeLibraryToResources)
    }
}

// Master build task that builds everything needed for Noesis
val buildNoesisComplete = tasks.register("buildNoesisComplete") {
    group = "build"
    description = "Complete build of Noesis Runtime with all dependencies"
    
    dependsOn(
        buildFlatBuffersKotlin,       // Build FlatBuffers first
        generateNoesisEventsFB,        // Generate FlatBuffer classes
        applyGptOssPatches,           // Apply GPT-OSS patches
        buildGptOss,                  // Build GPT-OSS with Metal
        buildNoesisRuntime,           // Build Rust runtime
        copyNativeLibraryToResources  // Copy to resources
    )
    
    doLast {
        logger.lifecycle("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.lifecycle("✅ NOESIS RUNTIME BUILD COMPLETE")
        logger.lifecycle("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.lifecycle("")
        logger.lifecycle("Built components:")
        logger.lifecycle("  • FlatBuffers compiler and Kotlin runtime")
        logger.lifecycle("  • GPT-OSS Metal backend with native shaders")
        logger.lifecycle("  • Noesis Rust runtime (Metal shaders loaded from GPT-OSS)")
        logger.lifecycle("  • Native libraries deployed to resources")
        logger.lifecycle("")
        logger.lifecycle("You can now run: ./gradlew runNoesis")
        logger.lifecycle("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    }
}

publishToMaven()
