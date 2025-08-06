import ai.koog.gradle.publish.maven.Publishing.publishToMaven
import org.gradle.internal.os.OperatingSystem
import java.io.ByteArrayOutputStream

plugins {
    id("ai.kotlin.multiplatform")
    alias(libs.plugins.kotlin.serialization)
}

// Configuration for GPT-OSS Metal integration
val gptOssBuildDir = layout.buildDirectory.dir("gpt-oss")
val gptOssSourceDir = gptOssBuildDir.map { it.dir("source") }
val gptOssBuildPath = gptOssBuildDir.map { it.dir("build") }
val gptOssRepo = "https://github.com/openai/gpt-oss.git"
val gptOssCommit = "main" // You can pin to a specific commit/tag for reproducibility

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

// Task to build GPT-OSS Metal library
val buildGptOss = tasks.register<Exec>("buildGptOss") {
    enabled = OperatingSystem.current().isMacOsX
    
    dependsOn(cloneGptOss)
    
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

// Task to build the Metal inference native library
val buildMetalInferenceLib = tasks.register<Exec>("buildMetalInferenceLib") {
    val nativeDir = file("native/metal-inference-jni")
    
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
            
            logger.lifecycle("Building Metal inference JNI library")
            logger.lifecycle("  GPT-OSS path: ${buildDir.absolutePath}")
        }
        
        commandLine("cargo", "build", "--release")
        
        doLast {
            val libFile = nativeDir.resolve("target/release/libmetal_inference_jni.dylib")
            if (libFile.exists()) {
                // Copy to multiple architecture directories for compatibility
                val architectures = listOf("aarch64", "arm64", System.getProperty("os.arch"))
                architectures.forEach { arch ->
                    val resourceDir = file("src/jvmMain/resources/native/$arch")
                    resourceDir.mkdirs()
                    libFile.copyTo(resourceDir.resolve("libmetal_inference_jni.dylib"), overwrite = true)
                }
                
                // Also copy to build resources for immediate testing
                val buildResourceDir = file("build/processedResources/jvm/main/native/${System.getProperty("os.arch")}")
                buildResourceDir.mkdirs()
                libFile.copyTo(buildResourceDir.resolve("libmetal_inference_jni.dylib"), overwrite = true)
                
                // Copy Metal shader library
                val metalLib = file("${buildDir}/default.metallib")
                if (metalLib.exists()) {
                    val targetMetalLib = nativeDir.resolve("target/release/default.metallib")
                    metalLib.copyTo(targetMetalLib, overwrite = true)
                    logger.lifecycle("Copied Metal shader library to ${targetMetalLib}")
                    
                    // Also copy to resources
                    architectures.forEach { arch ->
                        val resourceDir = file("src/jvmMain/resources/native/$arch")
                        metalLib.copyTo(resourceDir.resolve("default.metallib"), overwrite = true)
                    }
                }
                
                logger.lifecycle("Metal inference library built and copied to resources")
            } else {
                throw GradleException("Failed to build Metal inference library")
            }
        }
    } else {
        doFirst {
            logger.lifecycle("Skipping Metal inference build on non-macOS platform")
        }
    }
}

// Task to embed Metal shaders into the dylib (macOS specific)
val embedMetalShaders = tasks.register<Exec>("embedMetalShaders") {
    enabled = OperatingSystem.current().isMacOsX
    
    dependsOn(buildMetalInferenceLib)
    
    val buildDir = gptOssBuildPath.get().asFile
    val libFile = file("native/metal-inference-jni/target/release/libmetal_inference_jni.dylib")
    val metalLib = file("${buildDir}/default.metallib")
    
    onlyIf {
        libFile.exists() && metalLib.exists()
    }
    
    doFirst {
        logger.lifecycle("Attempting to embed Metal shaders into JNI library")
    }
    
    // This is a macOS-specific approach to embed resources
    // Note: This might not work perfectly with all configurations
    commandLine("bash", "-c", """
        # Copy metallib to the same directory as the dylib
        cp "${metalLib.absolutePath}" "${libFile.parent}/default.metallib"
        
        # Try to use install_name_tool to add rpath if needed
        install_name_tool -add_rpath @loader_path "${libFile.absolutePath}" 2>/dev/null || true
        
        # Alternative: Try to create a fat binary with embedded resources (experimental)
        # This would require more complex post-processing
        echo "Metal shaders copied to: ${libFile.parent}/default.metallib"
    """.trimIndent())
    
    doLast {
        logger.lifecycle("Metal shader embedding/copying completed")
    }
}

// Task to copy native library to resources
val copyNativeLibraryToResources = tasks.register<Copy>("copyNativeLibraryToResources") {
    dependsOn(buildMetalInferenceLib, embedMetalShaders)
    
    val osArch = System.getProperty("os.arch").lowercase()
    // Copy to src resources so it's included in the JAR
    val resourceDir = file("src/jvmMain/resources/native/$osArch")
    
    from(file("native/metal-inference-jni/target/release")) {
        include("*.dylib", "*.so", "*.dll", "*.metallib")
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
val cleanMetalInferenceLib = tasks.register<Delete>("cleanMetalInferenceLib") {
    delete(file("native/metal-inference-jni/target"))
    delete(fileTree("src/jvmMain/resources/native") {
        include("**/*.dylib", "**/*.so", "**/*.dll", "**/*.metallib")
    })
    delete(fileTree("build/processedResources") {
        include("**/*.dylib", "**/*.so", "**/*.dll", "**/*.metallib")
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
    
    dependsOn(buildMetalInferenceLib, embedMetalShaders)
    dependsOn(tasks.named("compileKotlinJvm"))
    dependsOn(tasks.named("jvmJar"))
    
    mainClass.set("ai.koog.prompt.executor.clients.harmony.MetalInferenceJNI")
    classpath = sourceSets["jvmMain"].runtimeClasspath
    
    // Set library path for native library loading
    systemProperty("java.library.path", file("native/metal-inference-jni/target/release").absolutePath)
    
    // Set working directory to where the Metal shaders are
    workingDir = file("native/metal-inference-jni/target/release")
    
    doFirst {
        logger.lifecycle("Testing Metal inference JNI integration...")
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
                exec {
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
    dependsOn(cleanMetalInferenceLib)
    // Optionally clean GPT-OSS as well (uncomment if desired)
    // dependsOn(cleanGptOss)
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

// Make JVM compilation depend on native library build
tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile>().configureEach {
    if (name.contains("Jvm")) {
        dependsOn(buildMetalInferenceLib)
        if (OperatingSystem.current().isMacOsX) {
            dependsOn(embedMetalShaders)
        }
    }
}

// Include native library in JAR
tasks.withType<Jar>().configureEach {
    if (name == "jvmJar") {
        dependsOn(buildMetalInferenceLib, copyNativeLibraryToResources)
        duplicatesStrategy = DuplicatesStrategy.EXCLUDE
        if (OperatingSystem.current().isMacOsX) {
            dependsOn(embedMetalShaders)
        }
        // Resources are automatically included from src/jvmMain/resources
    }
}

// Configure test tasks to use the native library
tasks.withType<Test>().configureEach {
    if (OperatingSystem.current().isMacOsX) {
        dependsOn(buildMetalInferenceLib, embedMetalShaders, copyNativeLibraryToResources)
        
        // The library will be loaded from resources, so we just need the working directory
        // for Metal shaders to be found at runtime
        workingDir = file("native/metal-inference-jni/target/release")
        
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

publishToMaven()