plugins {
    id("ai.kotlin.multiplatform")
    alias(libs.plugins.kotlin.serialization)
}

kotlin {
    jvm()
    js {
        useCommonJs()
        generateTypeScriptDefinitions()
        nodejs()
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(project(":agents:agents-core"))
                api(project(":agents:agents-features:agents-features-memory"))
                api(project(":agents:agents-features:agents-features-tokenizer"))
                api(project(":prompt:prompt-executor:prompt-executor-model"))
                api(project(":prompt:prompt-executor:prompt-executor-llms-all"))
                
                implementation(libs.kotlinx.serialization.json)
                implementation(libs.kotlinx.coroutines.core)
                implementation(libs.kotlinx.datetime)
                implementation(libs.oshai.kotlin.logging)
            }
        }

        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
                implementation(libs.kotlinx.coroutines.test)
                implementation(project(":agents:agents-test"))
            }
        }

        val jvmMain by getting {
            dependencies {
                implementation(libs.clikt)
            }
        }

        val jvmTest by getting {
            dependencies {
                implementation(kotlin("test-junit5"))
            }
        }
    }
}

tasks.withType<Test> {
    useJUnitPlatform()
}

// Create run task for benchmarks
tasks.register<JavaExec>("runBenchmark") {
    group = "benchmark"
    description = "Run the benchmark with command line arguments"
    
    dependsOn(tasks.getByName("jvmMainClasses"))
    
    val jvmTarget = kotlin.targets.getByName("jvm") as org.jetbrains.kotlin.gradle.targets.jvm.KotlinJvmTarget
    val mainCompilation = jvmTarget.compilations.getByName("main")
    
    classpath = mainCompilation.runtimeDependencyFiles + mainCompilation.output.allOutputs
    mainClass.set("ai.koog.agents.benchmark.cli.BenchmarkCliKt")
    
    // Set working directory to project root
    workingDir = rootProject.projectDir
    
    // Pass through command line arguments
    if (project.hasProperty("args")) {
        args(project.property("args").toString().split(" "))
    }
}