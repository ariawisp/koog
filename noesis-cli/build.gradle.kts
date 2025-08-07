group = rootProject.group
version = rootProject.version

plugins {
    id("ai.kotlin.jvm")
    application
}

repositories {
    mavenLocal() // For locally built FlatBuffers dependencies
    mavenCentral()
    maven(url = "https://packages.jetbrains.team/maven/p/ij/intellij-dependencies")
}

// Configure the application plugin
application {
    mainClass.set("ai.koog.noesis.cli.NoesisCLIKt")
}

dependencies {
    // Core NoesisRuntime - the only dependency our CLI actually needs
    api(project(":prompt:prompt-executor:prompt-executor-clients:noesis-executor"))
    
    // CLI framework
    implementation(libs.clikt)
    
    // Basic utilities
    api(libs.kotlinx.datetime)
    implementation(libs.kotlinx.coroutines.core)
    
    // Minimal logging for CLI
    implementation(libs.logback.classic)

    testImplementation(kotlin("test"))
    testImplementation(libs.kotlinx.coroutines.test)
}

// Use the native library built by noesis-executor project instead of building our own
// The CLI will get the library from the noesis-executor JAR dependencies

// No need for separate build tasks - the native library comes from noesis-executor dependency
// The noesis-executor JAR already contains the built native library

// Register run task for the CLI
val runNoesisCLI = tasks.register<JavaExec>("runNoesisCLI") {
    group = "application"
    description = "Run the Noesis CLI"
    classpath = sourceSets.main.get().runtimeClasspath
    mainClass.set("ai.koog.noesis.cli.NoesisCLIKt")
    
    // Enable console input for interactive mode
    standardInput = System.`in`
    
    // Pass through JVM arguments if needed
    jvmArgs("-Xmx4g") // Give it enough memory for model operations
}
