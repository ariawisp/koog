import org.jetbrains.kotlin.gradle.dsl.KotlinProjectExtension
import org.jetbrains.kotlin.gradle.dsl.KotlinVersion
import org.jetbrains.kotlin.gradle.tasks.KotlinCompilationTask
import org.jetbrains.kotlin.gradle.tasks.KotlinJvmCompile

extensions.getByType<KotlinProjectExtension>().apply {
    jvmToolchain {
        languageVersion.set(JavaLanguageVersion.of(17))
    }
    sourceSets.all {
        languageSettings {
            // K/Common
            optIn("kotlin.RequiresOptIn")
            optIn("kotlinx.serialization.ExperimentalSerializationApi")
            optIn("kotlinx.coroutines.ExperimentalCoroutinesApi")
            // K/JS
            optIn("kotlin.js.ExperimentalJsExport")
        }
    }
}

val kotlinLanguageVersion = KotlinVersion.KOTLIN_2_2
val kotlinApiVersion = KotlinVersion.KOTLIN_2_2

tasks.withType<KotlinCompilationTask<*>>().configureEach {
    compilerOptions {
        languageVersion.set(kotlinLanguageVersion)
        logger.info("'$path' Kotlin language version: $kotlinLanguageVersion")
        apiVersion.set(kotlinApiVersion)
        logger.info("'$path' Kotlin API version: $kotlinApiVersion")
        
        // Enable Kotlin 2.2 preview features for LLM parameter refactor
        freeCompilerArgs.addAll(
            // Core feature: Context parameters for implicit provider capabilities
            "-Xcontext-parameters",
            
            // Cleaner enum and sealed class resolution in DSLs
            "-Xcontext-sensitive-resolution",
            
            // Enable nested type aliases for domain-specific types
            "-Xnested-type-aliases",
            
            // Apply annotations to all property use-site targets
            "-Xannotation-target-all",
            
            // Improve default annotation targeting behavior
            "-Xannotation-default-target=param-property",
            
            // Store annotations in Kotlin metadata for future tooling
            "-Xannotations-in-metadata"
        )
        
        logger.info("'$path' Kotlin 2.2 features enabled: context parameters, context-sensitive resolution, nested type aliases, annotation enhancements")
    }
}

tasks.withType<KotlinJvmCompile>().configureEach {
    compilerOptions {
        freeCompilerArgs.add("-Xjvm-default=all")
    }
}
