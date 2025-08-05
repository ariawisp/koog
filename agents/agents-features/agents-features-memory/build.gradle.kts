import ai.koog.gradle.publish.maven.Publishing.publishToMaven

group = rootProject.group
version = rootProject.version

plugins {
    id("ai.kotlin.multiplatform")
    alias(libs.plugins.kotlin.serialization)
}

kotlin {
    sourceSets {
        commonMain {
            dependencies {
                api(project(":agents:agents-core"))
                api(project(":agents:agents-features:agents-features-common"))
                api(project(":agents:agents-features:agents-features-tokenizer"))
                api(project(":agents:agents-features:agents-features-event-handler"))
                api(project(":prompt:prompt-markdown"))
                api(project(":prompt:prompt-model"))
                api(project(":prompt:prompt-executor:prompt-executor-llms"))
                api(project(":rag:rag-base"))

                api(libs.kotlinx.serialization.json)
                api(libs.ktor.client.content.negotiation)
                api(libs.ktor.serialization.kotlinx.json)
                
                // Cryptography for secure memory operations
                api("dev.whyoleg.cryptography:cryptography-core:0.5.0")
                api("dev.whyoleg.cryptography:cryptography-provider-optimal:0.5.0")
            }
        }

        commonTest {
            dependencies {
                implementation(kotlin("test"))
                implementation(libs.kotlinx.coroutines.test)
                implementation(project(":agents:agents-test"))
                implementation(project(":prompt:prompt-executor:prompt-executor-clients:prompt-executor-openai-client"))
            }
        }

        jvmMain {
            dependencies {
                api(libs.ktor.client.cio)
            }
        }

        jvmTest {
            dependencies {
                implementation(kotlin("test-junit5"))
                implementation(project(":agents:agents-test"))
                implementation(libs.mockk)
            }
        }
    }

    explicitApi()
}

publishToMaven()
