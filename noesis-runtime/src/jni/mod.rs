// JNI bindings for Kotlin/Java integration

pub mod runtime;        // JNI bindings for NoesisRuntime
pub mod streaming;      // JNI bindings for streaming operations
pub mod memory;         // JNI bindings for memory operations
pub mod utils;          // JNI utility functions
pub mod binary;         // JNI binary FlatBuffers interface (ZERO-COPY)

// Re-export all JNI functions for backwards compatibility
pub use runtime::*;
pub use streaming::*;
pub use memory::*;
pub use utils::*;
pub use binary::*;