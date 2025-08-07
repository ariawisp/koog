// DEPRECATED: This error system is replaced by the unified errors::NoesisError
// This file is kept for compatibility during migration only.
// 
// New code should use: crate::errors::{NoesisError, NoesisResult}

// Legacy compatibility - redirect to unified error system
pub use crate::errors::{NoesisError as InferenceError, NoesisResult as Result};