// Harmony subsystem
pub mod encoder; // Thin wrapper around openai_harmony for convenience
pub use encoder::{HarmonyEncoder, SpecialTokens};
// BREAKING: Removed internal parser module; use openai_harmony::StreamableParser
