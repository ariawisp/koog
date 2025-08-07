// ERROR MIGRATION GUIDE - NoesisError Unification
//
// BREAKING CHANGE GUIDE: How to migrate from fragmented errors to NoesisError
// This file contains examples and utilities for migrating existing code

use super::*;

/// Migration utilities for converting old error types to NoesisError
pub struct ErrorMigration;

impl ErrorMigration {
    /// Convert old CognitiveError string to NoesisError
    pub fn from_cognitive_string(message: &str, component_name: &str) -> NoesisError {
        let component = match component_name {
            "channel_router" | "ChannelRouter" => CognitiveComponent::ChannelRouter,
            "contradiction_detector" | "ContradictionDetector" => CognitiveComponent::ContradictionDetector,
            "semantic_processor" | "SemanticProcessor" => CognitiveComponent::SemanticProcessor,
            "state_manager" | "StateManager" => CognitiveComponent::StateManager,
            "safety_filter" | "SafetyFilter" => CognitiveComponent::SafetyFilter,
            _ => CognitiveComponent::HarmonyIntegration, // Default fallback
        };
        
        NoesisError::cognitive(message, component)
    }
    
    /// Convert old InferenceError to NoesisError
    pub fn from_inference_error(error_type: &str, message: &str, path: Option<&str>) -> NoesisError {
        match error_type {
            "ModelNotFound" => NoesisError::ModelNotFound {
                path: path.unwrap_or("unknown").to_string(),
                search_paths: vec![],
            },
            "ModelLoadFailed" => NoesisError::ModelLoadFailed {
                path: path.unwrap_or("unknown").to_string(),
                model_type: None,
                cause: None,
            },
            "DeviceNotAvailable" => NoesisError::BackendInitialization {
                message: message.to_string(),
                backend_type: BackendType::Metal, // Default assumption
                cause: None,
            },
            "InferenceFailed" => NoesisError::InferenceOperation {
                message: message.to_string(),
                model_handle: None,
                context_length: None,
                cause: None,
            },
            _ => NoesisError::Generic {
                message: format!("{}: {}", error_type, message),
                operation: Some("inference".to_string()),
                cause: None,
            },
        }
    }
    
    /// Convert old RoutingError to NoesisError
    pub fn from_routing_error(message: &str, channel: Option<&str>, is_security: bool) -> NoesisError {
        NoesisError::ChannelRouting {
            message: message.to_string(),
            channel: channel.map(|s| s.to_string()),
            security_violation: is_security,
            cause: None,
        }
    }
}

/// MIGRATION EXAMPLES
/// 
/// This section shows how to convert old error patterns to new ones

#[cfg(feature = "migration_examples")]
mod examples {
    use super::*;
    
    // OLD PATTERN:
    // return Err(CognitiveError::ChannelError("Invalid channel".to_string()));
    //
    // NEW PATTERN:
    fn new_channel_error() -> NoesisResult<()> {
        Err(NoesisError::channel_routing("Invalid channel"))
    }
    
    // OLD PATTERN:
    // return Err(InferenceError::ModelNotFound { path: model_path.to_string() });
    //
    // NEW PATTERN:
    fn new_model_error(model_path: &str) -> NoesisResult<()> {
        Err(NoesisError::ModelNotFound {
            path: model_path.to_string(),
            search_paths: vec!["/models".to_string()],
        })
    }
    
    // OLD PATTERN:
    // return Err(CognitiveError::GpuError(format!("GPU operation failed: {}", details)));
    //
    // NEW PATTERN:
    fn new_gpu_error(details: &str) -> NoesisResult<()> {
        Err(NoesisError::gpu_operation(
            format!("GPU operation failed: {}", details),
            BackendType::Metal
        ))
    }
    
    // OLD PATTERN:
    // return Err(CognitiveError::HarmonyError(format!("Parser failed: {}", msg)));
    //
    // NEW PATTERN:
    fn new_harmony_error(msg: &str) -> NoesisResult<()> {
        Err(NoesisError::harmony(format!("Parser failed: {}", msg)))
    }
    
    // CHAINING ERRORS (NEW FEATURE):
    fn error_with_context() -> NoesisResult<()> {
        Err(NoesisError::harmony("Token parsing failed")
            .with_context("Processing user input"))
    }
    
    // ENHANCED ERROR INFORMATION (NEW FEATURE):
    fn detailed_contradiction_error() -> NoesisResult<()> {
        Err(NoesisError::ContradictionDetection {
            message: "Logical inconsistency detected".to_string(),
            contradiction_id: Some("LOGIC_001".to_string()),
            severity: 0.85,
            cause: None,
        })
    }
    
    // ASYNC ERROR HANDLING (NEW FEATURE):
    async fn async_error_handling() -> NoesisResult<()> {
        let result = tokio::spawn(async {
            // Some async work
            Ok::<(), NoesisError>(())
        }).await;
        
        match result {
            Ok(inner_result) => inner_result,
            Err(join_error) => Err(NoesisError::from(join_error)),
        }
    }
}

/// BREAKING CHANGES SUMMARY
/// 
/// 1. ALL error types unified under NoesisError
/// 2. Result<T, CognitiveError> → NoesisResult<T>
/// 3. Result<T, InferenceError> → NoesisResult<T>
/// 4. String errors now have structured fields
/// 5. Error context and chaining supported
/// 6. Enhanced error categorization and severity
/// 7. Async-specific error variants added

/// MIGRATION CHECKLIST
/// 
/// □ Replace all CognitiveError with NoesisError
/// □ Replace all InferenceError with NoesisError  
/// □ Replace all RoutingError with NoesisError
/// □ Update Result<T, OldError> to NoesisResult<T>
/// □ Use structured error constructors (NoesisError::cognitive, etc.)
/// □ Add error context where beneficial
/// □ Update error handling to use .category() and .severity()
/// □ Test error recovery logic with .is_recoverable()

/// TESTING UTILITIES
#[cfg(test)]
pub mod testing {
    use super::*;
    
    /// Helper to test error conversion
    pub fn assert_error_conversion<F>(old_error_fn: F, expected_category: &str) 
    where
        F: FnOnce() -> NoesisError,
    {
        let error = old_error_fn();
        assert_eq!(error.category(), expected_category);
    }
    
    /// Helper to test error severity
    pub fn assert_error_severity<F>(error_fn: F, expected_severity: ErrorSeverity)
    where
        F: FnOnce() -> NoesisError,
    {
        let error = error_fn();
        assert_eq!(error.severity(), expected_severity);
    }
    
    #[test]
    fn test_migration_utilities() {
        // Test cognitive error migration
        let error = ErrorMigration::from_cognitive_string("Test error", "ChannelRouter");
        assert_eq!(error.category(), "cognitive");
        
        // Test inference error migration
        let error = ErrorMigration::from_inference_error("ModelNotFound", "Model not found", Some("/path/to/model"));
        assert_eq!(error.category(), "inference");
        
        // Test routing error migration
        let error = ErrorMigration::from_routing_error("Security violation", Some("analysis"), true);
        assert_eq!(error.category(), "cognitive");
    }
}