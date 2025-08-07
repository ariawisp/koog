// PHASE 8A: UNIFIED ERROR ARCHITECTURE - Strategic Consolidation
//
// This replaces the fragmented error systems (InferenceError + NoesisError) with a 
// single strategic hierarchy optimized for clarity and performance.
//
// Design Principles:
// 1. Only 4 core error categories (no explosion of variants)
// 2. Rich context without complexity
// 3. Clear recovery guidance
// 4. Efficient error construction

use thiserror::Error;
use std::fmt;

/// Unified error type for all Noesis Runtime operations
///
/// This replaces both InferenceError and the complex NoesisError hierarchy
/// with a strategic 4-category system focused on actionability.
#[derive(Debug, Error)]
pub enum NoesisError {
    /// Runtime operation failed - recoverable or system issues
    #[error("Runtime operation failed: {operation}")]
    Runtime {
        operation: String,
        #[source] 
        cause: Box<dyn std::error::Error + Send + Sync>,
        recoverable: bool,
        context: Option<String>,
    },
    
    /// Resource unavailable or exhausted
    #[error("Resource unavailable: {resource_type}")]
    Resource {
        resource_type: ResourceType,
        #[source] 
        cause: Option<Box<dyn std::error::Error + Send + Sync>>,
        retry_after: Option<std::time::Duration>,
        context: Option<String>,
    },
    
    /// Invalid input or configuration
    #[error("Invalid input: {message}")]
    Input { 
        message: String,
        field: Option<String>,
        context: Option<String>,
    },
    
    /// System limitation or capability not available
    #[error("System limitation: {message}")]
    System { 
        message: String,
        capability: Option<String>,
        context: Option<String>,
    },
}

/// Resource types for strategic error categorization
#[derive(Debug, Clone)]
pub enum ResourceType {
    GpuMemory,
    GpuContext,
    ModelWeights,
    Buffer,
    ComputePipeline,
    Device,
    Thread,
    NetworkConnection,
}

impl fmt::Display for ResourceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResourceType::GpuMemory => write!(f, "GPU memory"),
            ResourceType::GpuContext => write!(f, "GPU context"),
            ResourceType::ModelWeights => write!(f, "model weights"),
            ResourceType::Buffer => write!(f, "buffer"),
            ResourceType::ComputePipeline => write!(f, "compute pipeline"),
            ResourceType::Device => write!(f, "device"),
            ResourceType::Thread => write!(f, "thread"),
            ResourceType::NetworkConnection => write!(f, "network connection"),
        }
    }
}

/// Recovery actions that can be taken for errors
#[derive(Debug, Clone)]
pub enum RecoveryAction {
    Retry,
    RetryWithBackoff,
    FreeMemory,
    ReduceBatchSize,
    SwitchBackend,
    RestartContext,
    ReloadModel,
    None,
}

/// Cognitive component types for error context
#[derive(Debug, Clone)]
pub enum CognitiveComponent {
    ChannelRouter,
    ContradictionDetector,
    SemanticProcessor,
    StateManager,
    SafetyFilter,
    HarmonyProcessor,
    AsyncEngine,
}

/// Buffer operation types
#[derive(Debug, Clone)]
pub enum BufferOperation {
    Read,
    Write,
    Allocate,
    Deallocate,
    Map,
    Unmap,
}

/// State management operations
#[derive(Debug, Clone)]  
pub enum StateOperation {
    Fork,
    Merge,
    Checkpoint,
    Restore,
    Update,
    Query,
}

impl NoesisError {
    /// Determine if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            NoesisError::Runtime { recoverable, .. } => *recoverable,
            NoesisError::Resource { .. } => true,  // Resource issues often recoverable
            NoesisError::Input { .. } => false,    // Input errors need fixing
            NoesisError::System { .. } => false,   // System limitations are permanent
        }
    }
    
    /// Get suggested recovery action
    pub fn recovery_suggestion(&self) -> Option<RecoveryAction> {
        match self {
            NoesisError::Runtime { recoverable: true, .. } => Some(RecoveryAction::Retry),
            NoesisError::Resource { resource_type, .. } => match resource_type {
                ResourceType::GpuMemory => Some(RecoveryAction::FreeMemory),
                ResourceType::GpuContext => Some(RecoveryAction::RestartContext),
                ResourceType::ModelWeights => Some(RecoveryAction::ReloadModel),
                ResourceType::Buffer => Some(RecoveryAction::ReduceBatchSize),
                _ => Some(RecoveryAction::RetryWithBackoff),
            },
            _ => None,
        }
    }
    
    /// Get error severity for logging/metrics
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            NoesisError::Runtime { recoverable: true, .. } => ErrorSeverity::Warning,
            NoesisError::Runtime { recoverable: false, .. } => ErrorSeverity::Error,
            NoesisError::Resource { .. } => ErrorSeverity::Warning,
            NoesisError::Input { .. } => ErrorSeverity::Info,
            NoesisError::System { .. } => ErrorSeverity::Error,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ErrorSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

// === STRATEGIC ERROR CONSTRUCTORS ===
// These provide convenient ways to construct errors while maintaining the unified hierarchy

impl NoesisError {
    /// Runtime operation error
    pub fn runtime<E: std::error::Error + Send + Sync + 'static>(
        operation: &str,
        cause: E,
        recoverable: bool,
    ) -> Self {
        Self::Runtime {
            operation: operation.to_string(),
            cause: Box::new(cause),
            recoverable,
            context: None,
        }
    }
    
    /// Resource unavailable error
    pub fn resource(
        resource_type: ResourceType,
        cause: Option<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self::Resource {
            resource_type,
            cause,
            retry_after: None,
            context: None,
        }
    }
    
    /// Invalid input error
    pub fn input(message: &str) -> Self {
        Self::Input {
            message: message.to_string(),
            field: None,
            context: None,
        }
    }
    
    /// System limitation error
    pub fn system(message: &str) -> Self {
        Self::System {
            message: message.to_string(),
            capability: None,
            context: None,
        }
    }
    
    /// Add context to any error
    pub fn with_context(mut self, context: &str) -> Self {
        match &mut self {
            NoesisError::Runtime { context: ref mut ctx, .. } => *ctx = Some(context.to_string()),
            NoesisError::Resource { context: ref mut ctx, .. } => *ctx = Some(context.to_string()),
            NoesisError::Input { context: ref mut ctx, .. } => *ctx = Some(context.to_string()),
            NoesisError::System { context: ref mut ctx, .. } => *ctx = Some(context.to_string()),
        }
        self
    }
}

// === CONVERSIONS FROM STANDARD ERRORS ===

impl From<std::io::Error> for NoesisError {
    fn from(e: std::io::Error) -> Self {
        NoesisError::runtime("IO operation", e, true)
    }
}

impl From<std::str::Utf8Error> for NoesisError {
    fn from(e: std::str::Utf8Error) -> Self {
        NoesisError::input(&format!("Invalid UTF-8: {}", e))
    }
}

impl From<std::string::FromUtf8Error> for NoesisError {
    fn from(e: std::string::FromUtf8Error) -> Self {
        NoesisError::input(&format!("Invalid UTF-8 string: {}", e))
    }
}

impl From<serde_json::Error> for NoesisError {
    fn from(e: serde_json::Error) -> Self {
        NoesisError::input(&format!("JSON parsing error: {}", e))
    }
}

impl From<anyhow::Error> for NoesisError {
    fn from(e: anyhow::Error) -> Self {
        NoesisError::system(&e.to_string())
    }
}

impl From<String> for NoesisError {
    fn from(s: String) -> Self {
        NoesisError::system(&s)
    }
}

impl From<&str> for NoesisError {
    fn from(s: &str) -> Self {
        NoesisError::system(s)
    }
}

/// Unified result type for all Noesis Runtime operations
pub type NoesisResult<T> = std::result::Result<T, NoesisError>;

// === COMPATIBILITY HELPERS ===
// These ease migration from the old error systems

impl NoesisError {
    /// Create GPU operation error (common case)
    pub fn gpu_operation(message: &str, backend_type: BackendType) -> Self {
        NoesisError::Runtime {
            operation: format!("GPU operation ({})", backend_type),
            cause: Box::new(std::io::Error::new(std::io::ErrorKind::Other, message)),
            recoverable: true,
            context: Some(format!("Backend: {}", backend_type)),
        }
    }
    
    /// Create inference error (common case)
    pub fn inference(message: &str) -> Self {
        NoesisError::runtime("inference", std::io::Error::new(std::io::ErrorKind::Other, message), true)
    }
    
    /// Create cognitive error (common case)
    pub fn cognitive(message: &str) -> Self {
        NoesisError::runtime("cognitive processing", std::io::Error::new(std::io::ErrorKind::Other, message), true)
    }
    
    /// Create memory error (common case)
    pub fn memory(message: &str) -> Self {
        NoesisError::resource(ResourceType::GpuMemory, Some(Box::new(std::io::Error::new(std::io::ErrorKind::Other, message))))
    }
}

#[derive(Debug, Clone)]
pub enum BackendType {
    Metal,
    Cuda,
    Onnx,
    Mock,
}

impl fmt::Display for BackendType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendType::Metal => write!(f, "Metal"),
            BackendType::Cuda => write!(f, "CUDA"),
            BackendType::Onnx => write!(f, "ONNX"),
            BackendType::Mock => write!(f, "Mock"),
        }
    }
}