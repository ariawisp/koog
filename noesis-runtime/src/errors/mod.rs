// PHASE 8A: UNIFIED ERROR ARCHITECTURE - Strategic Consolidation Complete
//
// This module now provides a single, strategic error hierarchy that replaces
// both InferenceError and the previous complex NoesisError system.
//
// Architecture: 4 core error categories focused on actionability and recovery

mod unified;
pub use unified::*;

// === LEGACY COMPATIBILITY ===
// Type aliases for smooth migration from old error systems

/// Legacy compatibility for old Result types
pub type Result<T> = NoesisResult<T>;

// === MIGRATION HELPERS ===
// These help components migrate from the old error systems

impl NoesisError {
    /// Migration helper: Create cognitive component error
    pub fn channel_routing(message: &str) -> Self {
        Self::cognitive(&format!("Channel routing: {}", message))
    }
    
    /// Migration helper: Create harmony error
    pub fn harmony(message: &str) -> Self {
        Self::runtime("harmony processing", std::io::Error::new(std::io::ErrorKind::Other, message), true)
    }
    
    /// Migration helper: Create contradiction detection error
    pub fn contradiction_detection(message: &str) -> Self {
        Self::cognitive(&format!("Contradiction detection: {}", message))
    }
    
    /// Migration helper: Create semantic processing error
    pub fn semantic_processing(message: &str) -> Self {
        Self::cognitive(&format!("Semantic processing: {}", message))
    }
    
    /// Migration helper: Create safety violation error
    pub fn safety_violation(message: &str) -> Self {
        Self::cognitive(&format!("Safety violation: {}", message))
    }
    
    /// Migration helper: Create state management error
    pub fn state_management(message: &str) -> Self {
        Self::cognitive(&format!("State management: {}", message))
    }
    
    /// Migration helper: Invalid input error
    pub fn invalid_input(message: &str) -> Self {
        Self::input(message)
    }
    
    /// Migration helper: Resource exhausted error  
    pub fn resource_exhausted(resource: ResourceType, message: &str) -> Self {
        Self::Resource {
            resource_type: resource,
            cause: Some(Box::new(std::io::Error::new(std::io::ErrorKind::Other, message))),
            retry_after: None,
            context: None,
        }
    }
    
    /// Migration helper: Backend initialization error
    pub fn backend_initialization(message: &str) -> Self {
        Self::system(&format!("Backend initialization: {}", message))
    }
    
    /// Migration helper: Memory allocation error
    pub fn memory_allocation(message: &str) -> Self {
        Self::resource(ResourceType::GpuMemory, Some(Box::new(std::io::Error::new(std::io::ErrorKind::Other, message))))
    }
    
    /// Migration helper: Buffer operation error
    pub fn buffer_operation(op: BufferOperation, message: &str) -> Self {
        Self::Runtime {
            operation: format!("Buffer {:?}", op),
            cause: Box::new(std::io::Error::new(std::io::ErrorKind::Other, message)),
            recoverable: true,
            context: None,
        }
    }
    
    /// Migration helper: Compute pipeline error
    pub fn compute_pipeline(message: &str) -> Self {
        Self::resource(ResourceType::ComputePipeline, Some(Box::new(std::io::Error::new(std::io::ErrorKind::Other, message))))
    }
}