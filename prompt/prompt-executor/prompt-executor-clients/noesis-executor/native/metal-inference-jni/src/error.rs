use thiserror::Error;

pub type Result<T> = std::result::Result<T, InferenceError>;

#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Metal device not available")]
    DeviceNotAvailable,
    
    #[error("Failed to create Metal device: {0}")]
    DeviceCreation(String),
    
    #[error("Failed to load model from path: {path}")]
    ModelLoadFailed { path: String },
    
    #[error("Model file not found: {path}")]
    ModelNotFound { path: String },
    
    #[error("Invalid model format: {details}")]
    InvalidModelFormat { details: String },
    
    #[error("Failed to create inference context: {0}")]
    ContextCreation(String),
    
    #[error("Inference failed: {0}")]
    InferenceFailed(String),
    
    #[error("Invalid input tokens: {0}")]
    InvalidInput(String),
    
    #[error("Out of memory: {0}")]
    OutOfMemory(String),
    
    #[error("Metal API error: {0}")]
    MetalError(String),
    
    #[error("FFI error: {0}")]
    FFIError(String),
    
    #[error("Null pointer encountered: {0}")]
    NullPointer(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("UTF-8 conversion error: {0}")]
    Utf8Error(#[from] std::str::Utf8Error),
    
    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl InferenceError {
    pub fn error_code(&self) -> i32 {
        match self {
            InferenceError::DeviceNotAvailable => 1001,
            InferenceError::DeviceCreation(_) => 1002,
            InferenceError::ModelLoadFailed { .. } => 2001,
            InferenceError::ModelNotFound { .. } => 2002,
            InferenceError::InvalidModelFormat { .. } => 2003,
            InferenceError::ContextCreation(_) => 3001,
            InferenceError::InferenceFailed(_) => 4001,
            InferenceError::InvalidInput(_) => 4002,
            InferenceError::OutOfMemory(_) => 5001,
            InferenceError::MetalError(_) => 6001,
            InferenceError::FFIError(_) => 7001,
            InferenceError::NullPointer(_) => 7002,
            InferenceError::ConfigError(_) => 8001,
            InferenceError::IoError(_) => 9001,
            InferenceError::Utf8Error(_) => 9002,
            InferenceError::Unknown(_) => 9999,
        }
    }
}