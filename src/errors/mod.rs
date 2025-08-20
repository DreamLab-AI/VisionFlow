//! Comprehensive error handling for the VisionFlow system
//! 
//! This module provides a unified error handling approach to replace
//! all panic! and unwrap() calls with proper error propagation.

use std::fmt;

/// Top-level error type for the VisionFlow system
#[derive(Debug, Clone)]
pub enum VisionFlowError {
    /// Actor system related errors
    Actor(ActorError),
    /// GPU computation errors
    GPU(GPUError),
    /// Settings and configuration errors
    Settings(SettingsError),
    /// Network and communication errors
    Network(NetworkError),
    /// File system and I/O errors
    IO(std::sync::Arc<std::io::Error>),
    /// Serialization/Deserialization errors
    Serialization(String),
    /// Generic error with context
    Generic { 
        message: String,
        source: Option<std::sync::Arc<dyn std::error::Error + Send + Sync + 'static>>
    },
}

/// Actor system specific errors
#[derive(Debug, Clone)]
pub enum ActorError {
    /// Actor failed to start
    StartupFailed { actor_name: String, reason: String },
    /// Actor crashed during operation
    RuntimeFailure { actor_name: String, reason: String },
    /// Message handling failed
    MessageHandlingFailed { message_type: String, reason: String },
    /// Actor supervision failure
    SupervisionFailed { supervisor: String, supervised: String, reason: String },
    /// Actor mailbox full or inaccessible
    MailboxError { actor_name: String, reason: String },
}

/// GPU computation specific errors
#[derive(Debug, Clone)]
pub enum GPUError {
    /// CUDA device initialization failed
    DeviceInitializationFailed(String),
    /// GPU memory allocation failed
    MemoryAllocationFailed { requested_bytes: usize, reason: String },
    /// Kernel execution failed
    KernelExecutionFailed { kernel_name: String, reason: String },
    /// Data transfer between CPU and GPU failed
    DataTransferFailed { direction: DataTransferDirection, reason: String },
    /// GPU compute fallback to CPU
    FallbackToCPU { reason: String },
    /// GPU driver or runtime error
    DriverError(String),
}

#[derive(Debug, Clone)]
pub enum DataTransferDirection {
    CPUToGPU,
    GPUToCPU,
}

/// Settings and configuration errors
#[derive(Debug, Clone)]
pub enum SettingsError {
    /// Settings file not found or inaccessible
    FileNotFound(String),
    /// Settings parsing failed
    ParseError { file_path: String, reason: String },
    /// Settings validation failed
    ValidationFailed { setting_path: String, reason: String },
    /// Settings save failed
    SaveFailed { file_path: String, reason: String },
    /// Cache access failed
    CacheError(String),
}

/// Network and communication errors
#[derive(Debug, Clone)]
pub enum NetworkError {
    /// TCP connection failed
    ConnectionFailed { host: String, port: u16, reason: String },
    /// WebSocket connection issues
    WebSocketError(String),
    /// MCP protocol errors
    MCPError { method: String, reason: String },
    /// HTTP request/response errors
    HTTPError { url: String, status: Option<u16>, reason: String },
    /// Timeout errors
    Timeout { operation: String, timeout_ms: u64 },
}

impl fmt::Display for VisionFlowError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            VisionFlowError::Actor(e) => write!(f, "Actor Error: {}", e),
            VisionFlowError::GPU(e) => write!(f, "GPU Error: {}", e),
            VisionFlowError::Settings(e) => write!(f, "Settings Error: {}", e),
            VisionFlowError::Network(e) => write!(f, "Network Error: {}", e),
            VisionFlowError::IO(e) => write!(f, "IO Error: {}", e),
            VisionFlowError::Serialization(e) => write!(f, "Serialization Error: {}", e),
            VisionFlowError::Generic { message, .. } => write!(f, "Error: {}", message),
        }
    }
}

impl fmt::Display for ActorError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ActorError::StartupFailed { actor_name, reason } => 
                write!(f, "Actor '{}' failed to start: {}", actor_name, reason),
            ActorError::RuntimeFailure { actor_name, reason } => 
                write!(f, "Actor '{}' runtime failure: {}", actor_name, reason),
            ActorError::MessageHandlingFailed { message_type, reason } => 
                write!(f, "Failed to handle '{}' message: {}", message_type, reason),
            ActorError::SupervisionFailed { supervisor, supervised, reason } => 
                write!(f, "Supervisor '{}' failed to supervise '{}': {}", supervisor, supervised, reason),
            ActorError::MailboxError { actor_name, reason } => 
                write!(f, "Mailbox error for actor '{}': {}", actor_name, reason),
        }
    }
}

impl fmt::Display for GPUError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GPUError::DeviceInitializationFailed(reason) => 
                write!(f, "GPU device initialization failed: {}", reason),
            GPUError::MemoryAllocationFailed { requested_bytes, reason } => 
                write!(f, "GPU memory allocation failed ({} bytes): {}", requested_bytes, reason),
            GPUError::KernelExecutionFailed { kernel_name, reason } => 
                write!(f, "GPU kernel '{}' execution failed: {}", kernel_name, reason),
            GPUError::DataTransferFailed { direction, reason } => 
                write!(f, "GPU data transfer failed ({:?}): {}", direction, reason),
            GPUError::FallbackToCPU { reason } => 
                write!(f, "Falling back to CPU computation: {}", reason),
            GPUError::DriverError(reason) => 
                write!(f, "GPU driver error: {}", reason),
        }
    }
}

impl fmt::Display for SettingsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SettingsError::FileNotFound(path) => 
                write!(f, "Settings file not found: {}", path),
            SettingsError::ParseError { file_path, reason } => 
                write!(f, "Failed to parse settings file '{}': {}", file_path, reason),
            SettingsError::ValidationFailed { setting_path, reason } => 
                write!(f, "Settings validation failed for '{}': {}", setting_path, reason),
            SettingsError::SaveFailed { file_path, reason } => 
                write!(f, "Failed to save settings to '{}': {}", file_path, reason),
            SettingsError::CacheError(reason) => 
                write!(f, "Settings cache error: {}", reason),
        }
    }
}

impl fmt::Display for NetworkError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            NetworkError::ConnectionFailed { host, port, reason } => 
                write!(f, "Connection to {}:{} failed: {}", host, port, reason),
            NetworkError::WebSocketError(reason) => 
                write!(f, "WebSocket error: {}", reason),
            NetworkError::MCPError { method, reason } => 
                write!(f, "MCP method '{}' failed: {}", method, reason),
            NetworkError::HTTPError { url, status, reason } => 
                write!(f, "HTTP error for '{}' (status: {:?}): {}", url, status, reason),
            NetworkError::Timeout { operation, timeout_ms } => 
                write!(f, "Timeout after {}ms for operation: {}", timeout_ms, operation),
        }
    }
}

impl std::error::Error for VisionFlowError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            VisionFlowError::IO(e) => Some(e),
            VisionFlowError::Generic { source: Some(source), .. } => Some(&**source),
            _ => None,
        }
    }
}

impl std::error::Error for ActorError {}
impl std::error::Error for GPUError {}
impl std::error::Error for SettingsError {}
impl std::error::Error for NetworkError {}

impl From<std::io::Error> for VisionFlowError {
    fn from(e: std::io::Error) -> Self {
        VisionFlowError::IO(std::sync::Arc::new(e))
    }
}

impl From<ActorError> for VisionFlowError {
    fn from(e: ActorError) -> Self {
        VisionFlowError::Actor(e)
    }
}

impl From<GPUError> for VisionFlowError {
    fn from(e: GPUError) -> Self {
        VisionFlowError::GPU(e)
    }
}

impl From<SettingsError> for VisionFlowError {
    fn from(e: SettingsError) -> Self {
        VisionFlowError::Settings(e)
    }
}

impl From<NetworkError> for VisionFlowError {
    fn from(e: NetworkError) -> Self {
        VisionFlowError::Network(e)
    }
}

// Convenience type alias for Results
pub type VisionFlowResult<T> = Result<T, VisionFlowError>;

/// Trait for providing error context
pub trait ErrorContext<T> {
    fn with_context<F>(self, f: F) -> VisionFlowResult<T>
    where
        F: FnOnce() -> String;
    
    fn with_actor_context(self, actor_name: &str) -> VisionFlowResult<T>;
    
    fn with_gpu_context(self, operation: &str) -> VisionFlowResult<T>;
}

impl<T, E> ErrorContext<T> for Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn with_context<F>(self, f: F) -> VisionFlowResult<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| VisionFlowError::Generic {
            message: f(),
            source: Some(std::sync::Arc::new(e)),
        })
    }
    
    fn with_actor_context(self, actor_name: &str) -> VisionFlowResult<T> {
        self.map_err(|e| VisionFlowError::Actor(ActorError::RuntimeFailure {
            actor_name: actor_name.to_string(),
            reason: e.to_string(),
        }))
    }
    
    fn with_gpu_context(self, operation: &str) -> VisionFlowResult<T> {
        self.map_err(|e| VisionFlowError::GPU(GPUError::KernelExecutionFailed {
            kernel_name: operation.to_string(),
            reason: e.to_string(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let actor_error = VisionFlowError::Actor(ActorError::StartupFailed {
            actor_name: "TestActor".to_string(),
            reason: "Init failed".to_string(),
        });
        
        assert!(actor_error.to_string().contains("TestActor"));
        assert!(actor_error.to_string().contains("Init failed"));
    }

    #[test]
    fn test_error_context() {
        let result: Result<(), std::io::Error> = Err(std::io::Error::new(
            std::io::ErrorKind::NotFound, 
            "File not found"
        ));
        
        let with_context = result.with_context(|| "Failed to read configuration".to_string());
        assert!(with_context.is_err());
        
        if let Err(VisionFlowError::Generic { message, .. }) = with_context {
            assert_eq!(message, "Failed to read configuration");
        } else {
            panic!("Expected Generic error with context");
        }
    }
}