//! Comprehensive error handling for the VisionFlow system
//!
//! This module provides a unified error handling approach to replace
//! all panic! and unwrap() calls with proper error propagation.

use serde::ser::{Serialize, Serializer};
use std::fmt;

/
fn serialize_io_error<S>(
    error: &std::sync::Arc<std::io::Error>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    error.to_string().serialize(serializer)
}

/
#[derive(Debug, Clone, serde::Serialize)]
pub enum VisionFlowError {
    
    Actor(ActorError),
    
    GPU(GPUError),
    
    Settings(SettingsError),
    
    Network(NetworkError),
    
    #[serde(serialize_with = "serialize_io_error")]
    IO(std::sync::Arc<std::io::Error>),
    
    Serialization(String),
    
    Speech(SpeechError),
    
    GitHub(GitHubError),
    
    Audio(AudioError),
    
    Resource(ResourceError),
    
    Performance(PerformanceError),
    
    Protocol(ProtocolError),
    
    Generic {
        message: String,
        #[serde(skip)]
        source: Option<std::sync::Arc<dyn std::error::Error + Send + Sync + 'static>>,
    },
}

/
#[derive(Debug, Clone, serde::Serialize)]
pub enum ActorError {
    
    StartupFailed { actor_name: String, reason: String },
    
    RuntimeFailure { actor_name: String, reason: String },
    
    MessageHandlingFailed {
        message_type: String,
        reason: String,
    },
    
    SupervisionFailed {
        supervisor: String,
        supervised: String,
        reason: String,
    },
    
    MailboxError { actor_name: String, reason: String },
    
    ActorNotAvailable(String),
}

/
#[derive(Debug, Clone, serde::Serialize)]
pub enum GPUError {
    
    DeviceInitializationFailed(String),
    
    MemoryAllocationFailed {
        requested_bytes: usize,
        reason: String,
    },
    
    KernelExecutionFailed { kernel_name: String, reason: String },
    
    DataTransferFailed {
        direction: DataTransferDirection,
        reason: String,
    },
    
    FallbackToCPU { reason: String },
    
    DriverError(String),
}

#[derive(Debug, Clone, serde::Serialize)]
pub enum DataTransferDirection {
    CPUToGPU,
    GPUToCPU,
}

/
#[derive(Debug, Clone, serde::Serialize)]
pub enum SettingsError {
    
    FileNotFound(String),
    
    ParseError { file_path: String, reason: String },
    
    ValidationFailed {
        setting_path: String,
        reason: String,
    },
    
    SaveFailed { file_path: String, reason: String },
    
    CacheError(String),
}

/
#[derive(Debug, Clone, serde::Serialize)]
pub enum NetworkError {
    
    ConnectionFailed {
        host: String,
        port: u16,
        reason: String,
    },
    
    WebSocketError(String),
    
    MCPError { method: String, reason: String },
    
    HTTPError {
        url: String,
        status: Option<u16>,
        reason: String,
    },
    
    RequestFailed { url: String, reason: String },
    
    Timeout { operation: String, timeout_ms: u64 },
}

/
#[derive(Debug, Clone, serde::Serialize)]
pub enum SpeechError {
    
    InitializationFailed(String),
    
    TTSFailed { text: String, reason: String },
    
    STTFailed { reason: String },
    
    AudioProcessingFailed { reason: String },
    
    ProviderConfigError { provider: String, reason: String },
}

/
#[derive(Debug, Clone, serde::Serialize)]
pub enum GitHubError {
    
    APIRequestFailed {
        url: String,
        status: Option<u16>,
        reason: String,
    },
    
    AuthenticationFailed(String),
    
    FileOperationFailed {
        path: String,
        operation: String,
        reason: String,
    },
    
    BranchOperationFailed { branch: String, reason: String },
    
    PullRequestFailed { reason: String },
}

/
#[derive(Debug, Clone, serde::Serialize)]
pub enum AudioError {
    
    FormatValidationFailed { format: String, reason: String },
    
    WAVHeaderValidationFailed(String),
    
    DataProcessingFailed(String),
    
    JSONProcessingFailed(String),
}

/
#[derive(Debug, Clone, serde::Serialize)]
pub enum ResourceError {
    
    MonitoringFailed(String),
    
    AvailabilityCheckFailed(String),
    
    FileDescriptorLimit { current: usize, limit: usize },
    
    MemoryLimit { current: u64, limit: u64 },
    
    ProcessLimit { current: usize, limit: usize },
}

/
#[derive(Debug, Clone, serde::Serialize)]
pub enum PerformanceError {
    
    BenchmarkFailed {
        benchmark_name: String,
        reason: String,
    },
    
    ReportGenerationFailed(String),
    
    MetricCollectionFailed { metric: String, reason: String },
    
    ComparisonFailed(String),
}

/
#[derive(Debug, Clone, serde::Serialize)]
pub enum ProtocolError {
    
    EncodingFailed { data_type: String, reason: String },
    
    DecodingFailed { data_type: String, reason: String },
    
    ValidationFailed(String),
    
    BinaryFormatError(String),
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
            VisionFlowError::Speech(e) => write!(f, "Speech Error: {}", e),
            VisionFlowError::GitHub(e) => write!(f, "GitHub Error: {}", e),
            VisionFlowError::Audio(e) => write!(f, "Audio Error: {}", e),
            VisionFlowError::Resource(e) => write!(f, "Resource Error: {}", e),
            VisionFlowError::Performance(e) => write!(f, "Performance Error: {}", e),
            VisionFlowError::Protocol(e) => write!(f, "Protocol Error: {}", e),
            VisionFlowError::Generic { message, .. } => write!(f, "Error: {}", message),
        }
    }
}

impl fmt::Display for ActorError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ActorError::StartupFailed { actor_name, reason } => {
                write!(f, "Actor '{}' failed to start: {}", actor_name, reason)
            }
            ActorError::RuntimeFailure { actor_name, reason } => {
                write!(f, "Actor '{}' runtime failure: {}", actor_name, reason)
            }
            ActorError::MessageHandlingFailed {
                message_type,
                reason,
            } => write!(f, "Failed to handle '{}' message: {}", message_type, reason),
            ActorError::SupervisionFailed {
                supervisor,
                supervised,
                reason,
            } => write!(
                f,
                "Supervisor '{}' failed to supervise '{}': {}",
                supervisor, supervised, reason
            ),
            ActorError::MailboxError { actor_name, reason } => {
                write!(f, "Mailbox error for actor '{}': {}", actor_name, reason)
            }
            ActorError::ActorNotAvailable(actor_name) => {
                write!(f, "Actor '{}' is not available", actor_name)
            }
        }
    }
}

impl fmt::Display for GPUError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GPUError::DeviceInitializationFailed(reason) => {
                write!(f, "GPU device initialization failed: {}", reason)
            }
            GPUError::MemoryAllocationFailed {
                requested_bytes,
                reason,
            } => write!(
                f,
                "GPU memory allocation failed ({} bytes): {}",
                requested_bytes, reason
            ),
            GPUError::KernelExecutionFailed {
                kernel_name,
                reason,
            } => write!(
                f,
                "GPU kernel '{}' execution failed: {}",
                kernel_name, reason
            ),
            GPUError::DataTransferFailed { direction, reason } => {
                write!(f, "GPU data transfer failed ({:?}): {}", direction, reason)
            }
            GPUError::FallbackToCPU { reason } => {
                write!(f, "Falling back to CPU computation: {}", reason)
            }
            GPUError::DriverError(reason) => write!(f, "GPU driver error: {}", reason),
        }
    }
}

impl fmt::Display for SettingsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SettingsError::FileNotFound(path) => write!(f, "Settings file not found: {}", path),
            SettingsError::ParseError { file_path, reason } => write!(
                f,
                "Failed to parse settings file '{}': {}",
                file_path, reason
            ),
            SettingsError::ValidationFailed {
                setting_path,
                reason,
            } => write!(
                f,
                "Settings validation failed for '{}': {}",
                setting_path, reason
            ),
            SettingsError::SaveFailed { file_path, reason } => {
                write!(f, "Failed to save settings to '{}': {}", file_path, reason)
            }
            SettingsError::CacheError(reason) => write!(f, "Settings cache error: {}", reason),
        }
    }
}

impl fmt::Display for NetworkError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            NetworkError::ConnectionFailed { host, port, reason } => {
                write!(f, "Connection to {}:{} failed: {}", host, port, reason)
            }
            NetworkError::WebSocketError(reason) => write!(f, "WebSocket error: {}", reason),
            NetworkError::MCPError { method, reason } => {
                write!(f, "MCP method '{}' failed: {}", method, reason)
            }
            NetworkError::HTTPError {
                url,
                status,
                reason,
            } => write!(
                f,
                "HTTP error for '{}' (status: {:?}): {}",
                url, status, reason
            ),
            NetworkError::Timeout {
                operation,
                timeout_ms,
            } => write!(
                f,
                "Timeout after {}ms for operation: {}",
                timeout_ms, operation
            ),
            NetworkError::RequestFailed { url, reason } => {
                write!(f, "Request to '{}' failed: {}", url, reason)
            }
        }
    }
}

impl fmt::Display for SpeechError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SpeechError::InitializationFailed(reason) => {
                write!(f, "Speech service initialization failed: {}", reason)
            }
            SpeechError::TTSFailed { text, reason } => {
                write!(f, "Text-to-speech failed for '{}': {}", text, reason)
            }
            SpeechError::STTFailed { reason } => write!(f, "Speech-to-text failed: {}", reason),
            SpeechError::AudioProcessingFailed { reason } => {
                write!(f, "Audio processing failed: {}", reason)
            }
            SpeechError::ProviderConfigError { provider, reason } => write!(
                f,
                "Speech provider '{}' configuration error: {}",
                provider, reason
            ),
        }
    }
}

impl fmt::Display for GitHubError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GitHubError::APIRequestFailed {
                url,
                status,
                reason,
            } => write!(
                f,
                "GitHub API request to '{}' failed (status: {:?}): {}",
                url, status, reason
            ),
            GitHubError::AuthenticationFailed(reason) => {
                write!(f, "GitHub authentication failed: {}", reason)
            }
            GitHubError::FileOperationFailed {
                path,
                operation,
                reason,
            } => write!(
                f,
                "GitHub file operation '{}' on '{}' failed: {}",
                operation, path, reason
            ),
            GitHubError::BranchOperationFailed { branch, reason } => write!(
                f,
                "GitHub branch operation on '{}' failed: {}",
                branch, reason
            ),
            GitHubError::PullRequestFailed { reason } => {
                write!(f, "GitHub pull request failed: {}", reason)
            }
        }
    }
}

impl fmt::Display for AudioError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AudioError::FormatValidationFailed { format, reason } => {
                write!(f, "Audio format '{}' validation failed: {}", format, reason)
            }
            AudioError::WAVHeaderValidationFailed(reason) => {
                write!(f, "WAV header validation failed: {}", reason)
            }
            AudioError::DataProcessingFailed(reason) => {
                write!(f, "Audio data processing failed: {}", reason)
            }
            AudioError::JSONProcessingFailed(reason) => {
                write!(f, "Audio JSON processing failed: {}", reason)
            }
        }
    }
}

impl fmt::Display for ResourceError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ResourceError::MonitoringFailed(reason) => {
                write!(f, "Resource monitoring failed: {}", reason)
            }
            ResourceError::AvailabilityCheckFailed(reason) => {
                write!(f, "Resource availability check failed: {}", reason)
            }
            ResourceError::FileDescriptorLimit { current, limit } => {
                write!(f, "File descriptor limit reached: {}/{}", current, limit)
            }
            ResourceError::MemoryLimit { current, limit } => {
                write!(f, "Memory limit reached: {} bytes/{} bytes", current, limit)
            }
            ResourceError::ProcessLimit { current, limit } => {
                write!(f, "Process limit reached: {}/{}", current, limit)
            }
        }
    }
}

impl fmt::Display for PerformanceError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PerformanceError::BenchmarkFailed {
                benchmark_name,
                reason,
            } => write!(f, "Benchmark '{}' failed: {}", benchmark_name, reason),
            PerformanceError::ReportGenerationFailed(reason) => {
                write!(f, "Performance report generation failed: {}", reason)
            }
            PerformanceError::MetricCollectionFailed { metric, reason } => write!(
                f,
                "Performance metric '{}' collection failed: {}",
                metric, reason
            ),
            PerformanceError::ComparisonFailed(reason) => {
                write!(f, "Performance comparison failed: {}", reason)
            }
        }
    }
}

impl fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ProtocolError::EncodingFailed { data_type, reason } => write!(
                f,
                "Protocol encoding failed for '{}': {}",
                data_type, reason
            ),
            ProtocolError::DecodingFailed { data_type, reason } => write!(
                f,
                "Protocol decoding failed for '{}': {}",
                data_type, reason
            ),
            ProtocolError::ValidationFailed(reason) => {
                write!(f, "Protocol validation failed: {}", reason)
            }
            ProtocolError::BinaryFormatError(reason) => {
                write!(f, "Binary format error: {}", reason)
            }
        }
    }
}

impl std::error::Error for VisionFlowError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            VisionFlowError::IO(e) => Some(e),
            VisionFlowError::Generic {
                source: Some(source),
                ..
            } => Some(&**source),
            _ => None,
        }
    }
}

impl std::error::Error for ActorError {}
impl std::error::Error for GPUError {}
impl std::error::Error for SettingsError {}
impl std::error::Error for NetworkError {}
impl std::error::Error for SpeechError {}
impl std::error::Error for GitHubError {}
impl std::error::Error for AudioError {}
impl std::error::Error for ResourceError {}
impl std::error::Error for PerformanceError {}
impl std::error::Error for ProtocolError {}

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

impl From<SpeechError> for VisionFlowError {
    fn from(e: SpeechError) -> Self {
        VisionFlowError::Speech(e)
    }
}

impl From<GitHubError> for VisionFlowError {
    fn from(e: GitHubError) -> Self {
        VisionFlowError::GitHub(e)
    }
}

impl From<AudioError> for VisionFlowError {
    fn from(e: AudioError) -> Self {
        VisionFlowError::Audio(e)
    }
}

impl From<ResourceError> for VisionFlowError {
    fn from(e: ResourceError) -> Self {
        VisionFlowError::Resource(e)
    }
}

impl From<PerformanceError> for VisionFlowError {
    fn from(e: PerformanceError) -> Self {
        VisionFlowError::Performance(e)
    }
}

impl From<ProtocolError> for VisionFlowError {
    fn from(e: ProtocolError) -> Self {
        VisionFlowError::Protocol(e)
    }
}

impl From<reqwest::Error> for VisionFlowError {
    fn from(e: reqwest::Error) -> Self {
        VisionFlowError::Network(NetworkError::RequestFailed {
            url: e.url().map(|u| u.to_string()).unwrap_or_default(),
            reason: e.to_string(),
        })
    }
}

impl From<String> for VisionFlowError {
    fn from(s: String) -> Self {
        VisionFlowError::Generic {
            message: s,
            source: None,
        }
    }
}

impl From<&str> for VisionFlowError {
    fn from(s: &str) -> Self {
        VisionFlowError::Generic {
            message: s.to_string(),
            source: None,
        }
    }
}

// Convenience type alias for Results
pub type VisionFlowResult<T> = Result<T, VisionFlowError>;

/
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
        self.map_err(|e| {
            VisionFlowError::Actor(ActorError::RuntimeFailure {
                actor_name: actor_name.to_string(),
                reason: e.to_string(),
            })
        })
    }

    fn with_gpu_context(self, operation: &str) -> VisionFlowResult<T> {
        self.map_err(|e| {
            VisionFlowError::GPU(GPUError::KernelExecutionFailed {
                kernel_name: operation.to_string(),
                reason: e.to_string(),
            })
        })
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
            "File not found",
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
