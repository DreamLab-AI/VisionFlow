use thiserror::Error;

pub type Result<T> = std::result::Result<T, LLMError>;

#[derive(Error, Debug)]
pub enum LLMError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Rate limit exceeded")]
    RateLimit,

    #[error("Token limit exceeded: {used}/{limit}")]
    TokenLimit { used: usize, limit: usize },

    #[error("Invalid API key")]
    InvalidApiKey,

    #[error("Invalid response format: {0}")]
    InvalidResponse(String),

    #[error("Provider error: {0}")]
    Provider(String),

    #[error("Timeout error")]
    Timeout,

    #[error("Retry exhausted after {attempts} attempts")]
    RetryExhausted { attempts: usize },

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("URL parse error: {0}")]
    UrlParse(#[from] url::ParseError),
}