use log::info;
use std::io;

pub fn init_logging() -> io::Result<()> {
    // Initialize env_logger - it automatically reads the RUST_LOG environment variable
    env_logger::init();
    info!("Logging initialized via env_logger with RUST_LOG={}", 
          std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()));
    Ok(())
}

// Check if debug mode is enabled via environment variable
pub fn is_debug_enabled() -> bool {
    std::env::var("DEBUG_ENABLED")
        .unwrap_or_else(|_| "false".to_string())
        .parse::<bool>()
        .unwrap_or(false)
}