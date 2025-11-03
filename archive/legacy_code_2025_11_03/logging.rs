use log::info;
use std::io;

pub fn init_logging() -> io::Result<()> {
    
    env_logger::init();
    info!(
        "Logging initialized via env_logger with RUST_LOG={}",
        std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string())
    );
    Ok(())
}

// Check if debug mode is enabled
// First checks environment variable (for override), then settings.yaml
pub fn is_debug_enabled() -> bool {
    
    if let Ok(val) = std::env::var("DEBUG_ENABLED") {
        return val.parse::<bool>().unwrap_or(false);
    }

    
    
    if let Ok(settings) = crate::config::AppFullSettings::new() {
        return settings.system.debug.enabled;
    }

    false
}
