//! Logging utilities for AutoSchemaKG

use crate::config::LoggingConfig;
use crate::errors::Result;
use log::LevelFilter;

/// Initialize logging with the given configuration
pub fn setup_logging(config: &LoggingConfig) -> Result<()> {
    let level = parse_log_level(&config.level)?;

    env_logger::Builder::new()
        .filter_level(level)
        .format_timestamp_secs()
        .init();

    log::info!("Logging initialized with level: {}", config.level);
    Ok(())
}

fn parse_log_level(level: &str) -> Result<LevelFilter> {
    match level.to_lowercase().as_str() {
        "off" => Ok(LevelFilter::Off),
        "error" => Ok(LevelFilter::Error),
        "warn" => Ok(LevelFilter::Warn),
        "info" => Ok(LevelFilter::Info),
        "debug" => Ok(LevelFilter::Debug),
        "trace" => Ok(LevelFilter::Trace),
        _ => Err(crate::errors::AutoSchemaError::configuration(
            format!("Invalid log level: {}", level)
        )),
    }
}