//! Input validation utilities

use crate::errors::{AutoSchemaError, Result};
use regex::Regex;
use std::sync::OnceLock;

static EMAIL_REGEX: OnceLock<Regex> = OnceLock::new();
static URL_REGEX: OnceLock<Regex> = OnceLock::new();

/// Validate email address format
pub fn validate_email(email: &str) -> Result<()> {
    let email_regex = EMAIL_REGEX.get_or_init(|| {
        Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
            .expect("Invalid email regex")
    });

    if email_regex.is_match(email) {
        Ok(())
    } else {
        Err(AutoSchemaError::validation("email", "Invalid email format"))
    }
}

/// Validate URL format
pub fn validate_url(url: &str) -> Result<()> {
    let url_regex = URL_REGEX.get_or_init(|| {
        Regex::new(r"^https?://[^\s/$.?#].[^\s]*$")
            .expect("Invalid URL regex")
    });

    if url_regex.is_match(url) {
        Ok(())
    } else {
        Err(AutoSchemaError::validation("url", "Invalid URL format"))
    }
}

/// Validate string is not empty and within length limits
pub fn validate_string(
    value: &str,
    field_name: &str,
    min_length: Option<usize>,
    max_length: Option<usize>,
) -> Result<()> {
    if value.trim().is_empty() {
        return Err(AutoSchemaError::validation(field_name, "Cannot be empty"));
    }

    let len = value.len();

    if let Some(min) = min_length {
        if len < min {
            return Err(AutoSchemaError::validation(
                field_name,
                format!("Must be at least {} characters", min),
            ));
        }
    }

    if let Some(max) = max_length {
        if len > max {
            return Err(AutoSchemaError::validation(
                field_name,
                format!("Must be no more than {} characters", max),
            ));
        }
    }

    Ok(())
}

/// Validate numeric value is within range
pub fn validate_range<T>(value: T, field_name: &str, min: Option<T>, max: Option<T>) -> Result<()>
where
    T: PartialOrd + std::fmt::Display + Copy,
{
    if let Some(min_val) = min {
        if value < min_val {
            return Err(AutoSchemaError::validation(
                field_name,
                format!("Must be at least {}", min_val),
            ));
        }
    }

    if let Some(max_val) = max {
        if value > max_val {
            return Err(AutoSchemaError::validation(
                field_name,
                format!("Must be no more than {}", max_val),
            ));
        }
    }

    Ok(())
}

/// Validate that value is one of allowed options
pub fn validate_enum<T>(value: &T, field_name: &str, allowed: &[T]) -> Result<()>
where
    T: PartialEq + std::fmt::Display,
{
    if allowed.contains(value) {
        Ok(())
    } else {
        let allowed_str: Vec<String> = allowed.iter().map(|v| v.to_string()).collect();
        Err(AutoSchemaError::validation(
            field_name,
            format!("Must be one of: {}", allowed_str.join(", ")),
        ))
    }
}

/// Validate UUID format
pub fn validate_uuid(uuid_str: &str) -> Result<uuid::Uuid> {
    uuid::Uuid::parse_str(uuid_str)
        .map_err(|_| AutoSchemaError::validation("uuid", "Invalid UUID format"))
}

/// Validate file path exists and is readable
pub fn validate_file_path(path: &str) -> Result<()> {
    let path = std::path::Path::new(path);

    if !path.exists() {
        return Err(AutoSchemaError::validation("file_path", "File does not exist"));
    }

    if !path.is_file() {
        return Err(AutoSchemaError::validation("file_path", "Path is not a file"));
    }

    // Check if file is readable
    match std::fs::File::open(path) {
        Ok(_) => Ok(()),
        Err(_) => Err(AutoSchemaError::validation("file_path", "File is not readable")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_email() {
        assert!(validate_email("test@example.com").is_ok());
        assert!(validate_email("invalid-email").is_err());
    }

    #[test]
    fn test_validate_url() {
        assert!(validate_url("https://example.com").is_ok());
        assert!(validate_url("invalid-url").is_err());
    }

    #[test]
    fn test_validate_string() {
        assert!(validate_string("hello", "test", Some(3), Some(10)).is_ok());
        assert!(validate_string("", "test", Some(1), None).is_err());
        assert!(validate_string("toolong", "test", None, Some(5)).is_err());
    }

    #[test]
    fn test_validate_range() {
        assert!(validate_range(5, "test", Some(1), Some(10)).is_ok());
        assert!(validate_range(0, "test", Some(1), Some(10)).is_err());
        assert!(validate_range(15, "test", Some(1), Some(10)).is_err());
    }

    #[test]
    fn test_validate_enum() {
        let allowed = vec!["option1", "option2", "option3"];
        assert!(validate_enum(&"option1", "test", &allowed).is_ok());
        assert!(validate_enum(&"invalid", "test", &allowed).is_err());
    }

    #[test]
    fn test_validate_uuid() {
        let valid_uuid = "550e8400-e29b-41d4-a716-446655440000";
        assert!(validate_uuid(valid_uuid).is_ok());
        assert!(validate_uuid("invalid-uuid").is_err());
    }
}