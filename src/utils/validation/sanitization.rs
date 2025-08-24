use super::{ValidationResult, ValidationError, MAX_STRING_LENGTH};
use super::errors::DetailedValidationError;
use serde_json::Value;
use std::collections::HashMap;
use regex::Regex;

/// Input sanitization utilities to prevent XSS, injection, and other attacks
pub struct Sanitizer;

impl Sanitizer {
    /// Sanitize a JSON value recursively
    pub fn sanitize_json(value: &mut Value) -> ValidationResult<()> {
        match value {
            Value::String(s) => {
                *s = Self::sanitize_string(s)?;
            }
            Value::Array(arr) => {
                for item in arr.iter_mut() {
                    Self::sanitize_json(item)?;
                }
            }
            Value::Object(obj) => {
                for (key, val) in obj.iter_mut() {
                    // Sanitize both keys and values
                    if Self::is_suspicious_key(key) {
                        return Err(ValidationError::malicious_content(key).into());
                    }
                    Self::sanitize_json(val)?;
                }
            }
            _ => {} // Numbers, booleans, null don't need sanitization
        }
        Ok(())
    }

    /// Sanitize a string by removing/escaping dangerous content
    pub fn sanitize_string(input: &str) -> ValidationResult<String> {
        // Check length first
        if input.len() > MAX_STRING_LENGTH {
            return Err(ValidationError::too_long("string", MAX_STRING_LENGTH).into());
        }

        // Check for null bytes
        if input.contains('\0') {
            return Err(ValidationError::malicious_content("string").into());
        }

        let mut sanitized = input.to_string();

        // Remove or escape potentially dangerous patterns
        sanitized = Self::remove_script_tags(&sanitized)?;
        sanitized = Self::escape_html(&sanitized);
        sanitized = Self::remove_sql_injection_patterns(&sanitized)?;
        sanitized = Self::remove_path_traversal(&sanitized)?;
        sanitized = Self::limit_unicode_control_chars(&sanitized)?;

        Ok(sanitized)
    }

    /// Remove actual XSS attempts, not legitimate content
    fn remove_script_tags(input: &str) -> ValidationResult<String> {
        // Only block actual XSS attempts with HTML context
        let xss_patterns = [
            // Actual script tags with content
            r"(?i)<script[^>]*>.*?</script>",
            // javascript: protocol in href/src context (with quotes)
            r#"(?i)(href|src)\s*=\s*["']?\s*javascript:"#,
            // vbscript: protocol
            r#"(?i)(href|src)\s*=\s*["']?\s*vbscript:"#,
            // data URIs with HTML content
            r"(?i)data:text/html[,;]",
            // Event handlers in HTML attributes (with quotes)
            r#"(?i)\s(on\w+)\s*=\s*["'][^"']*["']"#,
        ];

        let result = input.to_string();
        
        for pattern in &xss_patterns {
            let regex = Regex::new(pattern)
                .map_err(|_| DetailedValidationError::from(ValidationError::new("string", "Invalid sanitization regex", "REGEX_ERROR")))?;
            
            if regex.is_match(&result) {
                return Err(ValidationError::malicious_content("string").into());
            }
        }

        Ok(result)
    }

    /// Escape HTML special characters
    fn escape_html(input: &str) -> String {
        input
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&#x27;")
    }

    /// Remove SQL injection patterns - only block actual SQL injection attempts
    fn remove_sql_injection_patterns(input: &str) -> ValidationResult<String> {
        // Only check for actual SQL injection patterns, not common words
        // Look for SQL syntax combinations, not individual keywords
        let sql_injection_patterns = [
            // Actual SQL injection with multiple keywords together
            r"(?i)(union\s+select|select\s+.*\s+from\s+|insert\s+into\s+|delete\s+from\s+|drop\s+table\s+|update\s+.*\s+set\s+)",
            // SQL comments that are commonly used in injection
            r"(?i)(;\s*--|\*/\s*;)",
            // Classic OR 1=1 style injections with actual SQL context
            r"(?i)('\s+or\s+\d+\s*=\s*\d+|'\s+and\s+\d+\s*=\s*\d+)",
        ];

        for pattern in &sql_injection_patterns {
            let regex = Regex::new(pattern)
                .map_err(|_| DetailedValidationError::from(ValidationError::new("string", "Invalid SQL regex", "REGEX_ERROR")))?;
            
            if regex.is_match(input) {
                return Err(ValidationError::malicious_content("string").into());
            }
        }

        Ok(input.to_string())
    }

    /// Remove path traversal attempts
    fn remove_path_traversal(input: &str) -> ValidationResult<String> {
        let traversal_patterns = [
            r"\.\./",
            r"\.\.\\",
            r"%2e%2e%2f",
            r"%2e%2e%5c",
            r"..%2f",
            r"..%5c",
        ];

        for pattern in &traversal_patterns {
            let regex = Regex::new(&format!("(?i){}", pattern))
                .map_err(|_| DetailedValidationError::from(ValidationError::new("string", "Invalid path regex", "REGEX_ERROR")))?;
            
            if regex.is_match(input) {
                return Err(ValidationError::malicious_content("string").into());
            }
        }

        Ok(input.to_string())
    }

    /// Limit dangerous Unicode control characters
    fn limit_unicode_control_chars(input: &str) -> ValidationResult<String> {
        let mut result = String::with_capacity(input.len());
        
        for ch in input.chars() {
            match ch {
                // Allow common whitespace
                ' ' | '\t' | '\n' | '\r' => result.push(ch),
                // Block other control characters except basic printable ASCII and common Unicode
                c if c.is_control() && !matches!(c, '\u{0009}' | '\u{000A}' | '\u{000D}') => {
                    return Err(ValidationError::malicious_content("string").into());
                }
                // Allow printable characters
                c => result.push(c),
            }
        }

        Ok(result)
    }

    /// Check if a JSON key is suspicious
    fn is_suspicious_key(key: &str) -> bool {
        // Only check for actual dangerous prototype pollution patterns
        // Don't flag legitimate words that happen to contain "function" or "script"
        let dangerous_exact_keys = [
            "__proto__",
            "constructor", 
            "prototype",
        ];
        
        // Check for exact matches of dangerous keys
        if dangerous_exact_keys.iter().any(|&k| key == k) {
            return true;
        }
        
        // Check for obvious code injection attempts (standalone eval/script tags)
        if key == "eval" || key == "<script>" || key.starts_with("<script") {
            return true;
        }
        
        // Allow double underscores only for actual proto pollution attempts
        if key == "__proto__" || key == "__defineGetter__" || key == "__defineSetter__" {
            return true;
        }
        
        false
    }

    /// Sanitize filename for safe file operations
    pub fn sanitize_filename(filename: &str) -> ValidationResult<String> {
        if filename.is_empty() {
            return Err(ValidationError::new("filename", "Filename cannot be empty", "EMPTY_FILENAME").into());
        }

        if filename.len() > 255 {
            return Err(ValidationError::too_long("filename", 255).into());
        }

        // Remove dangerous characters
        let dangerous_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\0'];
        
        if filename.chars().any(|c| dangerous_chars.contains(&c)) {
            return Err(ValidationError::malicious_content("filename").into());
        }

        // Check for reserved names on Windows
        let reserved_names = [
            "CON", "PRN", "AUX", "NUL",
            "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
            "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
        ];

        let name_upper = filename.to_uppercase();
        if reserved_names.iter().any(|&name| name_upper == name || name_upper.starts_with(&format!("{}.", name))) {
            return Err(ValidationError::malicious_content("filename").into());
        }

        // Don't allow files starting with dot on Unix systems (hidden files)
        if filename.starts_with('.') {
            return Err(ValidationError::malicious_content("filename").into());
        }

        Ok(filename.to_string())
    }

    /// Sanitize and validate email addresses
    pub fn sanitize_email(email: &str) -> ValidationResult<String> {
        let sanitized = Self::sanitize_string(email)?;
        
        // Additional email-specific checks
        if sanitized.len() > 254 { // RFC 5321 limit
            return Err(ValidationError::too_long("email", 254).into());
        }

        // Check for multiple @ symbols
        if sanitized.matches('@').count() != 1 {
            return Err(ValidationError::invalid_format("email").into());
        }

        let parts: Vec<&str> = sanitized.split('@').collect();
        if parts.len() != 2 {
            return Err(ValidationError::invalid_format("email").into());
        }

        let (local, domain) = (parts[0], parts[1]);

        // Validate local part
        if local.is_empty() || local.len() > 64 {
            return Err(ValidationError::invalid_format("email").into());
        }

        // Validate domain part
        if domain.is_empty() || domain.len() > 255 {
            return Err(ValidationError::invalid_format("email").into());
        }

        // Check for consecutive dots
        if sanitized.contains("..") {
            return Err(ValidationError::invalid_format("email").into());
        }

        Ok(sanitized)
    }

    /// Sanitize URLs to prevent malicious redirects
    pub fn sanitize_url(url: &str) -> ValidationResult<String> {
        let sanitized = Self::sanitize_string(url)?;

        // Check URL length
        if sanitized.len() > 2048 {
            return Err(ValidationError::too_long("url", 2048).into());
        }

        // Parse URL to validate structure
        let parsed_url = url::Url::parse(&sanitized)
            .map_err(|_| DetailedValidationError::from(ValidationError::invalid_format("url")))?;

        // Only allow safe schemes
        let allowed_schemes = ["http", "https", "ftp", "ftps"];
        if !allowed_schemes.contains(&parsed_url.scheme()) {
            return Err(ValidationError::new(
                "url",
                "Only http, https, ftp, and ftps URLs are allowed",
                "INVALID_SCHEME"
            ).into());
        }

        // Block private/local network addresses
        if let Some(host) = parsed_url.host_str() {
            if Self::is_private_ip_or_localhost(host) {
                return Err(ValidationError::new(
                    "url",
                    "Private IP addresses and localhost are not allowed",
                    "PRIVATE_URL"
                ).into());
            }
        }

        Ok(sanitized)
    }

    /// Check if host is a private IP or localhost
    fn is_private_ip_or_localhost(host: &str) -> bool {
        if host == "localhost" || host == "127.0.0.1" || host == "::1" {
            return true;
        }

        // Check for private IP ranges
        if let Ok(ip) = host.parse::<std::net::IpAddr>() {
            match ip {
                std::net::IpAddr::V4(ipv4) => {
                    let octets = ipv4.octets();
                    // 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
                    octets[0] == 10 
                        || (octets[0] == 172 && (octets[1] >= 16 && octets[1] <= 31))
                        || (octets[0] == 192 && octets[1] == 168)
                        || octets[0] == 127 // loopback
                }
                std::net::IpAddr::V6(_) => {
                    // For simplicity, block all IPv6 private ranges
                    host.starts_with("::1") || host.starts_with("fc") || host.starts_with("fd")
                }
            }
        } else {
            false
        }
    }
}

/// Content Security Policy utilities
pub struct CSPUtils;

impl CSPUtils {
    /// Generate a Content Security Policy header value
    pub fn generate_csp_header() -> String {
        vec![
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'", // Note: Consider tightening this
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: blob:",
            "font-src 'self'",
            "connect-src 'self' ws: wss:",
            "media-src 'self'",
            "object-src 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            "frame-ancestors 'none'",
            "upgrade-insecure-requests",
        ].join("; ")
    }

    /// Generate security headers
    pub fn security_headers() -> HashMap<&'static str, &'static str> {
        let mut headers = HashMap::new();
        
        headers.insert("X-Content-Type-Options", "nosniff");
        headers.insert("X-Frame-Options", "DENY");
        headers.insert("X-XSS-Protection", "1; mode=block");
        headers.insert("Referrer-Policy", "strict-origin-when-cross-origin");
        headers.insert("Permissions-Policy", "geolocation=(), microphone=(), camera=()");
        headers.insert("Cross-Origin-Embedder-Policy", "require-corp");
        headers.insert("Cross-Origin-Opener-Policy", "same-origin");
        
        headers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_string() {
        assert!(Sanitizer::sanitize_string("<script>alert('xss')</script>").is_err());
        assert!(Sanitizer::sanitize_string("javascript:alert(1)").is_err());
        assert!(Sanitizer::sanitize_string("' OR 1=1 --").is_err());
        assert!(Sanitizer::sanitize_string("../../../etc/passwd").is_err());
        
        let safe = Sanitizer::sanitize_string("Hello World!").unwrap();
        assert_eq!(safe, "Hello World!");
    }

    #[test]
    fn test_sanitize_filename() {
        assert!(Sanitizer::sanitize_filename("").is_err());
        assert!(Sanitizer::sanitize_filename("file<>name").is_err());
        assert!(Sanitizer::sanitize_filename("CON").is_err());
        assert!(Sanitizer::sanitize_filename(".hidden").is_err());
        
        let safe = Sanitizer::sanitize_filename("document.pdf").unwrap();
        assert_eq!(safe, "document.pdf");
    }

    #[test]
    fn test_sanitize_url() {
        assert!(Sanitizer::sanitize_url("javascript:alert(1)").is_err());
        assert!(Sanitizer::sanitize_url("http://localhost/api").is_err());
        assert!(Sanitizer::sanitize_url("http://192.168.1.1/").is_err());
        
        let safe = Sanitizer::sanitize_url("https://example.com/api").unwrap();
        assert_eq!(safe, "https://example.com/api");
    }
}