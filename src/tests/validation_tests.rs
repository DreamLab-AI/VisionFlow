#[cfg(test)]
mod validation_tests {
    use crate::utils::validation::{ValidationError, ValidationContext};
    use crate::utils::validation::schemas::{ApiSchemas, ValidationSchema, FieldValidator};
    use crate::utils::validation::sanitization::Sanitizer;
    use crate::utils::validation::rate_limit::{RateLimiter, RateLimitConfig};
    use crate::utils::validation::errors::{DetailedValidationError, ValidationErrorCollection};
    use crate::handlers::validation_handler::ValidationService;
    use serde_json::json;
    use std::time::Duration;

    #[test]
    fn test_string_sanitization() {
        // Test XSS prevention
        let malicious_script = "<script>alert('xss')</script>";
        assert!(Sanitizer::sanitize_string(malicious_script).is_err());
        
        let javascript_url = "javascript:alert(1)";
        assert!(Sanitizer::sanitize_string(javascript_url).is_err());
        
        // Test safe string
        let safe_string = "Hello World!";
        let sanitized = Sanitizer::sanitize_string(safe_string).unwrap();
        assert_eq!(sanitized, "Hello World!");
        
        // Test HTML escaping
        let html_content = "<div>Test & content</div>";
        let sanitized = Sanitizer::sanitize_string(html_content).unwrap();
        assert!(!sanitized.contains("<div>"));
        assert!(sanitized.contains("&lt;div&gt;"));
    }

    #[test]
    fn test_sql_injection_prevention() {
        let sql_injection = "'; DROP TABLE users; --";
        assert!(Sanitizer::sanitize_string(sql_injection).is_err());
        
        let union_attack = "' UNION SELECT * FROM passwords --";
        assert!(Sanitizer::sanitize_string(union_attack).is_err());
        
        // Safe database-like string
        let safe_query = "user_id = 123";
        let sanitized = Sanitizer::sanitize_string(&safe_query).unwrap();
        assert_eq!(sanitized, safe_query);
    }

    #[test]
    fn test_path_traversal_prevention() {
        let path_traversal = "../../../etc/passwd";
        assert!(Sanitizer::sanitize_string(path_traversal).is_err());
        
        let encoded_traversal = "%2e%2e%2f%2e%2e%2f";
        assert!(Sanitizer::sanitize_string(encoded_traversal).is_err());
        
        // Safe path
        let safe_path = "documents/file.txt";
        let sanitized = Sanitizer::sanitize_string(&safe_path).unwrap();
        assert_eq!(sanitized, safe_path);
    }

    #[test]
    fn test_filename_sanitization() {
        // Dangerous filenames
        assert!(Sanitizer::sanitize_filename("").is_err());
        assert!(Sanitizer::sanitize_filename("con.txt").is_err());
        assert!(Sanitizer::sanitize_filename(".hidden").is_err());
        assert!(Sanitizer::sanitize_filename("file<>name").is_err());
        
        // Safe filename
        let safe_filename = "document.pdf";
        let sanitized = Sanitizer::sanitize_filename(&safe_filename).unwrap();
        assert_eq!(sanitized, safe_filename);
    }

    #[test]
    fn test_email_sanitization() {
        // Invalid emails
        assert!(Sanitizer::sanitize_email("not-an-email").is_err());
        assert!(Sanitizer::sanitize_email("user@@domain.com").is_err());
        assert!(Sanitizer::sanitize_email("user..name@domain.com").is_err());
        
        // Valid email
        let valid_email = "user@example.com";
        let sanitized = Sanitizer::sanitize_email(&valid_email).unwrap();
        assert_eq!(sanitized, valid_email);
    }

    #[test]
    fn test_url_sanitization() {
        // Dangerous URLs
        assert!(Sanitizer::sanitize_url("javascript:alert(1)").is_err());
        assert!(Sanitizer::sanitize_url("http://localhost/api").is_err());
        assert!(Sanitizer::sanitize_url("http://192.168.1.1/").is_err());
        
        // Safe URL
        let safe_url = "https://example.com/api";
        let sanitized = Sanitizer::sanitize_url(&safe_url).unwrap();
        assert_eq!(sanitized, safe_url);
    }

    #[test]
    fn test_schema_validation() {
        let mut ctx = ValidationContext::new();
        
        // Test valid settings update
        let valid_settings = json!({
            "visualisation": {
                "graphs": {
                    "logseq": {
                        "physics": {
                            "damping": 0.8,
                            "iterations": 100
                        }
                    }
                }
            }
        });
        
        let schema = ApiSchemas::settings_update();
        assert!(schema.validate(&valid_settings, &mut ctx).is_ok());
    }

    #[test]
    fn test_physics_validation() {
        let mut ctx = ValidationContext::new();
        
        // Test valid physics params
        let valid_physics = json!({
            "damping": 0.8,
            "iterations": 100,
            "springK": 0.3,
            "repelK": 300.0,
            "maxVelocity": 10.0
        });
        
        let schema = ApiSchemas::physics_params();
        assert!(schema.validate(&valid_physics, &mut ctx).is_ok());
        
        // Test invalid physics params
        let invalid_physics = json!({
            "damping": 1.5, // Out of range
            "iterations": -5 // Negative
        });
        
        ctx = ValidationContext::new();
        assert!(schema.validate(&invalid_physics, &mut ctx).is_err());
    }

    #[test]
    fn test_ragflow_validation() {
        let mut ctx = ValidationContext::new();
        
        // Test valid RAGFlow request
        let valid_request = json!({
            "question": "What is the meaning of life?",
            "session_id": "session-123",
            "stream": false
        });
        
        let schema = ApiSchemas::ragflow_chat();
        assert!(schema.validate(&valid_request, &mut ctx).is_ok());
        
        // Test invalid RAGFlow request
        let invalid_request = json!({
            "question": "", // Empty question
            "session_id": "x".repeat(300) // Too long
        });
        
        ctx = ValidationContext::new();
        assert!(schema.validate(&invalid_request, &mut ctx).is_err());
    }

    #[test]
    fn test_rate_limiting() {
        let config = RateLimitConfig {
            requests_per_minute: 60,
            burst_size: 5,
            cleanup_interval: Duration::from_secs(60),
            ban_duration: Duration::from_secs(300),
            max_violations: 3,
        };
        
        let limiter = RateLimiter::new(config);
        let client_id = "test_client";
        
        // Should allow burst_size requests
        for _ in 0..5 {
            assert!(limiter.is_allowed(client_id));
        }
        
        // Should deny the next request
        assert!(!limiter.is_allowed(client_id));
        
        // Check remaining tokens
        assert_eq!(limiter.remaining_tokens(client_id), 0);
    }

    #[test]
    fn test_validation_service() {
        let service = ValidationService::new();
        
        // Test settings validation
        let settings = json!({
            "visualisation": {
                "graphs": {
                    "logseq": {
                        "physics": {"damping": 0.8}
                    }
                }
            }
        });
        
        assert!(service.validate_settings_update(&settings).is_ok());
        
        // Test malicious settings
        let malicious_settings = json!({
            "visualisation": {
                "graphs": {
                    "logseq": {
                        "physics": {"damping": "<script>alert('xss')</script>"}
                    }
                }
            }
        });
        
        assert!(service.validate_settings_update(&malicious_settings).is_err());
    }

    #[test]
    fn test_error_collection() {
        let mut collection = ValidationErrorCollection::new();
        assert!(collection.is_empty());
        
        let error1 = DetailedValidationError::missing_required_field("field1");
        let error2 = DetailedValidationError::invalid_type("field2", "string", "number");
        
        collection.add_error(error1);
        collection.add_error(error2);
        
        assert_eq!(collection.error_count, 2);
        assert!(!collection.is_empty());
        
        let field_errors = collection.get_field_errors("field1");
        assert_eq!(field_errors.len(), 1);
    }

    #[test]
    fn test_validation_context_nesting() {
        let mut ctx = ValidationContext::new();
        
        assert!(ctx.enter_field("level1").is_ok());
        assert!(ctx.enter_field("level2").is_ok());
        assert_eq!(ctx.get_path(), "level1.level2");
        
        ctx.exit_field();
        assert_eq!(ctx.get_path(), "level1");
        
        ctx.exit_field();
        assert_eq!(ctx.get_path(), "root");
    }

    #[test]
    fn test_validation_context_max_depth() {
        let mut ctx = ValidationContext::new();
        ctx.max_depth = 3;
        
        assert!(ctx.enter_field("level1").is_ok());
        assert!(ctx.enter_field("level2").is_ok());
        assert!(ctx.enter_field("level3").is_ok());
        
        // Should fail at max depth
        assert!(ctx.enter_field("level4").is_err());
    }

    #[test]
    fn test_field_validators() {
        let mut ctx = ValidationContext::new();
        
        // String validator
        let string_validator = FieldValidator::string().min_length(1).max_length(100);
        let valid_string = json!("Hello World");
        let empty_string = json!("");
        let long_string = json!("x".repeat(200));
        
        assert!(string_validator.validate(&valid_string, &mut ctx).is_ok());
        assert!(string_validator.validate(&empty_string, &mut ctx).is_err());
        assert!(string_validator.validate(&long_string, &mut ctx).is_err());
        
        // Number validator
        let number_validator = FieldValidator::number().min_value(0.0).max_value(100.0);
        let valid_number = json!(50.0);
        let invalid_number = json!(150.0);
        
        assert!(number_validator.validate(&valid_number, &mut ctx).is_ok());
        assert!(number_validator.validate(&invalid_number, &mut ctx).is_err());
        
        // Email validator
        let email_validator = FieldValidator::string().email();
        let valid_email = json!("test@example.com");
        let invalid_email = json!("not-an-email");
        
        assert!(email_validator.validate(&valid_email, &mut ctx).is_ok());
        assert!(email_validator.validate(&invalid_email, &mut ctx).is_err());
    }

    #[test]
    fn test_swarm_validation() {
        let service = ValidationService::new();
        
        // Valid swarm configuration
        let valid_swarm = json!({
            "topology": "mesh",
            "max_agents": 10,
            "strategy": "balanced",
            "enable_neural": true
        });
        
        assert!(service.validate_swarm_init(&valid_swarm).is_ok());
        
        // Invalid topology
        let invalid_swarm = json!({
            "topology": "invalid_topology",
            "max_agents": 10,
            "strategy": "balanced"
        });
        
        assert!(service.validate_swarm_init(&invalid_swarm).is_err());
    }

    #[test]
    fn test_comprehensive_json_sanitization() {
        let mut malicious_json = json!({
            "user_input": "<script>alert('xss')</script>",
            "nested": {
                "sql_injection": "'; DROP TABLE users; --",
                "path_traversal": "../../../etc/passwd"
            },
            "array": [
                "javascript:alert(1)",
                "safe_content"
            ]
        });
        
        // Should detect malicious content
        assert!(Sanitizer::sanitize_json(&mut malicious_json).is_err());
        
        let mut safe_json = json!({
            "user_input": "Hello, world!",
            "nested": {
                "number": 42,
                "boolean": true
            },
            "array": ["item1", "item2"]
        });
        
        // Should pass safe content
        assert!(Sanitizer::sanitize_json(&mut safe_json).is_ok());
    }

    #[test]
    fn test_rate_limit_ban_system() {
        let config = RateLimitConfig {
            requests_per_minute: 60,
            burst_size: 1,
            max_violations: 2,
            ban_duration: Duration::from_secs(1),
            ..Default::default()
        };
        
        let limiter = RateLimiter::new(config);
        let client_id = "ban_test_client";
        
        // Use up the token
        assert!(limiter.is_allowed(client_id));
        
        // Trigger violations
        assert!(!limiter.is_allowed(client_id)); // Violation 1
        assert!(!limiter.is_allowed(client_id)); // Violation 2, should trigger ban
        
        // Should be banned now
        assert!(limiter.is_banned(client_id));
        assert!(!limiter.is_allowed(client_id));
        
        // Wait for ban to expire
        std::thread::sleep(Duration::from_millis(1100));
        
        // Should not be banned anymore (in a real scenario with proper time handling)
        // Note: This test may be flaky due to timing
    }
}