# Validation Handler Instructions

## File: `src/handlers/validation_handler.rs`

### Purpose
Comprehensive validation service for all API endpoints, providing schema validation, input sanitization, and custom business logic validation. Ensures data integrity and security across the entire application.

### Key Components

#### 1. Validation Service Structure
```rust
pub struct ValidationService {
    settings_schema: ValidationSchema,     // Settings update validation
    physics_schema: ValidationSchema,      // Physics parameters validation
    ragflow_schema: ValidationSchema,      // RAGFlow chat validation
    bots_schema: ValidationSchema,         // Bots data validation
    swarm_schema: ValidationSchema,        // Swarm initialization validation
}
```

#### 2. Validation Flow Pattern
```rust
pub fn validate_<entity>(&self, payload: &Value) -> ValidationResult<Value> {
    let mut ctx = ValidationContext::new();
    let mut sanitized_payload = payload.clone();
    
    // 1. Sanitize input data
    Sanitizer::sanitize_json(&mut sanitized_payload)?;
    
    // 2. Schema validation
    self.<entity>_schema.validate(&sanitized_payload, &mut ctx)?;
    
    // 3. Custom business logic validation (optional)
    self.validate_<entity>_custom(&sanitized_payload)?;
    
    // 4. Return sanitized and validated data
    Ok(sanitized_payload)
}
```

### Implementation Instructions

#### Service Initialization
```rust
impl ValidationService {
    pub fn new() -> Self {
        Self {
            settings_schema: ApiSchemas::settings_update(),
            physics_schema: ApiSchemas::physics_params(),
            ragflow_schema: ApiSchemas::ragflow_chat(),
            bots_schema: ApiSchemas::bots_data(),
            swarm_schema: ApiSchemas::swarm_init(),
        }
    }
}
```

#### Validation Methods Implementation

##### Settings Validation
**Purpose**: Validate configuration updates and system settings

```rust
pub fn validate_settings_update(&self, payload: &Value) -> ValidationResult<Value> {
    // Standard validation flow with custom settings validation
    self.validate_settings_custom(&sanitized_payload)?;
}

fn validate_settings_custom(&self, payload: &Value) -> ValidationResult<()> {
    // Custom validation rules:
    // 1. Check value ranges (e.g., opacity: 0.0-1.0)
    // 2. Validate color format (hex, rgb, hsl)
    // 3. Verify feature compatibility
    // 4. Check performance impact settings
    // 5. Validate environment-specific settings
}
```

##### Physics Parameters Validation
**Purpose**: Validate physics simulation parameters and constraints

```rust
pub fn validate_physics_params(&self, payload: &Value) -> ValidationResult<Value> {
    // Physics-specific validation
    self.validate_physics_custom(&sanitized_payload)?;
}

fn validate_physics_custom(&self, payload: &Value) -> ValidationResult<()> {
    // Custom physics validation:
    // 1. Verify physics constants within realistic bounds
    // 2. Check for conflicting parameter combinations
    // 3. Validate simulation stability constraints
    // 4. Ensure performance-safe parameter values
    // 5. Check dimensional consistency
}
```

##### RAGFlow Chat Validation
**Purpose**: Validate chat messages and RAGFlow integration parameters

```rust
pub fn validate_ragflow_chat(&self, payload: &Value) -> ValidationResult<Value> {
    // Chat-specific sanitization is critical for security
    // No custom validation needed - schema validation sufficient
}

// Schema should validate:
// 1. Message length limits
// 2. Content type restrictions
// 3. User identification format
// 4. Session management parameters
// 5. Rate limiting compliance
```

##### Bots Data Validation
**Purpose**: Validate agent/bot configuration and status updates

```rust
pub fn validate_bots_data(&self, payload: &Value) -> ValidationResult<Value> {
    // Bot data validation - schema validation sufficient
    // Custom validation can be added for:
    // 1. Agent type compatibility
    // 2. Resource allocation limits  
    // 3. Swarm membership constraints
    // 4. Performance thresholds
}
```

##### Swarm Initialization Validation
**Purpose**: Validate swarm configuration and initialization parameters

```rust
pub fn validate_swarm_init(&self, payload: &Value) -> ValidationResult<Value> {
    // Schema validation handles:
    // 1. Topology type validation
    // 2. Agent count limits
    // 3. Configuration parameter format
    // 4. Resource allocation constraints
    // 5. Networking requirements
}
```

### Schema Definition Requirements

#### Settings Update Schema
```json
{
    "type": "object",
    "properties": {
        "visualisation": {
            "type": "object",
            "properties": {
                "rendering": {
                    "properties": {
                        "ambientLightIntensity": {"type": "number", "minimum": 0, "maximum": 2},
                        "backgroundColor": {"type": "string", "pattern": "^#[0-9A-Fa-f]{6}$"},
                        "enableShadows": {"type": "boolean"}
                    }
                }
            }
        },
        "system": {"type": "object"},
        "xr": {"type": "object"}
    }
}
```

#### Physics Parameters Schema
```json
{
    "type": "object",
    "properties": {
        "gravity": {"type": "number", "minimum": -50, "maximum": 50},
        "damping": {"type": "number", "minimum": 0, "maximum": 1},
        "timeStep": {"type": "number", "minimum": 0.001, "maximum": 0.1},
        "maxIterations": {"type": "integer", "minimum": 1, "maximum": 100}
    },
    "required": ["gravity", "damping"]
}
```

#### RAGFlow Chat Schema
```json
{
    "type": "object",
    "properties": {
        "message": {
            "type": "string", 
            "minLength": 1, 
            "maxLength": 4000,
            "pattern": "^[\\s\\S]*$"
        },
        "sessionId": {"type": "string", "format": "uuid"},
        "userId": {"type": "string", "minLength": 1, "maxLength": 100},
        "context": {
            "type": "object",
            "additionalProperties": true
        }
    },
    "required": ["message"]
}
```

### Input Sanitization

#### JSON Sanitization Process
1. **HTML Entity Escaping**: Escape HTML entities in string values
2. **Script Tag Removal**: Remove or escape script tags
3. **SQL Injection Prevention**: Escape SQL metacharacters
4. **Path Traversal Protection**: Validate file paths
5. **Unicode Normalization**: Normalize unicode characters

```rust
impl Sanitizer {
    pub fn sanitize_json(payload: &mut Value) -> ValidationResult<()> {
        match payload {
            Value::String(s) => {
                *s = Self::sanitize_string(s)?;
            }
            Value::Object(map) => {
                for (_, value) in map.iter_mut() {
                    Self::sanitize_json(value)?;
                }
            }
            Value::Array(arr) => {
                for value in arr.iter_mut() {
                    Self::sanitize_json(value)?;
                }
            }
            _ => {} // Numbers, booleans, null don't need sanitization
        }
        Ok(())
    }
    
    fn sanitize_string(input: &str) -> ValidationResult<String> {
        // 1. HTML escape
        let escaped = html_escape::encode_text(input);
        
        // 2. Remove dangerous patterns
        let safe = Self::remove_dangerous_patterns(&escaped);
        
        // 3. Length validation
        if safe.len() > MAX_STRING_LENGTH {
            return Err(ValidationError::StringTooLong);
        }
        
        Ok(safe)
    }
}
```

### Error Handling

#### Validation Error Types
```rust
#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("Schema validation failed: {details}")]
    SchemaValidation { details: String },
    
    #[error("Sanitization failed: {reason}")]
    SanitizationFailed { reason: String },
    
    #[error("Custom validation failed: {field} - {message}")]
    CustomValidation { field: String, message: String },
    
    #[error("Value out of range: {field} must be between {min} and {max}")]
    ValueOutOfRange { field: String, min: f64, max: f64 },
    
    #[error("Invalid format: {field} - {expected_format}")]
    InvalidFormat { field: String, expected_format: String },
}
```

#### Detailed Error Responses
```rust
pub struct DetailedValidationError {
    pub error_type: String,
    pub field_path: String,
    pub message: String,
    pub expected_format: Option<String>,
    pub received_value: Option<String>,
}

impl ValidationService {
    fn create_detailed_error(&self, error: ValidationError) -> DetailedValidationError {
        match error {
            ValidationError::SchemaValidation { details } => {
                DetailedValidationError {
                    error_type: "schema_validation".to_string(),
                    field_path: Self::extract_field_path(&details),
                    message: details,
                    expected_format: None,
                    received_value: None,
                }
            }
            // Handle other error types...
        }
    }
}
```

### Performance Considerations

#### Caching Strategy
1. **Schema Caching**: Cache compiled JSON schemas
2. **Validation Result Caching**: Cache validation results for identical payloads
3. **Sanitization Caching**: Cache sanitized strings with hash keys
4. **Pattern Compilation**: Pre-compile regex patterns

#### Async Validation
```rust
pub async fn validate_async<T>(&self, payload: &Value) -> ValidationResult<T> 
where 
    T: serde::de::DeserializeOwned,
{
    // Async validation for large payloads or external validation services
    tokio::task::spawn_blocking(move || {
        self.validate_sync(payload)
    }).await?
}
```

### Integration Patterns

#### Middleware Integration
```rust
pub fn validation_middleware<T>() -> impl Fn(ServiceRequest, &mut T) -> Result<ServiceRequest, Error>
where
    T: Service<ServiceRequest, Error = Error>,
{
    // Automatic validation middleware for specific routes
}
```

#### Handler Integration Example
```rust
pub async fn update_settings(
    req: HttpRequest,
    payload: web::Json<Value>,
    validation_service: web::Data<ValidationService>,
) -> Result<HttpResponse, Error> {
    // 1. Extract client ID for rate limiting
    let client_id = extract_client_id(&req);
    
    // 2. Validate and sanitize payload
    let validated_data = validation_service.validate_settings_update(&payload)?;
    
    // 3. Process validated data
    // ... business logic
    
    Ok(HttpResponse::Ok().json(validated_data))
}
```

### Testing Requirements

1. **Schema Validation Tests**: Test all schema validation rules
2. **Sanitization Tests**: Verify XSS and injection prevention
3. **Custom Validation Tests**: Test business logic validation
4. **Performance Tests**: Validate under high load
5. **Security Tests**: Test against known attack vectors
6. **Integration Tests**: Test with actual API endpoints

### Security Hardening

1. **Input Length Limits**: Enforce maximum input lengths
2. **Pattern Validation**: Use whitelist-based validation
3. **Encoding Validation**: Validate character encoding
4. **Rate Limiting Integration**: Integrate with rate limiting service
5. **Audit Logging**: Log all validation failures for security monitoring