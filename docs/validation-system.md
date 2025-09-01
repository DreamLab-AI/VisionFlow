# VisionFlow Settings Validation System

## Overview

The VisionFlow settings system now includes comprehensive validation at multiple layers to ensure data integrity and provide clear user feedback.

## Validation Layers

### 1. Rust Backend Validation (Source of Truth)

#### Configuration Structure Validation

All configuration structs now include `validator` derive attributes:

```rust
#[derive(Debug, Serialize, Deserialize, Clone, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct NodeSettings {
    #[validate(regex(path = "HEX_COLOR_REGEX", message = "Must be a valid hex color (e.g., #ff0000)"))]
    pub base_color: String,
    
    #[validate(range(min = 0.0, max = 1.0, message = "Opacity must be between 0.0 and 1.0"))]
    pub opacity: f32,
    
    #[validate(range(min = 0.1, max = 100.0, message = "Node size must be between 0.1 and 100.0"))]
    pub node_size: f32,
    
    // ... other fields
}
```

#### Validation Rules by Field Type

##### Numeric Fields
- **Ports**: 1-65535
- **Percentages/Opacity**: 0.0-1.0 
- **Sizes**: 0.1-1000.0
- **Physics Parameters**: Range-specific limits (e.g., damping 0.0-1.0)
- **Update Rates**: 1-120 FPS

##### String Fields
- **Colors**: Valid hex format `#RRGGBB`
- **URLs**: Valid HTTP/HTTPS format
- **File Paths**: Valid path characters only
- **Domains**: Valid domain format
- **Required Fields**: Non-empty validation

##### Complex Fields
- **Range Arrays**: [min, max] where min < max
- **Color Arrays**: Valid hex colors for dual color pickers
- **Network Settings**: Comprehensive networking parameter validation

#### Custom Validation Functions

```rust
fn validate_hex_color(color: &str) -> Result<(), ValidationError> {
    if HEX_COLOR_REGEX.is_match(color) {
        Ok(())
    } else {
        Err(ValidationError::new("invalid_hex_color"))
    }
}

fn validate_port(port: u16) -> Result<(), ValidationError> {
    if port == 0 || port > 65535 {
        let mut error = ValidationError::new("invalid_port");
        error.message = Some("Port must be between 1 and 65535".into());
        return Err(error);
    }
    Ok(())
}
```

### 2. API Layer Validation

#### Input Validation
- Path-based value validation before processing
- Type checking for numeric values
- Format validation for colors, URLs, etc.
- Range validation for constrained values

#### Response Structure
```rust
#[derive(Debug, Serialize)]
pub struct SettingsSetResponse {
    pub success: bool,
    pub updated_paths: Vec<String>,
    pub errors: Vec<String>,
    pub validation_errors: Option<HashMap<String, String>>,
}
```

#### Validation Process
1. Validate individual setting values by path
2. Deserialize to AppFullSettings
3. Run comprehensive struct validation
4. Return detailed error messages on failure

### 3. Frontend Validation

#### Real-time Validation
- Immediate feedback on value changes
- Debounced validation for text inputs
- Visual error indicators

#### Validation Rules
```typescript
const validateValue = (value: any, settingDef: UISettingDefinition): string | null => {
  // Required field validation
  if (settingDef.required && (value === undefined || value === null || value === '')) {
    return 'This field is required';
  }

  // Type-specific validation
  switch (settingDef.type) {
    case 'numberInput':
    case 'slider':
      const numValue = typeof value === 'string' ? parseFloat(value) : value;
      if (isNaN(numValue)) return 'Must be a valid number';
      if (settingDef.min !== undefined && numValue < settingDef.min) {
        return `Must be at least ${settingDef.min}`;
      }
      if (settingDef.max !== undefined && numValue > settingDef.max) {
        return `Must be at most ${settingDef.max}`;
      }
      break;
    // ... other cases
  }
  return null;
};
```

#### UI Feedback
- Red border for invalid inputs
- Error icon with descriptive message
- Prevents submission of invalid data
- Accessible error messages with ARIA attributes

## Validation Boundaries & Defaults

### Network Settings
| Setting | Min | Max | Default | Description |
|---------|-----|-----|---------|-------------|
| port | 1 | 65535 | 3000 | Server port |
| maxRequestSize | 1KB | 100MB | 10MB | Request size limit |
| rateLimit | 1 | 10000 | 100 | Requests per window |
| timeout | 1s | 300s | 30s | Request timeout |

### Physics Settings
| Setting | Min | Max | Default | Description |
|---------|-----|-----|---------|-------------|
| damping | 0.0 | 1.0 | 0.95 | Motion damping |
| iterations | 1 | 10000 | 100 | Simulation steps |
| maxVelocity | 0.1 | 1000.0 | 1.0 | Maximum velocity |
| temperature | 0.0 | 10.0 | 0.01 | System temperature |

### WebSocket Settings  
| Setting | Min | Max | Default | Description |
|---------|-----|-----|---------|-------------|
| updateRate | 1 | 120 | 60 | FPS update rate |
| maxConnections | 1 | 10000 | 100 | Connection limit |
| heartbeatInterval | 1s | 60s | 10s | Heartbeat timing |
| maxMessageSize | 1KB | 100MB | 10MB | Message size limit |

### Visual Settings
| Setting | Min | Max | Default | Description |
|---------|-----|-----|---------|-------------|
| opacity | 0.0 | 1.0 | 1.0 | Transparency |
| nodeSize | 0.1 | 100.0 | 1.0 | Node scale |
| arrowSize | 0.1 | 10.0 | 1.0 | Arrow scale |
| colors | - | - | #000000 | Hex color format |

## Error Messages

### User-Friendly Messages
- "Port must be between 1 and 65535"
- "Must be a valid hex color (e.g., #ff0000)"
- "Opacity must be between 0.0 and 1.0"
- "Must be a valid URL (e.g., https://example.com)"
- "This field is required"
- "Minimum must be less than maximum"

### Technical Error Codes
- `VALIDATION_ERROR`: Field-level validation failure
- `DESERIALIZATION_ERROR`: JSON structure invalid
- `RANGE_ERROR`: Value outside acceptable range
- `FORMAT_ERROR`: Invalid format (color, URL, etc.)
- `REQUIRED_ERROR`: Missing required field

## Implementation Benefits

1. **Data Integrity**: Prevents invalid configurations from corrupting the system
2. **User Experience**: Clear, immediate feedback on input errors
3. **API Safety**: Server-side validation prevents malicious or malformed requests
4. **Type Safety**: Rust validation ensures compile-time safety
5. **Consistency**: Single source of truth for validation rules
6. **Accessibility**: ARIA-compliant error reporting

## Testing Validation

### Unit Tests
- Test boundary values (min, max, edge cases)
- Invalid format testing (malformed colors, URLs)
- Required field validation
- Type coercion validation

### Integration Tests
- End-to-end validation flow
- API error response formatting
- Frontend error display
- Settings persistence with validation

### Example Test Cases
```rust
#[test]
fn test_port_validation() {
    let invalid_ports = [0, 65536, 99999];
    for port in invalid_ports {
        assert!(validate_port(port).is_err());
    }
    
    let valid_ports = [1, 80, 443, 8080, 65535];
    for port in valid_ports {
        assert!(validate_port(port).is_ok());
    }
}
```

## Future Enhancements

1. **Dynamic Validation**: Context-dependent rules
2. **Cross-Field Validation**: Validation rules spanning multiple fields
3. **Async Validation**: Server-side validation for complex rules
4. **Validation Profiles**: Different validation sets for different user types
5. **Custom Validators**: User-defined validation rules