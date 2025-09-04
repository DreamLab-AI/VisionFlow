# Configuration Module Refactoring Instructions

This directory contains comprehensive instructions for refactoring the configuration system to support path-based operations, enhanced validation, and improved performance.

## Overview

The configuration system refactoring eliminates the JSON serialization bottleneck that occurs during settings updates, particularly problematic with UI slider interactions that caused ~90% CPU overhead. The new system introduces direct field access through path notation while maintaining type safety and validation.

## Key Files and Components

### 1. PathAccessible Trait Implementation
**File:** `path_access_trait_instructions.md`

- **Purpose:** Create efficient direct field access without JSON serialization
- **Key Features:**
  - Dot-notation path parsing (`"ui.theme.primaryColor"`)
  - Type-safe field access with error handling
  - Recursive nested field support
  - Macro-based implementation generation

### 2. Validation Enhancements
**File:** `validation_enhancements_instructions.md`

- **Purpose:** Add comprehensive validation with camelCase frontend compatibility
- **Key Features:**
  - Centralized regex patterns for common validation
  - Custom validation functions for business logic
  - Compile-time validation attributes
  - User-friendly error messages in camelCase format

### 3. Settings Structure Refactoring
**File:** `settings_structure_refactor_instructions.md`

- **Purpose:** Restructure the configuration module for better organization
- **Key Features:**
  - Modular file organization
  - Metadata tracking for change detection
  - Settings summary and utility functions
  - JSON export/import with validation

## Implementation Order

### Phase 1: Foundation
1. Create `src/config/path_access.rs` with PathAccessible trait
2. Update `src/config/mod.rs` with enhanced validation system
3. Add validation patterns and custom validation functions

### Phase 2: Structure Enhancement
1. Implement PathAccessible for main settings structures
2. Add settings metadata tracking
3. Create utility functions for settings management

### Phase 3: Integration
1. Update settings actor to use path-based operations
2. Refactor settings handler for granular API
3. Update message types for new operations

## Performance Improvements

### Before Refactoring
- **CPU Usage:** ~90% during slider interactions
- **Bottleneck:** JSON serialization of entire AppFullSettings struct
- **API Responses:** Large payloads with complete settings objects
- **Validation:** Manual validation code scattered throughout

### After Refactoring
- **CPU Usage:** Minimal overhead for single field updates
- **Direct Access:** Field access without JSON conversion
- **API Responses:** Granular responses for specific paths
- **Validation:** Centralized, compile-time validation with clear errors

## New Message Types Required

The refactoring requires new message types for the actor system:

```rust
// Path-based single operations
pub struct GetSettingByPath {
    pub path: String,
}

pub struct SetSettingByPath {
    pub path: String,
    pub value: Box<dyn Any>,
}

// Batch operations for performance
pub struct GetSettingsByPaths {
    pub paths: Vec<String>,
}

pub struct SetSettingsByPaths {
    pub updates: HashMap<String, Box<dyn Any>>,
}
```

## API Endpoint Changes

### Old Monolithic API
```
GET  /settings           # Returns entire settings object
PUT  /settings           # Updates entire settings object
```

### New Granular API
```
GET  /settings/{path}    # Get specific setting by path
PUT  /settings/{path}    # Update specific setting by path
POST /settings/batch     # Batch operations for multiple paths
```

## Frontend Integration

### Path Notation Examples
```javascript
// UI theme settings
"ui.theme.primaryColor"
"ui.theme.darkMode"

// Visualization settings
"visualisation.nodes.enabled"
"visualisation.edges.widthRange"

// Physics settings
"physics.gravity"
"physics.simulation.enabled"
```

### Error Response Format (camelCase)
```json
{
  "errors": {
    "primaryColor": ["Must be a valid hex color (#RRGGBB or #RRGGBBAA)"],
    "widthRange": ["Width range minimum must be less than maximum"]
  }
}
```

## Testing Requirements

### Unit Tests
- Path parsing validation
- Field access correctness
- Type safety verification
- Validation rule enforcement

### Integration Tests
- Settings actor message handling
- API endpoint functionality
- Frontend compatibility
- Error response format

### Performance Tests
- Single field update latency
- Batch operation throughput
- Memory usage comparison
- CPU usage during UI interactions

## Migration Strategy

### Phase 1: Backward Compatibility
- Keep old API endpoints functional
- Add new path-based endpoints
- Test both systems in parallel

### Phase 2: Frontend Migration
- Update frontend to use new granular APIs
- Test UI responsiveness improvements
- Validate error handling

### Phase 3: Legacy Removal
- Remove old monolithic endpoints
- Clean up deprecated code
- Update documentation

## Validation Patterns Reference

### Common Patterns
- **Hex Colors:** `#RRGGBB` or `#RRGGBBAA`
- **URLs:** `http://` or `https://` protocols
- **Ports:** Range 1-65535
- **Percentages:** Range 0-100
- **File Paths:** Unix/Windows compatible

### Custom Business Rules
- Physics gravity requires physics enabled
- Visualization settings require visualization enabled
- Width ranges must have min < max
- Color schemes must be complete sets

## Files to Modify

### New Files
- `src/config/path_access.rs` - PathAccessible trait
- `src/config/validation.rs` - Validation patterns and functions

### Modified Files
- `src/config/mod.rs` - Enhanced with validation and structure changes
- `src/actors/settings_actor.rs` - Path-based message handlers
- `src/handlers/settings_handler.rs` - Granular API endpoints
- `src/actors/messages.rs` - New message types

## Success Metrics

- **Performance:** <5ms response time for single field updates
- **CPU Usage:** <10% during UI slider interactions
- **API Response Size:** 90% reduction for single field requests
- **Error Clarity:** 100% camelCase error field names
- **Type Safety:** Zero runtime type errors in field access

## Support and Troubleshooting

### Common Issues
1. **Path Not Found:** Check dot notation syntax and field names
2. **Type Mismatch:** Ensure value types match field expectations
3. **Validation Errors:** Check validation rules and constraints
4. **Performance:** Monitor batch operation sizes and frequency

### Debug Tools
- Enable validation logging for detailed error information
- Use settings metadata to track changes and validation status
- Monitor API response times and payload sizes
- Check field access patterns in path-based operations