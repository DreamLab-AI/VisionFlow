# Source Files Patch Conversion

This document contains converted patches for the main Rust backend source files, organized by component and converted into clear, actionable refactoring tasks.

## **Task 1.1: Refactor Settings Actor for Path-Based Operations**

**Goal:** Replace bulk settings operations with efficient path-based access to eliminate JSON serialization bottleneck

**Actions:**

1. **src/actors/settings_actor.rs**
   - Replace: Bulk `GetSettings`/`UpdateSettings` message handlers that clone entire settings
   - With: Path-based handlers (`GetSettingByPath`, `SetSettingByPath`, `GetSettingsByPaths`, `SetSettingsByPaths`)
   - Reason: Eliminates ~90% CPU overhead from JSON serialization on every slider interaction

2. **Import Changes**
   - Replace: `use crate::actors::messages::{GetSettings, UpdateSettings, GetSettingByPath, UpdatePhysicsFromAutoBalance};`
   - With: `use crate::actors::messages::{GetSettingByPath, SetSettingByPath, GetSettingsByPaths, SetSettingsByPaths, UpdatePhysicsFromAutoBalance};`
   - Add: `use crate::config::{AppFullSettings, path_access::PathAccessible};`
   - Add: `use std::collections::HashMap;`

3. **Remove Legacy Methods**
   - Remove: `pub async fn get_settings(&self) -> AppFullSettings`
   - Remove: `pub async fn update_settings(&self, new_settings: AppFullSettings)`
   - Reason: These methods encouraged bulk operations and JSON serialization

4. **Add Path-Based Message Handlers**
   - Add: `Handler<GetSettingByPath>` - Returns single value by path
   - Add: `Handler<SetSettingByPath>` - Updates single value with validation
   - Add: `Handler<GetSettingsByPaths>` - Batch get multiple paths (performance critical)
   - Add: `Handler<SetSettingsByPaths>` - Transactional batch updates with validation

5. **Validation Integration**
   - Replace: Manual JSON path traversal
   - With: `current.validate_config_camel_case()` after path updates
   - Reason: Uses validator crate with camelCase field names for frontend compatibility

---

## **Task 1.2: Refactor Settings Handler for Granular API**

**Goal:** Replace monolithic 3000+ line settings handler with granular path-based API

**Actions:**

1. **src/handlers/settings_handler.rs**
   - Replace: Entire 3,117-line monolithic implementation
   - With: Compact 232-line granular API using automatic serde camelCase conversion
   - Reason: Simplifies maintenance and improves performance

2. **Remove Complex DTO Layer**
   - Remove: All DTO structs (`SettingsResponseDTO`, `SettingsUpdateDTO`, `VisualisationSettingsDTO`, etc.)
   - Remove: Manual camelCase conversion logic (1000+ lines)
   - Reason: Automatic serde conversion handles this more reliably

3. **Import Simplification**
   - Replace: `use actix_web::{web, Error, HttpResponse, HttpRequest};`
   - With: `use actix_web::{web, HttpResponse, HttpRequest, Result};`
   - Remove: Validation handler and rate limiting imports (moved to middleware)

4. **API Endpoints Refactor**
   - Replace: Single `/settings` endpoint with bulk operations
   - With: Granular endpoints:
     - `GET /settings/{path}` - Get specific setting by path
     - `PUT /settings/{path}` - Update specific setting by path  
     - `POST /settings/batch` - Batch operations for multiple paths

5. **Response Format**
   - Replace: Complex nested DTO responses
   - With: Direct JSON responses using automatic camelCase conversion
   - Reason: Reduces code complexity and improves performance

---

## **Task 1.3: Add Path Access Trait System**

**Goal:** Create efficient direct field access without JSON serialization bottleneck

**Actions:**

1. **src/config/path_access.rs (New File)**
   - Create: `PathAccessible` trait with `get_by_path()` and `set_by_path()` methods
   - Create: `parse_path()` helper for dot-notation path parsing
   - Create: `impl_field_access!` macro for common field access patterns
   - Reason: Enables direct struct field access without JSON conversion overhead

2. **Path Validation**
   - Add: Empty path validation
   - Add: Empty segment validation in dot-notation paths
   - Reason: Prevents runtime errors from malformed paths

3. **Macro Implementation**
   - Create: Pattern matching macro for field access
   - Support: Nested field access through recursive calls
   - Reason: Reduces boilerplate code for implementing PathAccessible

---

## **Task 1.4: Enhance Config Module with Validation**

**Goal:** Add comprehensive validation system with camelCase support

**Actions:**

1. **src/config/mod.rs**
   - Add: `use specta::Type;` for TypeScript type generation
   - Add: `use validator::{Validate, ValidationError};` for validation
   - Add: `use regex::Regex;` and `lazy_static::lazy_static;` for pattern validation

2. **Add Validation Patterns**
   - Add: `HEX_COLOR_REGEX` for color validation
   - Add: `URL_REGEX` for URL validation
   - Add: `FILE_PATH_REGEX` for file path validation
   - Add: `DOMAIN_REGEX` for domain validation
   - Reason: Centralized validation patterns reduce code duplication

3. **Custom Validation Functions**
   - Add: `validate_hex_color()` - Ensures valid hex color format
   - Add: `validate_width_range()` - Validates 2-element ranges with proper min/max
   - Add: `validate_port()` - Ensures port numbers are valid (1-65535)
   - Add: `validate_percentage()` - Ensures values are 0-100%

4. **Add Validation Attributes to Structs**
   - Update: `NodeSettings` with field-level validation attributes
   - Update: `EdgeSettings` with range and color validation
   - Update: `MovementAxes` with range validation (-100 to 100)
   - Reason: Compile-time validation ensures data integrity

5. **Remove Legacy Helper Functions**
   - Remove: `convert_empty_strings_to_null()` function (~40 lines)
   - Reason: Replaced by proper validation and serde configuration

---

## **Task 1.5: Clean Up Claude Flow Actor TCP**

**Goal:** Remove deprecated reconnection logic to prevent cascading failures

**Actions:**

1. **src/actors/claude_flow_actor_tcp.rs**
   - Remove: Deprecated periodic reconnection check in `started()` method
   - Remove: 8 lines of commented-out redundant connection checking code
   - Keep: Single comment explaining that reconnection is handled by ConnectionFailed handler
   - Reason: Prevents cascading failures and redundant reconnection attempts

2. **Code Cleanup**
   - Replace: Block comment explaining deprecation
   - With: Single line comment about handler delegation
   - Reason: Reduces code noise and improves readability

---

## **Key Performance Improvements Achieved:**

1. **JSON Serialization Elimination**: Path-based access removes need to serialize entire AppFullSettings struct for single field updates
2. **Batch Operations**: GetSettingsByPaths and SetSettingsByPaths enable efficient multi-field operations
3. **Automatic Validation**: Validator crate with camelCase support eliminates manual validation code
4. **Direct Field Access**: PathAccessible trait enables struct field access without JSON conversion
5. **Simplified API**: Granular endpoints reduce payload sizes and improve caching

## **Files Modified:**
- `src/actors/settings_actor.rs` - Path-based message handling
- `src/handlers/settings_handler.rs` - Granular API endpoints
- `src/config/path_access.rs` - New direct access trait
- `src/config/mod.rs` - Enhanced validation system
- `src/actors/claude_flow_actor_tcp.rs` - Cleanup deprecated code

## **Testing Requirements:**
- Verify path-based setting updates work correctly
- Ensure validation errors return camelCase field names
- Test batch operations for performance improvement
- Confirm WebSocket integration still functions
- Validate frontend compatibility with new API format