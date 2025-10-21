# Settings API Endpoints Audit

**Date**: 2025-10-21
**Objective**: Document actual HTTP endpoint formats, parameter naming conventions, and data flow

---

## Executive Summary

The settings API has **THREE separate handler implementations** with different conventions:

1. **`/api/settings/*`** (REST API) - Uses **camelCase** consistently
2. **`/settings/*`** (Legacy) - Converts camelCase → snake_case internally
3. **`/api/user-settings/*`** (User-specific) - Uses plain JSON (no serde rename)

**Key Finding**: There is NO inconsistency in the API layer. Both `/api/settings` and `/settings` properly handle camelCase from clients. The confusion stems from having multiple overlapping implementations.

---

## 1. REST API Endpoints (`/api/settings/*`)

**Handler**: `/home/devuser/workspace/project/src/handlers/api_handler/settings/mod.rs`
**Route Registration**: Line 46 in `src/handlers/api_handler/mod.rs`

### 1.1 List All Settings
```
GET /api/settings
```

**Request**: None
**Response Format**: camelCase
```json
{
  "settings": [
    {
      "key": "physics.damping",
      "value": 0.95,
      "valueType": "float"
    }
  ],
  "total": 1
}
```

**DTOs**:
- Response: `SettingsListResponse` with `#[serde(rename_all = "camelCase")]`
- Fields: `settings`, `total`

---

### 1.2 Get Single Setting
```
GET /api/settings/{key}
```

**Request**: Key in URL path (supports dots: `physics.damping`)
**Response Format**: camelCase
```json
{
  "key": "physics.damping",
  "value": 0.95,
  "valueType": "float"
}
```

**Note**: Response uses `valueType` (camelCase) hardcoded in handler (line 127)

---

### 1.3 Update Single Setting
```
PUT /api/settings/{key}
```

**Request Format**: camelCase
```json
{
  "value": 0.98
}
```

**DTO**: `UpdateSettingRequest` with `#[serde(rename_all = "camelCase")]`

**Response Format**: camelCase
```json
{
  "success": true,
  "key": "physics.damping",
  "message": "Setting updated successfully"
}
```

**Permissions**: Requires power user (checked via `check_power_user`)

---

### 1.4 Delete/Reset Setting
```
DELETE /api/settings/{key}
```

**Response Format**: camelCase
```json
{
  "success": true,
  "key": "physics.damping",
  "message": "Setting reset to default"
}
```

**Permissions**: Requires power user

---

### 1.5 Search Settings
```
GET /api/settings/search?q=pattern
```

**Request**: Query parameter `q`
**Response Format**: camelCase
```json
{
  "results": [
    {
      "key": "physics.damping",
      "value": 0.95,
      "valueType": "float"
    }
  ],
  "count": 1
}
```

---

### 1.6 Validate Setting
```
POST /api/settings/validate
```

**Request Format**: camelCase
```json
{
  "key": "physics.damping",
  "value": 0.98
}
```

**DTO**: `ValidateSettingRequest` with `#[serde(rename_all = "camelCase")]`

**Response Format**: camelCase
```json
{
  "isValid": true,
  "errors": [],
  "warnings": []
}
```

**DTO**: `ValidationResponse` with `#[serde(rename_all = "camelCase")]`

---

### 1.7 Get Settings Tree
```
GET /api/settings/tree/{prefix}
```

**Request**: Prefix in URL path
**Response**: Raw JSON tree (no specific DTO)

---

### 1.8 Physics Profile - Get
```
GET /api/settings/physics/{profile}
```

**Request**: Profile name in URL path
**Response**: `PhysicsSettings` struct

---

### 1.9 Physics Profile - Update
```
PUT /api/settings/physics/{profile}
```

**Request**: Full `PhysicsSettings` struct
**Permissions**: Requires power user

---

## 2. Path-Based Settings Endpoints (`/settings/path`, `/settings/batch`)

**Handler**: `/home/devuser/workspace/project/src/handlers/settings_paths.rs`
**Route Registration**: Included via `configure_settings_paths` in settings_handler config

### 2.1 Get by Path
```
GET /api/settings/path?path=visualisation.physics.damping
```

**Request**: Query parameter `path` (camelCase dot notation)
**DTO**: `PathQuery` with `#[serde(rename_all = "camelCase")]`

**Response Format**: camelCase
```json
{
  "value": 0.95,
  "path": "visualisation.physics.damping",
  "success": true
}
```

**Data Flow**:
1. Receives camelCase path: `visualisation.physics.damping`
2. Uses `JsonPathAccessible::get_json_by_path()` on `AppFullSettings`
3. Returns value at that path

---

### 2.2 Update by Path
```
PUT /api/settings/path
```

**Request Format**: camelCase
```json
{
  "path": "visualisation.physics.damping",
  "value": 0.98
}
```

**DTO**: `PathUpdateRequest` with `#[serde(rename_all = "camelCase")]`

**Response Format**: camelCase
```json
{
  "success": true,
  "path": "visualisation.physics.damping",
  "value": 0.98,
  "previousValue": 0.95,
  "message": "Settings updated successfully"
}
```

**Data Flow**:
1. Receives camelCase path and value
2. Gets current settings from actor
3. Calls `settings.set_json_by_path(path, value)`
4. Validates using `settings.validate_config_camel_case()`
5. Sends updated settings back to actor

**Validation**: Uses `AppFullSettings::validate_config_camel_case()`

---

### 2.3 Batch Read
```
POST /api/settings/batch
```

**Request Format**: camelCase
```json
{
  "paths": [
    "visualisation.physics.damping",
    "visualisation.physics.gravity"
  ]
}
```

**DTO**: `BatchPathReadRequest` with `#[serde(rename_all = "camelCase")]`

**Response Format**: Direct key-value map (camelCase keys)
```json
{
  "visualisation.physics.damping": 0.95,
  "visualisation.physics.gravity": 0.001
}
```

**Important**: Response is a direct `serde_json::Map`, NOT wrapped in `{"values": ...}` (line 282)

**Limits**: Maximum 50 paths per request

---

### 2.4 Batch Update
```
PUT /api/settings/batch
```

**Request Format**: camelCase
```json
{
  "updates": [
    {
      "path": "visualisation.physics.damping",
      "value": 0.98
    },
    {
      "path": "visualisation.physics.gravity",
      "value": 0.002
    }
  ]
}
```

**DTO**: `BatchPathUpdateRequest` with `#[serde(rename_all = "camelCase")]`

**Response Format**: camelCase
```json
{
  "success": true,
  "message": "Successfully updated 2 settings",
  "results": [
    {
      "path": "visualisation.physics.damping",
      "success": true,
      "value": 0.98,
      "previousValue": 0.95
    },
    {
      "path": "visualisation.physics.gravity",
      "success": true,
      "value": 0.002,
      "previousValue": 0.001
    }
  ]
}
```

**Data Flow**:
1. Receives array of camelCase path updates
2. Gets current settings
3. Applies each update via `set_json_by_path()`
4. Validates entire settings with `validate_config_camel_case()`
5. Sends updated settings to actor
6. Returns results array

**Limits**: Maximum 50 updates per request
**Validation**: Atomic - if any validation fails, entire batch is rejected

---

### 2.5 Get Schema
```
GET /api/settings/schema?path=visualisation.physics
```

**Request**: Query parameter `path`
**Response**: JSON schema for the specified path

---

## 3. Legacy Settings Endpoints (`/settings/*`)

**Handler**: `/home/devuser/workspace/project/src/handlers/settings_handler.rs`
**Route Registration**: Line 1673-1705

### 3.1 Get Settings (Legacy)
```
GET /settings
```

**Response Format**: camelCase (via `SettingsResponseDTO`)
```json
{
  "visualisation": {
    "physics": {
      "damping": 0.95
    }
  }
}
```

**DTO**: `SettingsResponseDTO` with `#[serde(rename_all = "camelCase")]`

---

### 3.2 Update Settings (Legacy)
```
POST /settings
```

**Request Format**: Accepts **camelCase**, converts to snake_case internally
```json
{
  "visualisation": {
    "physics": {
      "damping": 0.98
    }
  }
}
```

**Data Flow**:
1. Receives camelCase JSON
2. Calls `convert_to_snake_case_recursive(&mut update)` (line 2158)
3. Merges with existing settings
4. Validates
5. Sends to actor

**Response Format**: camelCase
```json
{
  "success": true,
  "message": "Settings updated successfully"
}
```

---

### 3.3 Current Settings
```
GET /settings/current
```

**Response Format**: camelCase with version info
```json
{
  "settings": {
    "visualisation": {...}
  },
  "version": 42,
  "timestamp": 1729512345
}
```

---

### 3.4 Batch Update (Legacy - DIFFERENT ENDPOINT!)
```
UNKNOWN - Function exists but route not clearly registered
```

**Function**: `batch_update_settings()` at line 1917
**Request Format**: camelCase
```json
{
  "updates": [
    {
      "path": "visualisation.physics.damping",
      "value": 0.98
    }
  ]
}
```

**Note**: This function exists but its route registration is unclear. May be deprecated.

---

### 3.5 Save Settings
```
POST /settings/save
```

Persists current settings to file.

---

### 3.6 Reset Settings
```
POST /settings/reset
```

Resets to default settings.

---

## 4. User-Specific Settings (`/api/user-settings/*`)

**Handler**: `/home/devuser/workspace/project/src/handlers/user_settings_handler.rs`
**Feature**: Only available with `ontology` feature flag

### 4.1 Get User Settings
```
GET /api/user-settings
```

**Authentication**: Required (via `extract_auth_context`)

**Response Format**: snake_case (NO serde rename!)
```json
{
  "settings": [
    {
      "key": "theme",
      "value": "dark",
      "created_at": 1729512345,
      "updated_at": 1729512345
    }
  ]
}
```

**DTO**: `UserSettingsResponse` (line 10) - **NO rename_all attribute**

---

### 4.2 Set User Setting
```
POST /api/user-settings
```

**Request Format**: snake_case
```json
{
  "key": "theme",
  "value": "dark"
}
```

**DTO**: `SetUserSettingRequest` (line 23) - **NO rename_all attribute**

**Permissions**: Requires `is_power_user` flag

---

### 4.3 Delete User Setting
```
DELETE /api/user-settings/{key}
```

**Permissions**: Requires `is_power_user` flag

---

## 5. Data Flow Diagrams

### 5.1 REST API Flow (camelCase → camelCase)

```
Client Request (camelCase)
    ↓
PUT /api/settings/physics.damping
{
  "value": 0.98         ← camelCase DTO deserialization
}
    ↓
UpdateSettingRequest (#[serde(rename_all = "camelCase")])
    ↓
json_value_to_setting_value() - No case conversion
    ↓
SettingsService::set_setting()
    ↓
DatabaseService (snake_case column names)
    ↓
Response (camelCase via json! macro)
{
  "success": true,      ← Manual camelCase construction
  "key": "physics.damping"
}
```

---

### 5.2 Path-Based Flow (camelCase → JsonPath → Validation)

```
Client Request (camelCase)
    ↓
PUT /api/settings/path
{
  "path": "visualisation.physics.damping",  ← camelCase path
  "value": 0.98
}
    ↓
PathUpdateRequest (#[serde(rename_all = "camelCase")])
    ↓
settings.get_json_by_path("visualisation.physics.damping")
    ↓
JsonPathAccessible trait - Direct field access
    ↓
settings.set_json_by_path(path, value)
    ↓
settings.validate_config_camel_case()  ← Validates with camelCase paths
    ↓
state.settings_addr.send(UpdateSettings { settings })
    ↓
Response (camelCase)
{
  "success": true,
  "path": "visualisation.physics.damping",
  "value": 0.98,
  "previousValue": 0.95
}
```

---

### 5.3 Legacy Flow (camelCase → snake_case → camelCase)

```
Client Request (camelCase)
    ↓
POST /settings
{
  "visualisation": {     ← camelCase structure
    "enableHologram": true
  }
}
    ↓
web::Json<Value> (raw JSON, no DTO)
    ↓
convert_to_snake_case_recursive(&mut update)  ← CASE CONVERSION HERE
    ↓
{
  "visualisation": {
    "enable_hologram": true  ← Now snake_case
  }
}
    ↓
Merge with AppFullSettings (snake_case fields)
    ↓
Serialize to JSON for response
    ↓
SettingsResponseDTO (#[serde(rename_all = "camelCase")])
    ↓
Response (camelCase)
{
  "visualisation": {
    "enableHologram": true  ← Back to camelCase
  }
}
```

---

## 6. Case Conversion Analysis

### 6.1 Where Case Conversion Happens

**REST API (`/api/settings/*`)**:
- **Request**: Serde deserializes camelCase → struct fields automatically
- **Response**: Serde serializes struct fields → camelCase automatically
- **No manual conversion**: DTOs have `#[serde(rename_all = "camelCase")]`

**Path-Based (`/settings/path`, `/settings/batch`)**:
- **Request**: DTOs use `#[serde(rename_all = "camelCase")]`
- **Processing**: JsonPath operates on camelCase field names directly
- **Validation**: `validate_config_camel_case()` expects camelCase paths
- **No manual conversion**: Everything stays in camelCase

**Legacy (`/settings`)**:
- **Request**: Accepts camelCase JSON as raw `Value`
- **Conversion**: Line 2158 - `convert_to_snake_case_recursive(&mut update)`
- **Processing**: Works with snake_case internally
- **Response**: DTO serializes back to camelCase

---

### 6.2 Validation Methods

| Method | Expected Format | Used By |
|--------|----------------|---------|
| `validate_config_camel_case()` | camelCase paths | Path-based endpoints, WebSocket |
| `validate_config()` (if exists) | snake_case paths | Legacy endpoints |
| `get_validation_errors_camel_case()` | Returns camelCase field names | Path-based responses |

---

## 7. Endpoint Inconsistencies

### 7.1 Multiple Implementations of Same Functionality

**Problem**: Three different ways to update settings:

1. `PUT /api/settings/{key}` - REST API, single key-value
2. `PUT /api/settings/path` - Path-based, single key-value
3. `POST /settings` - Legacy, full settings object
4. `PUT /api/settings/batch` - Path-based, multiple updates

**Recommendation**: Deprecate legacy `/settings` endpoints in favor of `/api/settings/*`

---

### 7.2 Response Format Differences

**Batch Read Response**:
- Path-based: Returns direct map `{"key": value}` (line 282)
- Expected by some clients: `{"values": [...]}`

**Suggestion**: Document this behavior or add wrapper for consistency

---

### 7.3 User Settings vs Global Settings

**Global Settings** (`/api/settings/*`):
- Uses camelCase consistently
- Permission checks for updates
- Stored in settings actor

**User Settings** (`/api/user-settings/*`):
- Uses snake_case (no serde rename)
- Requires authentication + power user
- Stored in database per-user

**Issue**: Naming confusion - both are "settings" but have different APIs

---

## 8. Complete Endpoint List

| Method | Endpoint | Handler | Request Format | Response Format | Auth Required |
|--------|----------|---------|----------------|-----------------|---------------|
| GET | `/api/settings` | REST | - | camelCase | No |
| GET | `/api/settings/{key}` | REST | - | camelCase | No |
| PUT | `/api/settings/{key}` | REST | camelCase | camelCase | Power User |
| DELETE | `/api/settings/{key}` | REST | - | camelCase | Power User |
| GET | `/api/settings/search?q=` | REST | - | camelCase | No |
| POST | `/api/settings/validate` | REST | camelCase | camelCase | No |
| GET | `/api/settings/tree/{prefix}` | REST | - | Raw JSON | No |
| GET | `/api/settings/physics/{profile}` | REST | - | PhysicsSettings | No |
| PUT | `/api/settings/physics/{profile}` | REST | PhysicsSettings | camelCase | Power User |
| GET | `/api/settings/path?path=` | Path | camelCase | camelCase | No |
| PUT | `/api/settings/path` | Path | camelCase | camelCase | No |
| POST | `/api/settings/batch` | Path | camelCase | Direct map | No |
| PUT | `/api/settings/batch` | Path | camelCase | camelCase | No |
| GET | `/api/settings/schema?path=` | Path | - | JSON schema | No |
| GET | `/settings` | Legacy | - | camelCase | No |
| POST | `/settings` | Legacy | camelCase→snake | camelCase | No |
| GET | `/settings/current` | Legacy | - | camelCase | No |
| POST | `/settings/save` | Legacy | - | - | No |
| POST | `/settings/reset` | Legacy | - | - | No |
| GET | `/api/user-settings` | User | - | snake_case | Yes |
| POST | `/api/user-settings` | User | snake_case | snake_case | Power User |
| DELETE | `/api/user-settings/{key}` | User | - | snake_case | Power User |

---

## 9. Recommendations

### 9.1 Immediate Actions

1. **Document batch read response format**: Clearly state it returns a direct map, not wrapped array
2. **Deprecate legacy endpoints**: Add deprecation warnings to `/settings` (non-path) endpoints
3. **Standardize user settings**: Add `#[serde(rename_all = "camelCase")]` to user settings DTOs
4. **Add endpoint versioning**: Consider `/api/v1/settings` for future changes

### 9.2 API Consolidation

**Keep**:
- `/api/settings/*` (REST API) - Primary interface
- `/api/settings/path` and `/api/settings/batch` - Efficient field-level access

**Deprecate**:
- `/settings` (non-path endpoints) - Redirect to `/api/settings`

### 9.3 Consistency Improvements

1. All endpoints should use camelCase for requests and responses
2. User settings should align with global settings naming
3. Batch operations should have consistent response structures
4. Permission checks should be consistent (some check, some don't)

---

## 10. Client Integration Guide

### 10.1 Recommended Endpoint Usage

**For reading single values**:
```javascript
GET /api/settings/path?path=visualisation.physics.damping
```

**For reading multiple values**:
```javascript
POST /api/settings/batch
{
  "paths": ["visualisation.physics.damping", "visualisation.physics.gravity"]
}
```

**For updating single value**:
```javascript
PUT /api/settings/path
{
  "path": "visualisation.physics.damping",
  "value": 0.98
}
```

**For updating multiple values atomically**:
```javascript
PUT /api/settings/batch
{
  "updates": [
    {"path": "visualisation.physics.damping", "value": 0.98},
    {"path": "visualisation.physics.gravity", "value": 0.002}
  ]
}
```

---

## 11. Validation Flow

```
Client sends camelCase request
    ↓
Serde deserializes to DTO (camelCase fields)
    ↓
Handler extracts path and value
    ↓
JsonPathAccessible::set_json_by_path(camelCase_path, value)
    ↓
AppFullSettings::validate_config_camel_case()
    ├─ Checks field types
    ├─ Validates ranges
    └─ Returns ValidationError with camelCase field names
    ↓
If valid: Save to actor
If invalid: Return error with camelCase field names
    ↓
Response serialized with camelCase
```

---

## Conclusion

The settings API is **consistent at the HTTP layer** - all public endpoints accept and return camelCase. The complexity arises from:

1. **Multiple implementations**: Three separate handler files doing similar things
2. **Internal conversions**: Legacy handler converts camelCase→snake_case internally
3. **Feature-gated endpoints**: User settings only available with `ontology` feature

**Key Insight**: Clients should use `/api/settings/path` and `/api/settings/batch` for all operations. These endpoints are clean, efficient, and consistently use camelCase throughout the entire stack.
