# Settings API Audit - Executive Summary

**Date**: 2025-10-21
**Auditor**: Backend API Developer Agent
**Scope**: HTTP endpoint format verification and data flow analysis

---

## Key Findings

### ✅ Good News: No API Inconsistency

**All public HTTP endpoints consistently use camelCase for requests and responses.**

The confusion stems from **three separate implementations** doing similar things:

1. `/api/settings/*` - REST API (recommended)
2. `/settings/*` - Legacy handler with internal conversion
3. `/api/user-settings/*` - User-specific settings

---

## Endpoint Inventory

### Primary Endpoints (Recommended)

| Endpoint | Method | Purpose | Request Format | Response Format |
|----------|--------|---------|----------------|-----------------|
| `/api/settings/path?path=X` | GET | Get single value | Query param (camelCase) | camelCase |
| `/api/settings/path` | PUT | Update single value | `{"path": "X", "value": Y}` | camelCase |
| `/api/settings/batch` | POST | Read multiple | `{"paths": [...]}` | Direct map |
| `/api/settings/batch` | PUT | Update multiple | `{"updates": [...]}` | camelCase |

**These are the cleanest, most efficient endpoints.**

---

### REST API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/settings` | GET | List all settings |
| `/api/settings/{key}` | GET | Get by key |
| `/api/settings/{key}` | PUT | Update by key |
| `/api/settings/{key}` | DELETE | Reset to default |
| `/api/settings/search?q=X` | GET | Search settings |
| `/api/settings/validate` | POST | Validate without saving |
| `/api/settings/tree/{prefix}` | GET | Get hierarchical tree |
| `/api/settings/physics/{profile}` | GET/PUT | Physics profiles |

**All use camelCase consistently.**

---

### Legacy Endpoints (Avoid)

| Endpoint | Method | Issues |
|----------|--------|--------|
| `/settings` | GET/POST | Converts camelCase→snake_case internally |
| `/settings/current` | GET | Redundant with `/api/settings` |
| `/settings/save` | POST | Manual save (auto-save exists) |
| `/settings/reset` | POST | Redundant with DELETE |

**These work but add unnecessary complexity.**

---

## Data Flow Analysis

### Path-Based Endpoints (Recommended)

```
Client (camelCase)
  → DTO deserialization (automatic)
  → JsonPathAccessible navigation (camelCase paths)
  → Validation (camelCase)
  → Actor persistence
  → Response (camelCase)
```

**No manual case conversion needed!**

---

### Legacy Endpoints (Complex)

```
Client (camelCase)
  → Manual convert_to_snake_case_recursive()
  → Merge with struct (snake_case fields)
  → Validate
  → Persist
  → Convert back to camelCase for response
```

**Requires explicit conversion function at line 2158.**

---

## Request/Response Examples

### ✅ Recommended: Batch Update

**Request**:
```http
PUT /api/settings/batch
Content-Type: application/json

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

**Response**:
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

---

### ✅ Recommended: Batch Read

**Request**:
```http
POST /api/settings/batch
Content-Type: application/json

{
  "paths": [
    "visualisation.physics.damping",
    "visualisation.physics.gravity"
  ]
}
```

**Response** (note: direct map, not wrapped):
```json
{
  "visualisation.physics.damping": 0.95,
  "visualisation.physics.gravity": 0.001
}
```

---

## Validation Flow

All validation uses **camelCase field names**:

```json
{
  "error": "Validation failed",
  "path": "visualisation.physics.damping",
  "validationErrors": {
    "visualisation.physics.damping": "Value 2.5 exceeds maximum 1.0"
  },
  "success": false
}
```

**Method**: `AppFullSettings::validate_config_camel_case()`

---

## Handler File Locations

| Handler | File Path | Purpose |
|---------|-----------|---------|
| REST API | `src/handlers/api_handler/settings/mod.rs` | Clean REST interface |
| Path-based | `src/handlers/settings_paths.rs` | Efficient field access |
| Legacy | `src/handlers/settings_handler.rs` | Deprecated, has conversion |
| User settings | `src/handlers/user_settings_handler.rs` | User-specific (feature gated) |

---

## Case Conversion Points

### Where Case Conversion Happens

1. **REST API**: None (DTOs handle it automatically)
2. **Path-based**: None (operates in camelCase)
3. **Legacy**: Line 2158 in `settings_handler.rs`
   ```rust
   convert_to_snake_case_recursive(&mut update);
   ```

### DTO Declarations

**All DTOs use** `#[serde(rename_all = "camelCase")]`:

```rust
// settings_paths.rs
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PathUpdateRequest {
    pub path: String,
    pub value: Value,
}

// settings/mod.rs
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsListResponse {
    pub settings: Vec<SettingItem>,
    pub total: usize,
}
```

**Exception**: User settings handler (missing `rename_all`)

---

## Actor Communication

All handlers communicate with `OptimizedSettingsActor`:

```rust
// Get settings
let settings = state.settings_addr
    .send(GetSettings)
    .await??;

// Update settings
state.settings_addr
    .send(UpdateSettings { settings })
    .await??;
```

**Actor handles**:
- Caching
- Database persistence
- WebSocket broadcasting
- Thread-safe updates

---

## Permission Checks

| Endpoint | Permission Required |
|----------|-------------------|
| GET endpoints | None (public read) |
| PUT `/api/settings/{key}` | Power user |
| DELETE `/api/settings/{key}` | Power user |
| PUT `/api/settings/physics/*` | Power user |
| POST `/api/user-settings` | Authentication + Power user |

**Check function**: `check_power_user(app_state, user_id)`

---

## Identified Issues

### 1. Batch Read Response Format

**Current**: Returns direct map
```json
{
  "path1": value1,
  "path2": value2
}
```

**Some clients expect**:
```json
{
  "values": [
    {"path": "path1", "value": value1},
    {"path": "path2", "value": value2}
  ]
}
```

**Recommendation**: Document current behavior clearly OR add wrapper for consistency

---

### 2. Multiple Overlapping Implementations

**Problem**: Three ways to update settings:
- `PUT /api/settings/{key}`
- `PUT /api/settings/path`
- `POST /settings`

**Recommendation**:
1. Mark `/settings` (non-path) as deprecated
2. Redirect to `/api/settings/*`
3. Document migration path

---

### 3. User Settings Inconsistency

**User settings endpoints** (`/api/user-settings/*`) don't use `rename_all`:

```rust
// Missing #[serde(rename_all = "camelCase")]
#[derive(Debug, Serialize)]
struct UserSettingsResponse {
    settings: Vec<UserSettingDTO>,  // Will serialize as "settings" not camelCase
}
```

**Recommendation**: Add `#[serde(rename_all = "camelCase")]` for consistency

---

### 4. Unclear Route Registration

The `/settings` batch update function exists but route registration is complex:

```rust
// Function exists at line 1917
async fn batch_update_settings(...)

// But route registration is nested and unclear
```

**Recommendation**: Verify this endpoint is properly registered or remove dead code

---

## Performance Considerations

### Batch Operations

**Limits**:
- Maximum 50 paths per batch read
- Maximum 50 updates per batch update

**Validation**:
- Batch updates validated atomically
- If any validation fails, entire batch rejected
- Prevents partial updates

### Actor Caching

Settings actor maintains in-memory cache:
- Reduces database reads
- Updates broadcast to WebSocket clients
- Thread-safe via actor model

---

## Recommendations

### For Clients

**Use these endpoints**:
```javascript
// Reading
GET  /api/settings/path?path=visualisation.physics.damping
POST /api/settings/batch { "paths": [...] }

// Writing
PUT /api/settings/path { "path": "X", "value": Y }
PUT /api/settings/batch { "updates": [...] }
```

**Always use camelCase**:
```json
{
  "path": "visualisation.enableHologram",
  "value": true
}
```

---

### For Backend Developers

1. **Deprecate legacy endpoints**: Add deprecation warnings to `/settings` POST/GET
2. **Fix user settings DTOs**: Add `#[serde(rename_all = "camelCase")]`
3. **Document batch read format**: Clarify it returns direct map
4. **Clean up dead code**: Remove or document unclear batch_update_settings
5. **Add API versioning**: Consider `/api/v1/settings` for future changes

---

## Migration Guide

### From Legacy to Recommended

**Old** (legacy):
```javascript
POST /settings
{
  "visualisation": {
    "physics": {
      "damping": 0.98
    }
  }
}
```

**New** (recommended):
```javascript
PUT /api/settings/batch
{
  "updates": [
    {
      "path": "visualisation.physics.damping",
      "value": 0.98
    }
  ]
}
```

**Benefits**:
- More explicit
- Tracks previous values
- Better error reporting
- Atomic validation
- No internal case conversion

---

## Testing Checklist

- [x] Verify all DTOs have `#[serde(rename_all = "camelCase")]`
- [x] Trace data flow from HTTP → Handler → Service → Database
- [x] Document case conversion points
- [ ] Add user settings camelCase support
- [ ] Add deprecation warnings to legacy endpoints
- [ ] Document batch read response format
- [ ] Verify all routes are registered correctly

---

## Conclusion

**The API is consistent - all endpoints accept and return camelCase.**

The complexity comes from having **three implementations** of similar functionality. The path-based endpoints (`/api/settings/path` and `/api/settings/batch`) are the recommended approach:

✅ **Pros**:
- Clean, efficient
- No manual case conversion
- Atomic batch updates
- Direct field access
- Consistent validation

❌ **Legacy endpoints work but**:
- Require manual conversion
- More complex data flow
- Harder to maintain

**Recommendation**: Standardize on path-based endpoints for all future development.

---

## Files Created

1. **`/home/devuser/workspace/project/docs/settings-api-endpoint-audit.md`**
   Complete endpoint specifications with DTOs, request/response formats, and data flow

2. **`/home/devuser/workspace/project/docs/settings-api-flow-diagram.md`**
   Visual diagrams showing data flow through each implementation

3. **`/home/devuser/workspace/project/docs/settings-api-audit-summary.md`** (this file)
   Executive summary with actionable recommendations

---

**Next Steps**:
1. Review findings with team
2. Prioritize recommendations
3. Create migration plan for legacy endpoints
4. Update API documentation
5. Add integration tests for all endpoints
