# Settings API Data Flow Diagrams

Visual representation of how data flows through the settings API endpoints.

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CLIENT (Browser/App)                     │
│                  Always sends camelCase JSON                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       HTTP LAYER                             │
│  ┌──────────────┬──────────────┬─────────────────────────┐  │
│  │  /api/       │  /settings/  │  /api/user-settings/   │  │
│  │  settings/*  │  (legacy)    │  (feature gated)       │  │
│  └──────────────┴──────────────┴─────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
           │                │                    │
           ▼                ▼                    ▼
┌─────────────────┬─────────────────┬──────────────────────┐
│   REST API      │  Legacy Handler │   User Service       │
│   (camelCase)   │  (converts)     │   (snake_case)       │
└─────────────────┴─────────────────┴──────────────────────┘
           │                │                    │
           ▼                ▼                    ▼
┌─────────────────┬─────────────────┬──────────────────────┐
│ SettingsService │ SettingsActor   │  DatabaseService     │
└─────────────────┴─────────────────┴──────────────────────┘
           │                │                    │
           ▼                ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                     PERSISTENCE LAYER                        │
│          database.db (SQLite with snake_case columns)       │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. REST API Path Flow (`/api/settings/path`)

### Request Processing

```
┌─────────────────────────────────────────────────────────┐
│ Client sends PUT /api/settings/path                     │
│ {                                                        │
│   "path": "visualisation.physics.damping",  ◄── camelCase
│   "value": 0.98                                         │
│ }                                                        │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Actix-Web deserializes into PathUpdateRequest          │
│ #[serde(rename_all = "camelCase")]                     │
│ struct PathUpdateRequest {                              │
│   pub path: String,    ◄── "visualisation.physics..."  │
│   pub value: Value,    ◄── 0.98                        │
│ }                                                        │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Handler: update_settings_by_path()                      │
│                                                          │
│ 1. Validate path not empty                             │
│ 2. Send GetSettings to actor                           │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ SettingsActor returns AppFullSettings                   │
│ (struct with camelCase field names via serde)          │
│                                                          │
│ pub struct AppFullSettings {                            │
│   pub visualisation: VisualisationSettings, ◄── camelCase
│   ...                                                   │
│ }                                                        │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Get previous value:                                      │
│ settings.get_json_by_path("visualisation.physics...")   │
│                                                          │
│ JsonPathAccessible trait navigates:                     │
│   settings                                              │
│     └─ visualisation    ◄── Match "visualisation"      │
│         └─ physics      ◄── Match "physics"            │
│             └─ damping  ◄── Match "damping"            │
│                 └─ Returns: 0.95                        │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Set new value:                                           │
│ settings.set_json_by_path("visualisation.physics...", 0.98)
│                                                          │
│ Updates the struct field directly:                      │
│   settings.visualisation.physics.damping = 0.98         │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Validation:                                              │
│ settings.validate_config_camel_case()                   │
│                                                          │
│ Checks:                                                 │
│ ✓ Type correctness (is 0.98 a valid f64?)              │
│ ✓ Range validation (is 0.98 within allowed range?)     │
│ ✓ Field existence (does "damping" exist?)              │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ If validation passes:                                    │
│ state.settings_addr.send(UpdateSettings { settings })   │
│                                                          │
│ Actor persists to database:                             │
│   INSERT/UPDATE settings                                │
│   SET value_json = '{"damping": 0.98, ...}'            │
│   WHERE key = 'visualisation.physics'                   │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Response (camelCase):                                    │
│ {                                                        │
│   "success": true,                                      │
│   "path": "visualisation.physics.damping",              │
│   "value": 0.98,                                        │
│   "previousValue": 0.95,                                │
│   "message": "Settings updated successfully"            │
│ }                                                        │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Batch Update Flow (`/api/settings/batch`)

### Request Processing

```
┌─────────────────────────────────────────────────────────┐
│ Client sends PUT /api/settings/batch                     │
│ {                                                        │
│   "updates": [                                          │
│     { "path": "visualisation.physics.damping",          │
│       "value": 0.98 },                                  │
│     { "path": "visualisation.physics.gravity",          │
│       "value": 0.002 }                                  │
│   ]                                                      │
│ }                                                        │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Deserialize into BatchPathUpdateRequest                 │
│ struct BatchPathUpdateRequest {                         │
│   pub updates: Vec<PathUpdateRequest>                   │
│ }                                                        │
│                                                          │
│ Each PathUpdateRequest has:                             │
│   path: String (camelCase)                              │
│   value: Value                                          │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Validation:                                              │
│ ✓ Updates not empty                                     │
│ ✓ Updates.len() <= 50 (prevent abuse)                  │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Get current settings from actor                          │
│ let mut settings = state.settings_addr.send(GetSettings)│
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Loop through each update:                                │
│                                                          │
│ for update in updates {                                 │
│   ┌─────────────────────────────────────────────────┐  │
│   │ 1. Get previous value:                          │  │
│   │    settings.get_json_by_path(update.path)       │  │
│   │                                                  │  │
│   │ 2. Apply update:                                │  │
│   │    settings.set_json_by_path(path, value)       │  │
│   │                                                  │  │
│   │ 3. Track result:                                │  │
│   │    results.push({                               │  │
│   │      "path": path,                              │  │
│   │      "success": true/false,                     │  │
│   │      "value": value,                            │  │
│   │      "previousValue": previous                  │  │
│   │    })                                           │  │
│   └─────────────────────────────────────────────────┘  │
│ }                                                        │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ If any update failed:                                    │
│   Return 400 BadRequest with partial results            │
│                                                          │
│ {                                                        │
│   "success": false,                                     │
│   "message": "Some updates failed",                     │
│   "results": [...]                                      │
│ }                                                        │
└─────────────────────────────────────────────────────────┘
                       │ (if all succeeded)
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Validate entire settings object:                         │
│ settings.validate_config_camel_case()                   │
│                                                          │
│ This ensures all fields are still valid after batch     │
│ update (atomic validation)                              │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ If validation fails:                                     │
│   Return 400 BadRequest with validation errors          │
│                                                          │
│ {                                                        │
│   "error": "Validation failed after batch update",      │
│   "validationErrors": [                                 │
│     "visualisation.physics.damping: out of range"       │
│   ],                                                     │
│   "success": false                                      │
│ }                                                        │
└─────────────────────────────────────────────────────────┘
                       │ (if valid)
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Persist to database:                                     │
│ state.settings_addr.send(UpdateSettings { settings })   │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Success response:                                        │
│ {                                                        │
│   "success": true,                                      │
│   "message": "Successfully updated 2 settings",         │
│   "results": [                                          │
│     {                                                    │
│       "path": "visualisation.physics.damping",          │
│       "success": true,                                  │
│       "value": 0.98,                                    │
│       "previousValue": 0.95                             │
│     },                                                   │
│     { ... }                                             │
│   ]                                                      │
│ }                                                        │
└─────────────────────────────────────────────────────────┘
```

**Key Points**:
- **Atomic validation**: All updates applied first, then validated together
- **Partial success tracking**: Each update tracked individually
- **Error handling**: Any failure stops processing and returns error
- **Rate limiting**: Maximum 50 updates per batch

---

## 4. Legacy Handler Flow (`POST /settings`)

### The Conversion Pipeline

```
┌─────────────────────────────────────────────────────────┐
│ Client sends POST /settings                              │
│ {                                                        │
│   "visualisation": {           ◄── camelCase from client
│     "enableHologram": true,                             │
│     "hologramSettings": {                               │
│       "ringCount": 5                                    │
│     }                                                    │
│   }                                                      │
│ }                                                        │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Deserialized as raw Value (no DTO)                      │
│ let mut update: Value = payload.into_inner();           │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ CASE CONVERSION (Line 2158):                            │
│ convert_to_snake_case_recursive(&mut update);           │
│                                                          │
│ Recursively walks JSON and converts all keys:           │
│   "visualisation"     → "visualisation" (unchanged)     │
│   "enableHologram"    → "enable_hologram"               │
│   "hologramSettings"  → "hologram_settings"             │
│   "ringCount"         → "ring_count"                    │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ After conversion:                                        │
│ {                                                        │
│   "visualisation": {           ◄── Now all snake_case   │
│     "enable_hologram": true,                            │
│     "hologram_settings": {                              │
│       "ring_count": 5                                   │
│     }                                                    │
│   }                                                      │
│ }                                                        │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Validate converted JSON:                                 │
│ validate_settings_update(&update)                       │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Get current settings (has snake_case fields):            │
│ let mut app_settings = state.settings_addr.send(GetSettings)
│                                                          │
│ pub struct AppFullSettings {                            │
│   #[serde(rename = "visualisation")]                   │
│   pub visualisation: VisualisationSettings,             │
│   ...                                                   │
│ }                                                        │
│                                                          │
│ pub struct VisualisationSettings {                      │
│   pub enable_hologram: bool,        ◄── snake_case     │
│   pub hologram_settings: HologramSettings,              │
│   ...                                                   │
│ }                                                        │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Merge snake_case update into settings:                  │
│                                                          │
│ The snake_case JSON matches struct field names,         │
│ so serde can deserialize properly:                      │
│                                                          │
│   update JSON: "enable_hologram": true                  │
│   struct field: pub enable_hologram: bool               │
│   ✓ Names match, merge succeeds                        │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Persist updated settings                                 │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Response (convert back to camelCase):                    │
│ let response_dto: SettingsResponseDTO = (&app_settings).into();
│                                                          │
│ SettingsResponseDTO has:                                │
│ #[serde(rename_all = "camelCase")]                     │
│                                                          │
│ So serialization produces:                              │
│ {                                                        │
│   "visualisation": {           ◄── Back to camelCase    │
│     "enableHologram": true,                             │
│     "hologramSettings": {                               │
│       "ringCount": 5                                    │
│     }                                                    │
│   }                                                      │
│ }                                                        │
└─────────────────────────────────────────────────────────┘
```

**Why This Works**:
1. Client sends camelCase
2. Server converts to snake_case to match struct fields
3. Merge and validation work with snake_case
4. Response DTO converts back to camelCase for client

**Why This is Complex**:
- Requires explicit conversion function
- Two validation passes (pre and post merge)
- More error-prone than direct DTO deserialization

---

## 5. REST API vs Path-Based Comparison

### REST API Flow (Simple)

```
Client (camelCase)
    ↓
    [DTO with #[serde(rename_all = "camelCase")]]
    ↓
Handler (works with DTO fields)
    ↓
Service layer
    ↓
Database (snake_case columns)
    ↓
Response (camelCase via DTO)
```

**Pros**:
- Simple, declarative
- Serde handles all conversions
- Type-safe DTOs

**Cons**:
- Requires separate endpoint per setting group
- Full object updates only

---

### Path-Based Flow (Flexible)

```
Client (camelCase paths)
    ↓
    [DTO with camelCase path string]
    ↓
Handler (uses JsonPathAccessible)
    ↓
Direct field access via path navigation
    ↓
Validation (camelCase paths)
    ↓
Actor (entire settings object)
    ↓
Database (serialized JSON)
    ↓
Response (camelCase)
```

**Pros**:
- Single endpoint for all fields
- Efficient partial updates
- Batch operations
- Direct path navigation

**Cons**:
- Requires custom JsonPathAccessible implementation
- String-based paths (less type-safe)
- More complex validation

---

## 6. Error Flow

### Validation Error Flow

```
Client sends invalid value
    ↓
Handler receives request
    ↓
┌─────────────────────────────────────────────────────────┐
│ settings.set_json_by_path(path, invalid_value)          │
│                                                          │
│ Example:                                                │
│   path = "visualisation.physics.damping"                │
│   value = 2.5 (out of range 0.0-1.0)                   │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ settings.validate_config_camel_case()                   │
│                                                          │
│ Returns Err(ValidationErrors):                          │
│   [                                                     │
│     ValidationError {                                   │
│       field: "visualisation.physics.damping",           │
│       message: "Value 2.5 exceeds maximum 1.0"         │
│     }                                                   │
│   ]                                                     │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ AppFullSettings::get_validation_errors_camel_case()     │
│                                                          │
│ Converts to client-friendly format:                     │
│   {                                                     │
│     "visualisation.physics.damping":                    │
│       "Value 2.5 exceeds maximum 1.0"                  │
│   }                                                     │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ Handler returns 400 BadRequest:                          │
│ {                                                        │
│   "error": "Validation failed",                         │
│   "path": "visualisation.physics.damping",              │
│   "validationErrors": {                                 │
│     "visualisation.physics.damping":                    │
│       "Value 2.5 exceeds maximum 1.0"                  │
│   },                                                     │
│   "success": false                                      │
│ }                                                        │
└─────────────────────────────────────────────────────────┘
```

**All error messages use camelCase field names!**

---

## 7. Actor Communication Pattern

```
┌─────────────────────────────────────────────────────────┐
│                    HTTP Handler                          │
└─────────────────────────────────────────────────────────┘
                       │
                       │ (1) Send message
                       ▼
┌─────────────────────────────────────────────────────────┐
│ state.settings_addr.send(GetSettings)                    │
│                                                          │
│ Returns: Future<Result<AppFullSettings, String>>        │
└─────────────────────────────────────────────────────────┘
                       │
                       │ (2) Actor processes
                       ▼
┌─────────────────────────────────────────────────────────┐
│             OptimizedSettingsActor                       │
│                                                          │
│ Handle message:                                         │
│   match msg {                                           │
│     GetSettings => {                                    │
│       // Load from cache or database                   │
│       Ok(self.current_settings.clone())                │
│     }                                                   │
│     UpdateSettings { settings } => {                   │
│       // Validate                                      │
│       // Save to database                              │
│       // Update cache                                  │
│       // Broadcast to WebSocket clients                │
│       Ok(())                                           │
│     }                                                   │
│   }                                                     │
└─────────────────────────────────────────────────────────┘
                       │
                       │ (3) Return result
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    HTTP Handler                          │
│                                                          │
│ match result {                                          │
│   Ok(Ok(settings)) => { /* Process settings */ }        │
│   Ok(Err(e)) => { /* Actor error */ }                  │
│   Err(e) => { /* Mailbox error */ }                    │
│ }                                                        │
└─────────────────────────────────────────────────────────┘
```

**Actor Benefits**:
- Centralized settings management
- Thread-safe state updates
- Automatic caching
- WebSocket broadcast integration

---

## 8. Database Schema

```
┌─────────────────────────────────────────────────────────┐
│                  settings table                          │
│                                                          │
│  Column       │ Type    │ Description                   │
│  ────────────┼─────────┼─────────────────────────────  │
│  key         │ TEXT    │ Primary key (snake_case)      │
│  value_type  │ TEXT    │ "string", "integer", etc.     │
│  value_text  │ TEXT    │ For string values             │
│  value_int   │ INTEGER │ For integer values            │
│  value_float │ REAL    │ For float values              │
│  value_bool  │ BOOLEAN │ For boolean values            │
│  value_json  │ TEXT    │ For complex objects/arrays    │
│  updated_at  │ INTEGER │ Unix timestamp                │
└─────────────────────────────────────────────────────────┘

Example rows:

┌─────────────────────────────────────────────────────────┐
│ key: "visualisation_physics"                             │
│ value_type: "json"                                      │
│ value_json: '{                                          │
│   "damping": 0.95,                                      │
│   "gravity": 0.001,                                     │
│   "charge_strength": 150.0                              │
│ }'                                                       │
│ updated_at: 1729512345                                  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ key: "visualisation_enable_hologram"                     │
│ value_type: "boolean"                                   │
│ value_bool: true                                        │
│ updated_at: 1729512345                                  │
└─────────────────────────────────────────────────────────┘
```

**Key Points**:
- Database uses **snake_case** for keys
- Complex objects stored as JSON in `value_json`
- Type information preserved in `value_type`
- Service layer handles conversion to/from AppFullSettings

---

## Summary

**Three Implementation Patterns**:

1. **REST API** (`/api/settings/*`): Clean DTO-based, fully camelCase
2. **Path-based** (`/settings/path`, `/settings/batch`): Direct field access, fully camelCase
3. **Legacy** (`/settings`): Converts camelCase→snake_case→camelCase

**Recommended Usage**:
- Use `/api/settings/path` and `/api/settings/batch` for all new code
- Avoid legacy `/settings` endpoints
- These provide the best balance of simplicity and efficiency

**All public APIs accept and return camelCase consistently!**
