# Settings Flow Diagram - Visual Reference

## Complete Request Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER (TypeScript)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  settingsStore.ts:279                                                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ const essentialSettings =                                        │  │
│  │   await settingsApi.getSettingsByPaths(ESSENTIAL_PATHS);        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                          ↓                                              │
│  settingsApi.ts:269-280                                                 │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ async getSettingsByPaths(paths: string[]) {                     │  │
│  │   return await settingsCacheClient.getBatch(paths, options);    │  │
│  │ }                                                                │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                          ↓                                              │
│  SettingsCacheClient.ts:204-263                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ public async getBatch(paths: string[]) {                        │  │
│  │   // Check cache first                                          │  │
│  │   if (cached && valid) { return cached.value; }                 │  │
│  │                                                                  │  │
│  │   // Fetch from server                                          │  │
│  │   const response = await fetch('/api/settings/batch', {        │  │
│  │     method: 'POST',                                             │  │
│  │     body: JSON.stringify({ paths: uncachedPaths })             │  │
│  │   });                                                           │  │
│  │                                                                  │  │
│  │   return batchResult; // { path: value, ... }                  │  │
│  │ }                                                                │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                          ↓                                              │
└─────────────────────────────────────────────────────────────────────────┘
                           ↓
           ┌───────────────────────────────┐
           │   HTTP POST /api/settings/batch  │
           │                               │
           │   Body:                       │
           │   {                           │
           │     "paths": [                │
           │       "system.debug.enabled", │
           │       "auth.enabled"          │
           │     ]                         │
           │   }                           │
           └───────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      API HANDLER LAYER (Rust)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  settings_paths.rs:241-300                                              │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ pub async fn batch_read_settings_by_path(                      │  │
│  │     body: web::Json<BatchPathReadRequest>,                     │  │
│  │     state: web::Data<AppState>                                 │  │
│  │ ) -> ActixResult<HttpResponse> {                               │  │
│  │                                                                  │  │
│  │   // Validate batch size (max 50 paths)                        │  │
│  │   if body.paths.len() > 50 { return BadRequest; }             │  │
│  │                                                                  │  │
│  │   // Get settings from Actor                                   │  │
│  │   match state.settings_addr.send(GetSettings).await {         │  │
│  │     Ok(Ok(settings)) => {                                      │  │
│  │       let mut results = Map::new();                            │  │
│  │                                                                  │  │
│  │       for path in &body.paths {                                │  │
│  │         if let Ok(value) = settings.get_json_by_path(path) {  │  │
│  │           results.insert(path.clone(), value);                 │  │
│  │         } else {                                                │  │
│  │           // ❌ Returns null for missing paths                │  │
│  │           results.insert(path.clone(), Value::Null);          │  │
│  │         }                                                       │  │
│  │       }                                                         │  │
│  │                                                                  │  │
│  │       Ok(HttpResponse::Ok().json(results))                     │  │
│  │     }                                                           │  │
│  │   }                                                             │  │
│  │ }                                                               │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                          ↓                                              │
└─────────────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    SETTINGS SERVICE LAYER (Rust)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  settings_service.rs:59-92                                              │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ pub async fn get_setting(&self, key: &str)                     │  │
│  │     -> Result<Option<SettingValue>, String> {                  │  │
│  │                                                                  │  │
│  │   // 1. Normalize key (camelCase → snake_case)                │  │
│  │   let normalized_key = self.normalize_key(key);               │  │
│  │   // "systemDebugEnabled" → "system_debug_enabled"            │  │
│  │                                                                  │  │
│  │   // 2. Check in-memory cache (5 min TTL)                     │  │
│  │   if let Some(cached) = cache.get(&normalized_key) {          │  │
│  │     if cached.timestamp.elapsed() < 300s {                    │  │
│  │       return Ok(Some(cached.value));                          │  │
│  │     }                                                           │  │
│  │   }                                                             │  │
│  │                                                                  │  │
│  │   // 3. Query database with fallback                          │  │
│  │   match self.db.get_setting(&normalized_key) {                │  │
│  │     Ok(Some(value)) => {                                       │  │
│  │       // Update cache                                          │  │
│  │       cache.insert(normalized_key, value);                    │  │
│  │       Ok(Some(value))                                          │  │
│  │     }                                                           │  │
│  │     Ok(None) => Ok(None), // ❌ Not found in DB              │  │
│  │     Err(e) => Err(format!("Database error: {}", e))           │  │
│  │   }                                                             │  │
│  │ }                                                               │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                          ↓                                              │
└─────────────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                     DATABASE SERVICE LAYER (Rust)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  database_service.rs:130-146                                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ pub fn get_setting(&self, key: &str)                           │  │
│  │     -> SqliteResult<Option<SettingValue>> {                    │  │
│  │                                                                  │  │
│  │   // 1. Try exact match first                                  │  │
│  │   if let Some(value) = self.get_setting_exact(key)? {         │  │
│  │     return Ok(Some(value));                                    │  │
│  │   }                                                             │  │
│  │                                                                  │  │
│  │   // 2. If key has underscore, try camelCase fallback         │  │
│  │   if key.contains('_') {                                       │  │
│  │     let camel_key = Self::to_camel_case(key);                 │  │
│  │     // "spring_k" → "springK"                                 │  │
│  │     if let Some(value) = self.get_setting_exact(&camel_key)? {│  │
│  │       return Ok(Some(value));                                  │  │
│  │     }                                                           │  │
│  │   }                                                             │  │
│  │                                                                  │  │
│  │   // 3. Not found                                              │  │
│  │   Ok(None) // ❌ Returns None when DB is empty               │  │
│  │ }                                                               │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                          ↓                                              │
│  database_service.rs:96-118                                             │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ fn get_setting_exact(&self, key: &str)                        │  │
│  │     -> SqliteResult<Option<SettingValue>> {                    │  │
│  │                                                                  │  │
│  │   let conn = self.conn.lock().unwrap();                        │  │
│  │   let mut stmt = conn.prepare("                                │  │
│  │     SELECT value_type, value_text, value_integer,             │  │
│  │            value_float, value_boolean, value_json             │  │
│  │     FROM settings                                              │  │
│  │     WHERE key = ?1                                             │  │
│  │   ")?;                                                          │  │
│  │                                                                  │  │
│  │   stmt.query_row(params![key], |row| {                        │  │
│  │     let value_type: String = row.get(0)?;                     │  │
│  │     match value_type.as_str() {                                │  │
│  │       "string" => SettingValue::String(row.get(1)?),          │  │
│  │       "integer" => SettingValue::Integer(row.get(2)?),        │  │
│  │       "float" => SettingValue::Float(row.get(3)?),            │  │
│  │       "boolean" => SettingValue::Boolean(                     │  │
│  │                      row.get::<_, i32>(4)? == 1),             │  │
│  │       "json" => {                                              │  │
│  │         let json_str: String = row.get(5)?;                   │  │
│  │         SettingValue::Json(                                    │  │
│  │           serde_json::from_str(&json_str)                     │  │
│  │             .unwrap_or(Value::Null)                           │  │
│  │         )                                                      │  │
│  │       },                                                       │  │
│  │       _ => SettingValue::String(String::new()),               │  │
│  │     }                                                          │  │
│  │   }).optional() // Returns None if no rows found              │  │
│  │ }                                                               │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                          ↓                                              │
└─────────────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         SQLITE DATABASE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  /home/devuser/workspace/project/data/ontology.db                       │
│                                                                         │
│  File Size: 0 bytes  ❌ EMPTY DATABASE                                 │
│                                                                         │
│  Expected Schema:                                                       │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ CREATE TABLE settings (                                         │  │
│  │   id INTEGER PRIMARY KEY,                                       │  │
│  │   key TEXT NOT NULL UNIQUE,                                     │  │
│  │   value_type TEXT CHECK(value_type IN                          │  │
│  │     ('string','integer','float','boolean','json')),            │  │
│  │   value_text TEXT,                                              │  │
│  │   value_integer INTEGER,                                        │  │
│  │   value_float REAL,                                             │  │
│  │   value_boolean INTEGER CHECK(value_boolean IN (0, 1)),        │  │
│  │   value_json TEXT,                                              │  │
│  │   description TEXT,                                             │  │
│  │   created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,              │  │
│  │   updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP               │  │
│  │ );                                                              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  Actual Query Result:                                                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ SELECT ... FROM settings WHERE key = 'system.debug.enabled';   │  │
│  │                                                                  │  │
│  │ Result: (no rows) ❌                                            │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                           ↓
           ┌───────────────────────────────┐
           │   Response Path (Current)     │
           │                               │
           │   200 OK                      │
           │   {                           │
           │     "system.debug.enabled": null,  ❌
           │     "auth.enabled": null      ❌
           │   }                           │
           └───────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLIENT RECEIVES RESPONSE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  SettingsCacheClient.ts:246-249                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ const batchResult = await response.json();                      │  │
│  │                                                                  │  │
│  │ Object.entries(batchResult).forEach(([path, value]) => {       │  │
│  │   results[path] = value; // ❌ Stores null values              │  │
│  │   this.setCachedValue(path, value); // ❌ Caches nulls         │  │
│  │ });                                                             │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                          ↓                                              │
│  settingsStore.ts:830                                                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ if (loadedPaths.size === ESSENTIAL_PATHS.length) {             │  │
│  │   // ❌ All paths "loaded" but values are null                │  │
│  │   // App thinks initialization succeeded                       │  │
│  │ }                                                               │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                          ↓                                              │
│                    ❌ APPLICATION BROKEN                                │
│                                                                         │
│  • Settings panel shows "No Available" for all values                  │
│  • Physics simulation uses undefined/null parameters                   │
│  • Authentication state is undefined                                   │
│  • WebSocket connection fails (missing config)                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Conversion Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        KEY FORMAT CONVERSIONS                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Client Request Path:                                                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ "system.debug.enabled"                                          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                          ↓ (no conversion)                              │
│  Handler Path:                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ "system.debug.enabled"                                          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                          ↓ normalize_key()                              │
│  Service Normalized:                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ "system.debug.enabled" (unchanged - no camelCase)               │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                          ↓ get_setting()                                │
│  Database Query #1 (Exact):                                             │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ SELECT ... FROM settings WHERE key = 'system.debug.enabled';   │  │
│  │ Result: (no rows) ❌                                            │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                          ↓ to_camel_case() (skip - no underscore)       │
│  Database Query #2 (Fallback):                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ (skipped - key doesn't contain underscore)                      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                          ↓                                              │
│  Final Result:                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Ok(None) ❌ Setting not found                                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                     PHYSICS PARAMETER EXAMPLE                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Client Request Path:                                                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ "visualisation.graphs.logseq.physics.spring_k"                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                          ↓ (no conversion)                              │
│  Handler Path:                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ "visualisation.graphs.logseq.physics.spring_k"                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                          ↓ normalize_key()                              │
│  Service Normalized:                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ "visualisation.graphs.logseq.physics.spring_k" (unchanged)      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                          ↓ get_setting()                                │
│  Database Query #1 (Exact):                                             │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ SELECT ... FROM settings                                        │  │
│  │ WHERE key = 'visualisation.graphs.logseq.physics.spring_k';    │  │
│  │ Result: (no rows) ❌                                            │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                          ↓ to_camel_case() (key has underscore!)        │
│  Database Query #2 (Fallback):                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ // Converts "spring_k" → "springK" in full path                │  │
│  │ SELECT ... FROM settings WHERE                                  │  │
│  │   key = 'visualisation.graphs.logseq.physics.springK';         │  │
│  │ Result: (no rows) ❌ (DB is empty)                             │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                          ↓                                              │
│  Final Result:                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Ok(None) ❌ Setting not found                                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Missing Initialization Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CURRENT STARTUP (BROKEN)                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  main.rs:259-273                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 1. Create database connection                                   │  │
│  │    let db_service = DatabaseService::new(&db_file)?;           │  │
│  │    ✅ SUCCESS                                                   │  │
│  │                                                                  │  │
│  │ 2. Initialize schema                                            │  │
│  │    if let Err(e) = db_service.initialize_schema() {            │  │
│  │      warn!("Schema initialization warning: {}", e);            │  │
│  │    }                                                             │  │
│  │    ⚠️ FAILS SILENTLY (database is 0 bytes)                     │  │
│  │                                                                  │  │
│  │ 3. ❌ MISSING: Seed default settings                           │  │
│  │    (This step doesn't exist!)                                   │  │
│  │                                                                  │  │
│  │ 4. Initialize AppState                                          │  │
│  │    let app_state = AppState::new(...).await?;                  │  │
│  │    ✅ SUCCESS (but settings are empty)                         │  │
│  │                                                                  │  │
│  │ 5. Start HTTP server                                            │  │
│  │    HttpServer::new(...).bind(...).run().await                  │  │
│  │    ✅ SUCCESS (server starts but has no data)                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                          ↓                                              │
│  Client connects and requests essential settings                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ GET /api/settings/batch                                         │  │
│  │ { "paths": ["system.debug.enabled", ...] }                     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                          ↓                                              │
│  Server responds with all nulls                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 200 OK                                                           │  │
│  │ {                                                                │  │
│  │   "system.debug.enabled": null,                                 │  │
│  │   "system.websocket.updateRate": null,                          │  │
│  │   ...all null...                                                │  │
│  │ }                                                                │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                          ↓                                              │
│                    ❌ CLIENT BREAKS                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                     REQUIRED STARTUP (FIXED)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  main.rs (AFTER FIX)                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 1. Create database connection                                   │  │
│  │    let db_service = DatabaseService::new(&db_file)?;           │  │
│  │    ✅ SUCCESS                                                   │  │
│  │                                                                  │  │
│  │ 2. Initialize schema (FATAL on error)                          │  │
│  │    db_service.initialize_schema()                              │  │
│  │      .map_err(|e| fatal!("Schema init failed: {}", e))?;      │  │
│  │    ✅ SUCCESS (tables created)                                 │  │
│  │                                                                  │  │
│  │ 3. ✅ NEW: Check if seeding needed                             │  │
│  │    let needs_seeding = db_service.load_all_settings()?         │  │
│  │      .is_none();                                                │  │
│  │    ✅ Returns true (DB is empty)                               │  │
│  │                                                                  │  │
│  │ 4. ✅ NEW: Seed default settings                               │  │
│  │    if needs_seeding {                                           │  │
│  │      let migration = SettingsMigration::new(db_service);       │  │
│  │      let defaults = AppFullSettings::default();                │  │
│  │      let report = migration.seed_from_struct(&defaults)        │  │
│  │        .await?;                                                 │  │
│  │                                                                  │  │
│  │      info!("Seeded {} settings ({}% coverage)",                │  │
│  │        report.migrated_count,                                   │  │
│  │        report.coverage_percent);                               │  │
│  │    }                                                             │  │
│  │    ✅ SUCCESS (50-100+ settings inserted)                      │  │
│  │                                                                  │  │
│  │ 5. Initialize AppState                                          │  │
│  │    let app_state = AppState::new(...).await?;                  │  │
│  │    ✅ SUCCESS (settings loaded from DB)                        │  │
│  │                                                                  │  │
│  │ 6. Start HTTP server                                            │  │
│  │    HttpServer::new(...).bind(...).run().await                  │  │
│  │    ✅ SUCCESS (server ready with data)                         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                          ↓                                              │
│  Client connects and requests essential settings                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ GET /api/settings/batch                                         │  │
│  │ { "paths": ["system.debug.enabled", ...] }                     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                          ↓                                              │
│  Server responds with actual values                                     │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 200 OK                                                           │  │
│  │ {                                                                │  │
│  │   "system.debug.enabled": false,                                │  │
│  │   "system.websocket.updateRate": 100,                           │  │
│  │   "auth.enabled": false,                                        │  │
│  │   ...all valid values...                                        │  │
│  │ }                                                                │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                          ↓                                              │
│                    ✅ CLIENT WORKS                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Summary

The entire settings infrastructure is **architecturally correct** but has **one critical missing step** in the startup sequence:

**Root Cause**: Database seeding never happens
**Impact**: All settings queries return null
**Fix**: Add 20 lines of code in main.rs to seed defaults on first run
**Time**: 15-30 minutes to implement and test
