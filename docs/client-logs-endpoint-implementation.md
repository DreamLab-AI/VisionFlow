# Client Logs Endpoint Implementation

## Overview
Implemented a simplified `/api/client-logs-simple` endpoint for frontend error logging that forwards client-side logs directly to the server logger.

## Files Created/Modified

### 1. Handler Implementation
**File**: `/home/devuser/workspace/project/src/handlers/client_logs.rs`

```rust
use actix_web::{web, HttpResponse, Error};
use serde::Deserialize;
use log::{info, warn, error, debug};

#[derive(Deserialize)]
pub struct ClientLogEntry {
    level: String,
    message: String,
    timestamp: Option<String>,
    context: Option<String>,
}

/// Simple handler to receive client-side logs and forward to server logger
pub async fn post_client_logs(
    web::Json(payload): web::Json<ClientLogEntry>
) -> Result<HttpResponse, Error> {
    let prefix = "[CLIENT]";
    let msg = if let Some(ctx) = payload.context {
        format!("{} {}", payload.message, ctx)
    } else {
        payload.message
    };

    match payload.level.to_lowercase().as_str() {
        "error" => error!("{} {}", prefix, msg),
        "warn" => warn!("{} {}", prefix, msg),
        "debug" => debug!("{} {}", prefix, msg),
        _ => info!("{} {}", prefix, msg),
    }

    Ok(HttpResponse::Ok().json(serde_json::json!({"success": true})))
}
```

### 2. Module Registration
**File**: `/home/devuser/workspace/project/src/handlers/mod.rs`

Added:
```rust
pub mod client_logs; // Simplified client logs handler
```

### 3. Route Registration
**File**: `/home/devuser/workspace/project/src/main.rs`

**Import added (line 18)**:
```rust
client_logs,
```

**Route registered (line 626)**:
```rust
.route("/client-logs-simple", web::post().to(client_logs::post_client_logs))
```

## API Endpoint

### POST /api/client-logs-simple

**Purpose**: Receive client-side logs and forward to server logger

**Request Body**:
```json
{
  "level": "error|warn|info|debug",
  "message": "Log message",
  "timestamp": "ISO8601 timestamp (optional)",
  "context": "Additional context (optional)"
}
```

**Response**:
```json
{
  "success": true
}
```

**Example Usage**:
```bash
curl -X POST http://localhost:4000/api/client-logs-simple \
  -H "Content-Type: application/json" \
  -d '{
    "level": "error",
    "message": "Frontend error occurred",
    "context": "Component: LoginForm",
    "timestamp": "2025-10-21T10:30:00Z"
  }'
```

## Differences from Existing `/api/client-logs`

The existing `/api/client-logs` endpoint (handled by `client_log_handler::handle_client_logs`) provides:
- Session ID tracking
- Correlation ID management
- Telemetry logging
- File-based logging to `/app/logs/client.log`
- Stack trace handling
- User agent and URL tracking

The new `/api/client-logs-simple` endpoint provides:
- **Simple forwarding to server logger only**
- Minimal overhead
- No file I/O
- No session tracking
- No telemetry integration

## Log Output Format

Logs appear in the server console with `[CLIENT]` prefix:
```
[2025-10-21T10:30:00Z ERROR] [CLIENT] Frontend error occurred Component: LoginForm
[2025-10-21T10:30:01Z WARN] [CLIENT] Performance warning
[2025-10-21T10:30:02Z INFO] [CLIENT] User action completed
[2025-10-21T10:30:03Z DEBUG] [CLIENT] Debug information
```

## Integration Notes

- Both endpoints can coexist
- Use `/api/client-logs-simple` for lightweight logging
- Use `/api/client-logs` for comprehensive telemetry and session tracking
- All logs follow Rust `log` crate conventions and respect `RUST_LOG` environment variable

## Testing

To test the endpoint after deployment:

```javascript
// Frontend JavaScript
fetch('/api/client-logs-simple', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    level: 'error',
    message: 'Test error from frontend',
    context: 'Component: TestComponent'
  })
});
```

## Dependencies

- `actix-web`: Web framework
- `serde`: JSON deserialization
- `log`: Logging facade

No additional dependencies required beyond existing project dependencies.
