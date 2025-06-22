# Request Handlers Architecture

## Overview
The handler layer manages HTTP and WebSocket endpoints, providing API interfaces for client interactions. All handlers are defined in the `/src/handlers/` directory and exposed through the main module.

## Handler Modules

```rust
// src/handlers/mod.rs
pub mod api_handler;
pub mod health_handler;
pub mod pages_handler;
pub mod perplexity_handler;
pub mod ragflow_handler;
pub mod settings_handler;
pub mod socket_flow_handler;
pub mod speech_socket_handler;
pub mod nostr_handler;
```

## Core Handlers

### API Handler ([`src/handlers/api_handler/mod.rs`](../../src/handlers/api_handler/mod.rs))
This module configures the main REST API routes under the base scope `/api`.

```rust
// In src/handlers/api_handler/mod.rs
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api")
            // Files API routes (e.g., /process, /get_content/{filename})
            .configure(files::config)
            // Graph API routes (e.g., /data, /data/paginated, /update, /refresh)
            .configure(graph::config)
            // Visualisation settings routes (e.g., /settings/{category}, /get_settings/{category})
            .configure(visualisation::config)
            // Nostr authentication routes (e.g., /auth/nostr, /auth/nostr/verify, /auth/nostr/api-keys)
            .service(web::scope("/auth/nostr").configure(crate::handlers::nostr_handler::config))
            // User settings routes (e.g., /user-settings, /user-settings/sync)
            .service(web::scope("/user-settings").configure(crate::handlers::settings_handler::config_public))
            // RAGFlow chat route
            .service(web::scope("/ragflow").configure(crate::handlers::ragflow_handler::config))
            // Health check routes
            .service(web::scope("/health").configure(crate::handlers::health_handler::config))
            // Static pages/assets (if served via /api/pages)
            .service(web::scope("/pages").configure(crate::handlers::pages_handler::config))
            // Perplexity AI service
            .service(web::scope("/perplexity").configure(crate::handlers::perplexity_handler::config))
    );
}
```
-   Organizes REST API endpoints into logical sub-scopes:
    -   `/api/files` - File processing and content retrieval
    -   `/api/graph` - Graph data operations (CRUD, pagination, refresh)
    -   `/api/visualisation` - Visualization settings management
    -   `/api/auth/nostr` - Nostr authentication and authorization
    -   `/api/user-settings` - User settings storage and synchronization
    -   `/api/ragflow` - RAGFlow AI chat integration
    -   `/api/perplexity` - Perplexity AI service integration
-   Handles request validation, deserialization, calls appropriate services, and serializes responses.

### File Handler ([`src/handlers/file_handler.rs`](../../src/handlers/file_handler.rs))
Handles file processing and content retrieval operations.
-   **Key Functions:**
    -   `fetch_and_process_files` - Processes public markdown files and updates graph data
    -   `get_file_content` - Retrieves content of specific files
-   **Features:**
    -   Optimized file processing with metadata caching
    -   Integration with GraphService for automatic graph updates
    -   GPU compute integration for node position calculations

### WebSocket Handler ([`src/handlers/socket_flow_handler.rs`](../../src/handlers/socket_flow_handler.rs))
Manages WebSocket connections for real-time graph data updates.
-   **Path:** `/wss` (or as configured in `main.rs` or `docker-compose.yml` via Nginx proxy)
-   **Function:** `socket_flow_handler(req: HttpRequest, stream: web::Payload, srv: web::Data<Arc<ClientManager>>)`
-   Handles client connections, disconnections, and messages:
    -   `ping` - Keep-alive messages
    -   `requestInitialData` - Initial graph data request
-   Uses the static `APP_CLIENT_MANAGER` to register clients and broadcast graph updates from `GraphService`.
-   Supports binary protocol for efficient position updates

### Health Handler ([`src/handlers/health_handler.rs`](../../src/handlers/health_handler.rs))
Provides endpoints for system health monitoring.
-   **Base Path:** `/api/health`
-   **Endpoints:**
    -   `/api/health` - General health check
    -   `/api/health/physics` - Physics simulation status
-   Reports the status of core services and dependencies
-   Returns service availability and performance metrics

### Pages Handler ([`src/handlers/pages_handler.rs`](../../src/handlers/pages_handler.rs))
Serves static frontend assets and the main `index.html` page.
-   **Base Path:** `/api/pages` (or could be `/` if Nginx proxies static assets differently)
-   Handles routing for client-side application entry points
-   Serves static files using `actix_files`

### Perplexity Handler ([`src/handlers/perplexity_handler.rs`](../../src/handlers/perplexity_handler.rs))
Provides integration with Perplexity AI service.
-   **Base Path:** `/api/perplexity`
-   **Endpoints:**
    -   `POST /api/perplexity` - Send query to Perplexity AI
-   **Request/Response:**
    -   Request: `PerplexityRequest` with `query` and optional `conversation_id`
    -   Response: `PerplexityResponse` with `answer` and `conversation_id`
-   Handles service availability checks and error responses

### RAGFlow Handler ([`src/handlers/ragflow_handler.rs`](../../src/handlers/ragflow_handler.rs))
Manages RAGFlow AI chat service integration.
-   **Base Path:** `/api/ragflow`
-   **Key Functions:**
    -   `send_message` - Send chat messages to RAGFlow service
    -   `create_session` - Create new chat sessions
-   **Features:**
    -   Streaming response support
    -   Optional TTS (Text-to-Speech) integration
    -   Session management with configurable IDs
-   **Request Format:**
    -   `question`: The user's question
    -   `stream`: Enable streaming responses (default: true)
    -   `session_id`: Optional session ID
    -   `enable_tts`: Enable text-to-speech (default: false)

### Nostr Handler ([`src/handlers/nostr_handler.rs`](../../src/handlers/nostr_handler.rs))
Handles Nostr protocol authentication and authorization.
-   **Base Path:** `/api/auth/nostr`
-   **Endpoints:**
    -   `POST /api/auth/nostr` - Login with Nostr
    -   `DELETE /api/auth/nostr` - Logout
    -   `POST /api/auth/nostr/verify` - Verify authentication
    -   `POST /api/auth/nostr/refresh` - Refresh authentication
    -   `POST /api/auth/nostr/api-keys` - Update user API keys
    -   `GET /api/auth/nostr/api-keys` - Get user API keys
    -   `GET /api/auth/nostr/power-user-status` - Check power user status
    -   `GET /api/auth/nostr/features` - Get available features
    -   `GET /api/auth/nostr/features/{feature}` - Check specific feature access
-   **Features:**
    -   Public key based authentication
    -   Feature-based access control
    -   Power user privileges
    -   API key management for external services

### Settings Handler ([`src/handlers/settings_handler.rs`](../../src/handlers/settings_handler.rs))
Manages user and application settings.
-   **Base Path:** `/api/user-settings`
-   **Endpoints:**
    -   `GET /api/user-settings` - Get public settings
    -   `POST /api/user-settings` - Update settings
    -   `GET /api/user-settings/sync` - Get user-specific settings
    -   `POST /api/user-settings/sync` - Update user-specific settings
    -   `POST /api/user-settings/clear-cache` - Clear settings cache
    -   `POST /api/admin/settings/clear-all-cache` - Clear all caches (power users only)
-   **Features:**
    -   Settings synchronization across devices
    -   Cache management for performance
    -   Role-based access control
    -   Conversion between internal and client-facing settings formats

### Speech WebSocket Handler ([`src/handlers/speech_socket_handler.rs`](../../src/handlers/speech_socket_handler.rs))
Manages WebSocket connections specifically for speech-related functionalities (STT/TTS).
-   **Path:** `/speech` (or as configured)
-   **Features:**
    -   Real-time audio streaming
    -   Speech-to-text (STT) processing
    -   Text-to-speech (TTS) generation
    -   Integration with Kokoro voice service
-   Interacts with `SpeechService` to process audio streams and broadcast responses

### Visualization Handler ([`src/handlers/visualization_handler.rs`](../../src/handlers/visualization_handler.rs))
Manages visualization-specific settings and configurations.
-   **Features:**
    -   Category-based settings management
    -   Snake case to camel case conversion for client compatibility
    -   Setting validation and error handling
-   Works with the visualization settings actor for state management

## Middleware Integration

### CORS Configuration
```rust
let cors = Cors::default()
    .allow_any_origin()
    .allow_any_method()
    .allow_any_header()
    .max_age(3600)
    .supports_credentials();
```

### Compression
```rust
.wrap(middleware::Compress::default())
```

### Logging
```rust
.wrap(middleware::Logger::default())
```

## Error Handling

### Request Validation
- Input sanitization
- Parameter validation
- Type checking

### Response Formatting
- Error standardization
- Status codes
- Error messages

## Security

### Authentication
- Token validation
- Session management
- Access control

### Authorization
- Role-based access
- Permission checking
- Resource protection