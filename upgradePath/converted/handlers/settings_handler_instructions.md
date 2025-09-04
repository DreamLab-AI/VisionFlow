# Settings Handler Instructions

## File: `src/handlers/settings_handler.rs`

### Purpose
Unified Settings Handler serving as the single source of truth for AppFullSettings. Provides comprehensive configuration management for the VisionFlow application with rate limiting, validation, and WebSocket real-time updates.

### Key Components

#### 1. Settings Response DTO
```rust
// Converts internal settings to camelCase JSON for frontend
#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SettingsResponseDTO {
    pub visualisation: VisualisationSettingsDTO,
    pub system: SystemSettingsDTO,
    pub xr: XRSettingsDTO,
    pub auth: AuthSettingsDTO,
    // Optional service configurations
    pub ragflow: Option<RagFlowSettingsDTO>,
    pub perplexity: Option<PerplexitySettingsDTO>,
    pub openai: Option<OpenAISettingsDTO>,
    pub kokoro: Option<KokoroSettingsDTO>,
    pub whisper: Option<WhisperSettingsDTO>,
}
```

#### 2. Settings Update DTO
```rust
// Handles partial updates with camelCase input
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SettingsUpdateDTO {
    // All fields optional for partial updates
    pub visualisation: Option<VisualisationSettingsDTO>,
    pub system: Option<SystemSettingsDTO>,
    // ... other optional fields
}
```

### Implementation Instructions

#### GET Settings Endpoint
1. **Rate Limiting**: Apply endpoint-specific rate limits using `RateLimiter`
2. **Actor Communication**: Send `GetSettings` message to configuration actor
3. **Response Transformation**: Convert internal `AppFullSettings` to `SettingsResponseDTO`
4. **Error Handling**: Return appropriate HTTP status codes with detailed error messages

#### POST Settings Update Endpoint
1. **Request Validation**: 
   - Check request size against `MAX_REQUEST_SIZE`
   - Validate JSON structure using `ValidationService`
2. **Selective Updates**: Apply only provided fields, preserving existing values
3. **Actor Updates**: Send `UpdateSettings` message to configuration actor
4. **WebSocket Broadcast**: Notify connected clients of setting changes
5. **Response**: Return updated settings in standardized format

#### WebSocket Integration
1. **Real-time Updates**: Broadcast setting changes to all connected WebSocket clients
2. **Subscription Management**: Allow clients to subscribe to specific setting categories
3. **Connection Tracking**: Maintain client connection registry for targeted updates

### Settings Categories

#### Visualization Settings
- **Rendering**: Light settings, shadows, antialiasing
- **Animations**: Speed, transitions, effects
- **Glow Effects**: Intensity, color, bloom
- **Hologram**: Transparency, shimmer effects
- **Graphs**: Layout algorithms, clustering parameters

#### System Settings
- **Performance**: Update rates, memory limits
- **Logging**: Levels, file rotation
- **Security**: Authentication, session management

#### XR Settings
- **Quest 3 Integration**: Hand tracking, spatial anchors
- **Camera Settings**: FOV, movement controls
- **Space Pilot**: Navigation, interaction modes

### Error Handling Patterns

1. **Validation Errors**: Return 400 with specific field errors
2. **Rate Limit Exceeded**: Return 429 with retry information
3. **Actor Communication Failures**: Return 500 with fallback behavior
4. **Malformed JSON**: Return 422 with parsing error details

### Security Considerations

1. **Input Sanitization**: Validate all input parameters
2. **Rate Limiting**: Prevent abuse of settings endpoints
3. **Authentication**: Verify user permissions for sensitive settings
4. **Audit Logging**: Log all setting changes for security tracking

### Performance Optimizations

1. **Caching**: Cache frequently accessed settings
2. **Partial Updates**: Only update changed fields
3. **Async Processing**: Use actors for non-blocking operations
4. **Connection Pooling**: Efficient WebSocket connection management

### Testing Requirements

1. **Unit Tests**: Each DTO serialization/deserialization
2. **Integration Tests**: Full endpoint behavior with rate limiting
3. **WebSocket Tests**: Real-time update propagation
4. **Performance Tests**: Load testing with concurrent requests
5. **Security Tests**: Input validation and rate limiting effectiveness