# API Handler Instructions

## File: `src/handlers/api_handler/mod.rs` and Sub-modules

### Purpose
Centralized API handler module that organizes and configures all REST API endpoints for the VisionFlow application. Provides modular structure for different API categories.

### Module Structure

#### Core Sub-modules
1. **files**: File management and processing
2. **graph**: Graph data and visualization endpoints  
3. **visualisation**: Visualization settings and configuration
4. **bots**: Agent/bot management endpoints
5. **analytics**: Data analytics and clustering
6. **quest3**: Quest 3 XR integration endpoints

### Implementation Instructions

#### Module Configuration Pattern
```rust
// Main configuration function
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("") // No redundant /api prefix
            .configure(files::config)
            .configure(graph::config)
            .configure(visualisation::config)
            .configure(bots::config)
            .configure(analytics::config)
            .configure(quest3::config)
            // External handler integrations
            .configure(crate::handlers::nostr_handler::config)
            .configure(crate::handlers::settings_handler::config)
            .configure(crate::handlers::ragflow_handler::config)
            .configure(crate::handlers::clustering_handler::config)
            .configure(crate::handlers::constraints_handler::config)
    );
}
```

#### Re-export Pattern
```rust
// Export commonly used functions for easy access
pub use files::{
    fetch_and_process_files,
    get_file_content,
};

pub use graph::{
    get_graph_data,
    get_paginated_graph_data, 
    refresh_graph,
    update_graph,
};

pub use visualisation::get_visualisation_settings;
```

### Sub-module Implementation Guidelines

#### Files Module (`api_handler/files/mod.rs`)
**Purpose**: Handle file operations, uploads, downloads, and processing

**Key Endpoints**:
- `GET /files` - List files with pagination
- `POST /files/upload` - File upload endpoint
- `GET /files/{id}/content` - Get file content
- `DELETE /files/{id}` - Delete file
- `POST /files/process` - Process uploaded files

**Implementation Requirements**:
1. **File Validation**: MIME type checking, size limits
2. **Security**: Path traversal prevention, virus scanning
3. **Streaming**: Large file streaming support
4. **Metadata**: File metadata extraction and storage
5. **Caching**: File content caching with TTL

#### Graph Module (`api_handler/graph/mod.rs`)
**Purpose**: Provide graph data for visualization components

**Key Endpoints**:
- `GET /graph` - Get complete graph data
- `GET /graph/paginated` - Paginated graph data
- `POST /graph/refresh` - Force graph refresh
- `PUT /graph/update` - Update graph data
- `GET /graph/metrics` - Graph performance metrics

**Implementation Requirements**:
1. **Data Aggregation**: Combine data from multiple sources
2. **Real-time Updates**: WebSocket integration for live updates  
3. **Filtering**: Agent type, swarm ID, time range filters
4. **Performance**: Efficient data structures, pagination
5. **Caching**: Multi-level caching strategy

#### Visualization Module (`api_handler/visualisation/mod.rs`)
**Purpose**: Manage visualization settings and configuration

**Key Endpoints**:
- `GET /visualisation/settings` - Get current settings
- `PUT /visualisation/settings` - Update settings
- `GET /visualisation/presets` - Get predefined presets
- `POST /visualisation/presets` - Save custom preset

**Implementation Requirements**:
1. **Schema Validation**: Validate settings structure
2. **Real-time Apply**: Apply settings without restart
3. **Presets Management**: Save/load visualization presets
4. **Performance Impact**: Monitor settings impact on performance

#### Bots Module (`api_handler/bots/mod.rs`)
**Purpose**: Agent/bot management and monitoring

**Key Endpoints**:
- `GET /bots` - List all agents with status
- `POST /bots/spawn` - Create new agent
- `GET /bots/{id}` - Get specific agent details
- `PUT /bots/{id}/status` - Update agent status
- `DELETE /bots/{id}` - Terminate agent
- `GET /bots/{id}/metrics` - Agent performance metrics

**Implementation Requirements**:
1. **MCP Integration**: Interface with MCP servers
2. **Real-time Status**: Live agent status updates
3. **Performance Monitoring**: Track agent performance
4. **Lifecycle Management**: Complete agent lifecycle
5. **Error Handling**: Graceful failure handling

#### Analytics Module (`api_handler/analytics/mod.rs`)
**Purpose**: Data analytics and clustering operations

**Key Endpoints**:
- `GET /analytics/overview` - System analytics overview
- `POST /analytics/clustering` - Run clustering analysis
- `GET /analytics/metrics` - Performance metrics
- `GET /analytics/anomalies` - Anomaly detection results

**Implementation Requirements**:
1. **Data Processing**: Efficient data processing pipelines
2. **Clustering Algorithms**: K-means, hierarchical clustering
3. **Anomaly Detection**: Statistical anomaly detection
4. **Visualization Data**: Prepare data for visualization
5. **Background Processing**: Async analytics processing

#### Quest3 Module (`api_handler/quest3/mod.rs`)
**Purpose**: Quest 3 XR integration and spatial computing

**Key Endpoints**:
- `GET /quest3/status` - XR system status
- `POST /quest3/calibrate` - Spatial calibration
- `GET /quest3/anchors` - Spatial anchors
- `PUT /quest3/tracking` - Update tracking settings

**Implementation Requirements**:
1. **XR Integration**: Quest 3 SDK integration
2. **Spatial Computing**: 3D coordinate systems
3. **Hand Tracking**: Hand gesture recognition
4. **Performance Optimization**: VR-specific optimizations
5. **Safety Validation**: XR safety boundary checks

### Cross-cutting Concerns

#### Error Handling Strategy
```rust
// Consistent error response format
#[derive(Serialize)]
pub struct ApiErrorResponse {
    pub error: String,
    pub error_code: String,
    pub timestamp: String,
    pub request_id: Option<String>,
}
```

#### Authentication & Authorization
1. **JWT Validation**: Validate bearer tokens
2. **Role-Based Access**: Check endpoint permissions
3. **Rate Limiting**: Prevent API abuse
4. **Audit Logging**: Log all API access

#### Performance Monitoring
1. **Request Metrics**: Track response times
2. **Resource Usage**: Monitor CPU/memory usage
3. **Cache Hit Rates**: Monitor caching effectiveness
4. **Error Rates**: Track error frequencies

#### Documentation Standards
1. **OpenAPI Spec**: Complete API documentation
2. **Request/Response Examples**: Real examples for each endpoint
3. **Error Codes**: Documented error conditions
4. **Rate Limits**: Document rate limiting policies

### Testing Requirements

1. **Unit Tests**: Each handler function
2. **Integration Tests**: Full endpoint behavior
3. **Performance Tests**: Load testing for scalability
4. **Security Tests**: Authentication and authorization
5. **Contract Tests**: API contract validation
6. **Mock Testing**: External service mocking

### Security Considerations

1. **Input Validation**: Sanitize all input parameters
2. **SQL Injection**: Use parameterized queries
3. **XSS Prevention**: Escape output data
4. **CORS Configuration**: Proper CORS headers
5. **File Upload Security**: Validate uploaded files
6. **Rate Limiting**: Prevent DoS attacks