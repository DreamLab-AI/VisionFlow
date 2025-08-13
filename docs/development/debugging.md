# Debug System Architecture

## Overview

This document outlines the debug system architecture for our application, featuring a clean client-server separation design. The system uses localStorage for client-side debugging controls and environment variables for server-side configuration, ensuring clear separation of concerns and optimal performance.

### Key Design Principles

- **Clean Separation**: Client and server debug systems operate independently
- **No Backend Pollution**: Client debug state never affects server operations
- **Instant Control**: Client-side debugging provides immediate feedback
- **Persistent Configuration**: Server-side settings persist across deployments
- **Development-First**: Optimized for developer experience and productivity

## Architecture Design

### System Separation

```
┌─────────────────────────────────────────────────────────────┐
│                    Debug Architecture                       │
├─────────────────────────┬───────────────────────────────────┤
│      Client-Side        │         Server-Side               │
├─────────────────────────┼───────────────────────────────────┤
│ • localStorage keys     │ • Environment variables           │
│ • Control panels        │ • RUST_LOG configuration          │
│ • WebGL inspector       │ • env_logger integration          │
│ • Instant toggles       │ • Persistent logging              │
│ • UI debug overlays     │ • Performance metrics             │
└─────────────────────────┴───────────────────────────────────┘
```

### Design Benefits

- **Performance**: No network overhead for debug toggles
- **Security**: Sensitive server logs never exposed to client
- **Scalability**: Each system optimised for its specific needs
- **Maintainability**: Clear boundaries reduce complexity

## Client-Side Debug System

### localStorage Architecture

The client-side debug system uses localStorage for persistent, instant debug control:

```javascript
// Debug State Management
const clientDebugState = {
  // Core debugging flags
  enabled: localStorage.getItem('debug_enabled') === 'true',
  verbose: localStorage.getItem('debug_verbose') === 'true',
  
  // Feature-specific debugging
  webgl: localStorage.getItem('debug_webgl') === 'true',
  network: localStorage.getItem('debug_network') === 'true',
  performance: localStorage.getItem('debug_performance') === 'true',
  
  // UI debugging
  showBounds: localStorage.getItem('debug_show_bounds') === 'true',
  showFPS: localStorage.getItem('debug_show_fps') === 'true',
  
  // Development tools
  mockData: localStorage.getItem('debug_mock_data') === 'true',
  skipAnimations: localStorage.getItem('debug_skip_animations') === 'true'
};
```

### Control Panel Implementation

```javascript
// Debug Control Panel
class DebugControlPanel {
  constructor() {
    this.panel = this.createPanel();
    this.bindEvents();
  }
  
  createPanel() {
    const panel = document.createElement('div');
    panel.className = 'debug-panel';
    panel.innerHTML = `
      <h3>Debug Controls</h3>
      <label><input type="checkbox" id="debug-enabled"> Enable Debug Mode</label>
      <label><input type="checkbox" id="debug-webgl"> WebGL Inspector</label>
      <label><input type="checkbox" id="debug-performance"> Performance Monitor</label>
      <label><input type="checkbox" id="debug-network"> Network Debug</label>
    `;
    return panel;
  }
  
  bindEvents() {
    this.panel.addEventListener('change', (e) => {
      const key = e.target.id.replace('-', '_');
      localStorage.setItem(key, e.target.checked.toString());
      this.applyDebugSettings();
    });
  }
  
  applyDebugSettings() {
    // Apply debug settings immediately
    if (clientDebugState.webgl) {
      this.enableWebGLInspector();
    }
    if (clientDebugState.performance) {
      this.showPerformanceMetrics();
    }
  }
}
```

### Unified Debug State Management

```javascript
// Centralized debug state manager
class ClientDebugManager {
  constructor() {
    this.state = this.loadState();
    this.observers = [];
  }
  
  loadState() {
    return Object.keys(clientDebugState).reduce((state, key) => {
      state[key] = localStorage.getItem(`debug_${key}`) === 'true';
      return state;
    }, {});
  }
  
  setState(key, value) {
    this.state[key] = value;
    localStorage.setItem(`debug_${key}`, value.toString());
    this.notifyObservers(key, value);
  }
  
  subscribe(callback) {
    this.observers.push(callback);
  }
  
  notifyObservers(key, value) {
    this.observers.forEach(callback => callback(key, value));
  }
}
```

### Debug Features

#### WebGL Inspector
```javascript
class WebGLInspector {
  enable() {
    if (!clientDebugState.webgl) return;
    
    // Capture WebGL calls
    const gl = canvas.getContext('webgl');
    this.wrapWebGLContext(gl);
  }
  
  wrapWebGLContext(gl) {
    const originalMethods = {};
    ['drawArrays', 'drawElements', 'useProgram'].forEach(method => {
      originalMethods[method] = gl[method];
      gl[method] = (...args) => {
        console.log(`WebGL ${method}:`, args);
        return originalMethods[method].apply(gl, args);
      };
    });
  }
}
```

#### Performance Monitor
```javascript
class PerformanceMonitor {
  constructor() {
    this.metrics = {
      fps: 0,
      frameTime: 0,
      memoryUsage: 0
    };
  }
  
  enable() {
    if (!clientDebugState.performance) return;
    
    this.startMonitoring();
    this.createOverlay();
  }
  
  startMonitoring() {
    let lastTime = performance.now();
    let frames = 0;
    
    const monitor = (currentTime) => {
      frames++;
      const delta = currentTime - lastTime;
      
      if (delta >= 1000) {
        this.metrics.fps = Math.round((frames * 1000) / delta);
        this.metrics.frameTime = delta / frames;
        frames = 0;
        lastTime = currentTime;
        this.updateOverlay();
      }
      
      requestAnimationFrame(monitor);
    };
    
    requestAnimationFrame(monitor);
  }
}
```

## Server-Side Debug System

### Environment Variables Configuration

The server-side debug system uses environment variables for persistent, deployment-safe configuration:

```bash
# Core debug settings
DEBUG_ENABLED=true
RUST_LOG=debug

# Feature-specific logging
RUST_LOG_DATABASE=info
RUST_LOG_NETWORK=debug
RUST_LOG_AUTH=warn

# Performance monitoring
RUST_LOG_PERFORMANCE=trace
DEBUG_PERFORMANCE_METRICS=true

# Development settings
DEBUG_SQL_QUERIES=true
DEBUG_CACHE_OPERATIONS=false
```

### env_logger Integration

```rust
use env_logger::{Builder, Env};
use log::{debug, info, warn, error};

// Logger initialisation
pub fn init_logger() {
    let env = Env::default()
        .filter_or("RUST_LOG", "info")
        .write_style_or("RUST_LOG_STYLE", "always");
    
    Builder::from_env(env)
        .format_timestamp_secs()
        .init();
    
    info!("Debug system initialised");
}

// Debug macros for conditional logging
macro_rules! debug_if_enabled {
    ($($arg:tt)*) => {
        if std::env::var("DEBUG_ENABLED").unwrap_or_default() == "true" {
            debug!($($arg)*);
        }
    };
}
```

### Logging Levels and Integration

```rust
// Structured logging with context
use serde_json::json;

pub struct DebugContext {
    module: String,
    operation: String,
    metadata: serde_json::Value,
}

impl DebugContext {
    pub fn log_operation<T>(&self, result: &Result<T, Box<dyn std::error::Error>>) {
        match result {
            Ok(_) => {
                debug!("Operation successful: {}", json!({
                    "module": self.module,
                    "operation": self.operation,
                    "status": "success",
                    "metadata": self.metadata
                }));
            }
            Err(e) => {
                error!("Operation failed: {}", json!({
                    "module": self.module,
                    "operation": self.operation,
                    "status": "error",
                    "error": e.to_string(),
                    "metadata": self.metadata
                }));
            }
        }
    }
}
```

### Code Integration Examples

```rust
// Database operations
pub async fn execute_query(&self, query: &str) -> Result<Vec<Row>, DatabaseError> {
    let debug_ctx = DebugContext {
        module: "database".to_string(),
        operation: "execute_query".to_string(),
        metadata: json!({ "query": query }),
    };
    
    debug_if_enabled!("Executing query: {}", query);
    
    let result = self.connection.query(query).await;
    debug_ctx.log_operation(&result);
    
    result
}

// API handlers
pub async fn handle_request(req: Request) -> Result<Response, ApiError> {
    let debug_ctx = DebugContext {
        module: "api".to_string(),
        operation: "handle_request".to_string(),
        metadata: json!({
            "method": req.method().as_str(),
            "path": req.uri().path(),
            "headers": req.headers()
        }),
    };
    
    debug_if_enabled!("Processing request: {} {}", req.method(), req.uri());
    
    let result = process_request(req).await;
    debug_ctx.log_operation(&result);
    
    result
}
```

## Docker Integration

### Container Configuration

```dockerfile
# Development Dockerfile
FROM rust:1.70-slim

# Debug environment variables
ENV DEBUG_ENABLED=true
ENV RUST_LOG=debug
ENV RUST_LOG_STYLE=always

# Development tools
RUN apt-get update && apt-get install -y \
    gdb \
    strace \
    htop

COPY . .
RUN cargo build --features debug

CMD ["cargo", "run", "--features", "debug"]
```

### Environment Inheritance

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    environment:
      # Debug settings from .env file
      - DEBUG_ENABLED=${DEBUG_ENABLED:-false}
      - RUST_LOG=${RUST_LOG:-info}
      - RUST_LOG_PERFORMANCE=${RUST_LOG_PERFORMANCE:-warn}
    volumes:
      # Mount logs directory
      - ./logs:/app/logs
    ports:
      - "8080:8080"

  # Development override
  app-dev:
    extends:
      service: app
    environment:
      - DEBUG_ENABLED=true
      - RUST_LOG=debug
      - RUST_LOG_PERFORMANCE=trace
    volumes:
      # Live code reloading
      - .:/app
      - /app/target
```

### Development vs Production Settings

```bash
# .env.development
DEBUG_ENABLED=true
RUST_LOG=debug
RUST_LOG_PERFORMANCE=trace
DEBUG_SQL_QUERIES=true
DEBUG_CACHE_OPERATIONS=true

# .env.production
DEBUG_ENABLED=false
RUST_LOG=info
RUST_LOG_PERFORMANCE=warn
DEBUG_SQL_QUERIES=false
DEBUG_CACHE_OPERATIONS=false
```

## Migration Workflows

### Development Migration

```bash
#!/bin/bash
# migrate-to-new-debug.sh

echo "Migrating to new debug system..."

# 1. Update client-side debug keys
echo "Updating client localStorage keys..."
cat << 'EOF' > migrate-client-debug.js
// Migration script for client-side debug settings
const oldKeys = ['debugMode', 'showDebugInfo', 'enableLogging'];
const newKeys = ['debug_enabled', 'debug_verbose', 'debug_performance'];

oldKeys.forEach((oldKey, index) => {
  const value = localStorage.getItem(oldKey);
  if (value !== null) {
    localStorage.setItem(newKeys[index], value);
    localStorage.removeItem(oldKey);
  }
});

console.log('Client debug migration completed');
EOF

# 2. Update server environment variables
echo "Updating server environment variables..."
if [ -f .env ]; then
  sed -i 's/OLD_DEBUG_FLAG/DEBUG_ENABLED/g' .env
  sed -i 's/LOG_LEVEL/RUST_LOG/g' .env
fi

# 3. Update Docker configurations
echo "Updating Docker configurations..."
docker-compose down
docker-compose up --build -d

echo "Migration completed successfully!"
```

### Production Migration

```bash
#!/bin/bash
# production-migration.sh

echo "Performing production debug system migration..."

# 1. Backup current settings
kubectl get configmap debug-config -o yaml > debug-config-backup.yaml

# 2. Create new debug ConfigMap
cat << EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: debug-config-v2
data:
  DEBUG_ENABLED: "false"
  RUST_LOG: "info"
  RUST_LOG_PERFORMANCE: "warn"
EOF

# 3. Rolling update deployment
kubectl set env deployment/app --from=configmap/debug-config-v2
kubectl rollout status deployment/app

echo "Production migration completed!"
```

### Legacy System Transition

```rust
// Legacy compatibility layer
pub struct LegacyDebugAdapter {
    legacy_enabled: bool,
    new_system: DebugManager,
}

impl LegacyDebugAdapter {
    pub fn new() -> Self {
        Self {
            legacy_enabled: std::env::var("OLD_DEBUG_MODE").unwrap_or_default() == "1",
            new_system: DebugManager::new(),
        }
    }
    
    pub fn should_debug(&self) -> bool {
        // Support both old and new debug flags during transition
        self.legacy_enabled || self.new_system.is_enabled()
    }
    
    pub fn log_debug(&self, message: &str) {
        if self.should_debug() {
            debug!("{}", message);
        }
    }
}
```

## Troubleshooting

### Common Issues

#### Issue 1: localStorage Not Persisting
```javascript
// Problem: Debug settings reset on page reload
// Solution: Check browser storage limits and clear old data

function cleanupDebugStorage() {
  const debugKeys = Object.keys(localStorage)
    .filter(key => key.startsWith('debug_'));
  
  console.log(`Found ${debugKeys.length} debug keys`);
  
  // Remove invalid entries
  debugKeys.forEach(key => {
    const value = localStorage.getItem(key);
    if (value !== 'true' && value !== 'false') {
      localStorage.removeItem(key);
      console.log(`Removed invalid debug key: ${key}`);
    }
  });
}
```

#### Issue 2: Environment Variables Not Loading
```bash
# Problem: Server debug settings not taking effect
# Solution: Check environment variable precedence

# Debug environment loading
echo "Current debug environment:"
env | grep -E "(DEBUG|RUST_LOG)" | sort

# Check if variables are properly set
if [ -z "$DEBUG_ENABLED" ]; then
  echo "Warning: DEBUG_ENABLED not set"
fi

if [ -z "$RUST_LOG" ]; then
  echo "Warning: RUST_LOG not set, defaulting to 'info'"
  export RUST_LOG=info
fi
```

#### Issue 3: Performance Impact
```rust
// Problem: Debug logging affecting performance
// Solution: Use conditional compilation for hot paths

#[cfg(debug_assertions)]
macro_rules! debug_trace {
    ($($arg:tt)*) => {
        if std::env::var("DEBUG_PERFORMANCE").unwrap_or_default() == "true" {
            trace!($($arg)*);
        }
    };
}

#[cfg(not(debug_assertions))]
macro_rules! debug_trace {
    ($($arg:tt)*) => {};
}

// Usage in performance-critical code
pub fn hot_path_function(&self) -> Result<Data, Error> {
    debug_trace!("Entering hot path function");
    
    // Performance-critical code here
    let result = expensive_operation();
    
    debug_trace!("Hot path function completed: {:?}", result);
    result
}
```

### Performance Considerations

#### Client-Side Performance

```javascript
// Optimize debug checks
class DebugOptimizer {
  constructor() {
    // Cache debug state to avoid localStorage reads
    this.cachedState = this.loadDebugState();
    
    // Update cache when storage changes
    window.addEventListener('storage', () => {
      this.cachedState = this.loadDebugState();
    });
  }
  
  loadDebugState() {
    return {
      enabled: localStorage.getItem('debug_enabled') === 'true',
      webgl: localStorage.getItem('debug_webgl') === 'true',
      performance: localStorage.getItem('debug_performance') === 'true'
    };
  }
  
  isDebugEnabled() {
    // Fast cached lookup instead of localStorage read
    return this.cachedState.enabled;
  }
}
```

#### Server-Side Performance

```rust
// Lazy static debug flags for performance
use lazy_static::lazy_static;
use std::sync::atomic::{AtomicBool, Ordering};

lazy_static! {
    static ref DEBUG_ENABLED: AtomicBool = {
        let enabled = std::env::var("DEBUG_ENABLED").unwrap_or_default() == "true";
        AtomicBool::new(enabled)
    };
}

// Fast debug checks
pub fn is_debug_enabled() -> bool {
    DEBUG_ENABLED.load(Ordering::Relaxed)
}

// Update debug state at runtime
pub fn set_debug_enabled(enabled: bool) {
    DEBUG_ENABLED.store(enabled, Ordering::Relaxed);
}
```

## Best Practices

### Development Best Practices

1. **Use Semantic Debug Keys**
   ```javascript
   // Good: Descriptive and hierarchical
   localStorage.setItem('debug_render_webgl_shaders', 'true');
   localStorage.setItem('debug_network_api_responses', 'true');
   
   // Bad: Generic and unclear
   localStorage.setItem('debug1', 'true');
   localStorage.setItem('dbg', 'true');
   ```

2. **Implement Debug Namespaces**
   ```rust
   // Organize debug logs with structured prefixes
   debug!(target: "auth::login", "User login attempt: {}", user_id);
   debug!(target: "database::query", "Executing query: {}", sql);
   debug!(target: "cache::operations", "Cache hit for key: {}", key);
   ```

3. **Create Debug Profiles**
   ```bash
   # .env.debug.minimal
   DEBUG_ENABLED=true
   RUST_LOG=info
   
   # .env.debug.full
   DEBUG_ENABLED=true
   RUST_LOG=debug
   RUST_LOG_PERFORMANCE=trace
   DEBUG_SQL_QUERIES=true
   ```

### Production Best Practices

1. **Safe Debug Toggles**
   ```rust
   // Never expose sensitive data in debug logs
   pub fn safe_debug_user(user: &User) -> serde_json::Value {
       json!({
           "id": user.id,
           "username": user.username,
           // Never log sensitive fields like passwords, tokens, etc.
           "created_at": user.created_at
       })
   }
   ```

2. **Performance-Safe Logging**
   ```rust
   // Use format_args! for expensive string formatting
   if is_debug_enabled() {
       debug!("Complex operation result: {}", format_args!("{:#?}", complex_data));
   }
   ```

3. **Monitoring Integration**
   ```rust
   // Integrate debug system with monitoring
   pub fn log_with_metrics(level: log::Level, message: &str, metrics: &HashMap<String, f64>) {
       log!(level, "{}", message);
       
       // Send to monitoring system if debug enabled
       if is_debug_enabled() {
           metrics_client.send_debug_metrics(metrics);
       }
   }
   ```

### Team Collaboration Guidelines

1. **Debug Documentation Standards**
   ```javascript
   /**
    * Debug Configuration
    * 
    * @debug debug_feature_name - Enable/disable specific feature debugging
    * @env DEBUG_FEATURE_NAME - Server-side equivalent
    * @purpose Helps debug rendering pipeline issues
    * @performance_impact Low - only affects debug builds
    */
   class FeatureDebugger {
     // Implementation
   }
   ```

2. **Shared Debug Configurations**
   ```json
   {
     "debug_profiles": {
       "frontend_dev": {
         "debug_enabled": true,
         "debug_webgl": true,
         "debug_performance": false
       },
       "backend_dev": {
         "DEBUG_ENABLED": true,
         "RUST_LOG": "debug",
         "DEBUG_SQL_QUERIES": true
       },
       "qa_testing": {
         "debug_enabled": true,
         "debug_network": true,
         "DEBUG_ENABLED": false,
         "RUST_LOG": "info"
       }
     }
   }
   ```

3. **Debug Handoff Protocols**
   ```markdown
   ## Debug Handoff Checklist
   
   ### Frontend Debug State
   - [ ] `debug_enabled`: true/false
   - [ ] `debug_webgl`: true/false  
   - [ ] `debug_performance`: true/false
   - [ ] Custom debug keys: [list any custom keys]
   
   ### Backend Debug State
   - [ ] `DEBUG_ENABLED`: true/false
   - [ ] `RUST_LOG`: [current level]
   - [ ] `RUST_LOG_PERFORMANCE`: [current level]
   - [ ] Environment: [development/staging/production]
   
   ### Reproduction Steps
   1. Set debug flags as listed above
   2. [Additional steps]
   
   ### Expected Debug Output
   - Console logs: [describe expected logs]
   - Server logs: [describe expected logs]
   - Performance metrics: [describe expected metrics]
   ```

## Evolution and Current Implementation

This debug system has evolved from simple console.log statements and basic server logging to a sophisticated, production-ready architecture. The current implementation represents best practices learned from:

- **Performance Optimization**: Moving from runtime debug checks to cached states
- **Security Hardening**: Separating client and server debug contexts
- **Developer Experience**: Creating intuitive control panels and clear debug APIs
- **Operational Excellence**: Integrating with monitoring and deployment systems

The architecture is designed to scale with application growth while maintaining clean separation of concerns and optimal performance characteristics.

---

*This documentation is part of the comprehensive development guide and should be updated as the debug system evolves.*