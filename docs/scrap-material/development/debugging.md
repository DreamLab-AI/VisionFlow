# VisionFlow Debug Architecture Guide

## Overview

The VisionFlow debug system uses a clean separation between client and server debugging, with each layer using the most appropriate storage and control mechanisms. This architecture eliminates configuration conflicts and provides optimal performance for both development and production environments.

## Architecture Principles

### 1. Clean Separation
- **Client Debug**: Browser/desktop app controlled via localStorage
- **Server Debug**: Rust backend controlled via environment variables
- **No Cross-Contamination**: Each layer manages its own debug state independently

### 2. Single Source of Truth
- **Client**: localStorage keys managed by `clientDebugState`
- **Server**: `DEBUG_ENABLED` and `RUST_LOG` environment variables
- **No Synchronization**: Debug states don't sync between client and server

## Client-Side Debug System

### Storage & Control

Client debug is entirely browser/desktop based:

- **Storage**: localStorage only (`debug.enabled`, `debug.data`, etc.)
- **Control Points**:
  - **DebugControlPanel**: Quick access via `Ctrl+Shift+D`
  - **Settings Panel**: Developer section in main settings
- **State Management**: Unified `clientDebugState` utility
- **Network**: No backend API calls for debug settings

### Key localStorage Keys

```javascript
debug.enabled                  // Master debug toggle
debug.consoleLogging          // Console logging control
debug.logLevel               // Client log verbosity
debug.data                   // Data flow debugging
debug.performance            // Performance metrics
debug.showNodeIds           // Node visualization debug
debug.enableWebsocketDebug  // WebSocket message logging
debug.visualization         // Visualization debug features
debug.network              // Network request debugging
```

### Client Debug Features

1. **Console Logging**: Controlled debug output to browser console
2. **Performance Metrics**: Real-time performance monitoring
3. **Data Flow Debugging**: Track data through application layers
4. **Visualization Debug**: Node IDs, physics debug, shader debug
5. **Network Debug**: API calls, WebSocket message logging
6. **Component Debug**: React component state and props inspection

### Usage Examples

```javascript
// Enable debug in browser console
localStorage.setItem('debug.enabled', 'true');
localStorage.setItem('debug.performance', 'true');

// Or use the Developer Settings UI
// Settings → Developer → Debug Options
```

## Server-Side Debug System

### Architecture

Server debug uses a two-tier environment-based system:

1. **`DEBUG_ENABLED`**: Master switch for debug features (boolean)
2. **`RUST_LOG`**: Granular logging control using `env_logger` standard

### Configuration

#### Environment Variables (.env)

```bash
# Master debug toggle
DEBUG_ENABLED=false

# Logging configuration (when debug is enabled)
RUST_LOG=info

# Development example
DEBUG_ENABLED=true
RUST_LOG=debug,webxr=trace,actix_web=info

# Production example  
DEBUG_ENABLED=false
RUST_LOG=warn
```

#### RUST_LOG Examples

```bash
# Verbose development logging
RUST_LOG=debug,webxr=trace,actix_web=info

# Production minimal logging
RUST_LOG=info,webxr=warn

# Debug specific modules
RUST_LOG=warn,webxr::services::claude_flow=trace

# Debug GPU operations
RUST_LOG=info,webxr::gpu=debug

# Debug WebSocket connections
RUST_LOG=info,webxr::handlers::socket=debug
```

### Logging Levels

From most to least verbose:
- **`trace`**: Very detailed debugging information
- **`debug`**: General debugging information
- **`info`**: Informational messages
- **`warn`**: Warning messages
- **`error`**: Error messages only

### Code Integration

```rust
use webxr::utils::logging::is_debug_enabled;

// Check debug state
if is_debug_enabled() {
    // Debug-only functionality
    debug!("Debug information");
}

// Or via AppState
if app_state.debug_enabled {
    // Debug-only code
}
```

## Docker Integration

### Environment Inheritance

All Docker compose files inherit from `.env`:

```yaml
services:
  backend:
    env_file:
      - .env
    environment:
      - VITE_DEBUG=${DEBUG_ENABLED:-false}
      # RUST_LOG automatically inherited from .env
```

### Frontend Integration

The frontend inherits the master debug flag:

```bash
# Frontend receives backend debug state
VITE_DEBUG=${DEBUG_ENABLED}
```

## Migration & Deployment

### Development Setup

1. **Create/Update .env file**:
   ```bash
   cp .env_template .env
   ```

2. **Set development values**:
   ```bash
   DEBUG_ENABLED=true
   RUST_LOG=debug,webxr=trace
   ```

3. **Restart services**:
   ```bash
   docker-compose down
   docker-compose build
   docker-compose up
   ```

### Production Setup

1. **Production .env configuration**:
   ```bash
   DEBUG_ENABLED=false
   RUST_LOG=warn
   ```

2. **Verify client debug is disabled**:
   ```javascript
   // In browser console
   localStorage.setItem('debug.enabled', 'false');
   ```

### Migration from Legacy System

If upgrading from the old configuration system:

1. **Remove old settings**: Debug section from `data/settings.yaml` (if present)
2. **Remove Docker overrides**: Delete hardcoded `RUST_LOG` from compose files
3. **Replace variables**: Change `DEBUG_MODE` to `DEBUG_ENABLED` in `.env`
4. **Rebuild containers**: Full rebuild required for changes

## Development Workflows

### Full Debug Mode (Development)

```bash
# .env
DEBUG_ENABLED=true
RUST_LOG=debug,webxr=trace

# Browser console
localStorage.setItem('debug.enabled', 'true');
localStorage.setItem('debug.performance', 'true');
localStorage.setItem('debug.data', 'true');
```

### Targeted Debugging

```bash
# Debug only specific components
DEBUG_ENABLED=true
RUST_LOG=warn,webxr::services::claude_flow=trace

# Browser - enable only needed debug
localStorage.setItem('debug.enabled', 'true');
localStorage.setItem('debug.network', 'true');
localStorage.removeItem('debug.performance');
```

### Production Debugging

```bash
# Minimal server debug for troubleshooting
DEBUG_ENABLED=true
RUST_LOG=error,webxr::handlers=info

# Client debug remains off
localStorage.setItem('debug.enabled', 'false');
```

## Troubleshooting

### Debug Not Working

1. **Server Debug Issues**:
   - Verify `.env` file exists and is readable
   - Check `RUST_LOG` syntax is correct
   - Ensure containers were rebuilt after `.env` changes
   - Confirm `env_file: .env` in docker-compose.yml

2. **Client Debug Issues**:
   - Check localStorage values in browser dev tools
   - Verify Settings Panel → Developer section reflects correct state
   - Ensure `Ctrl+Shift+D` opens DebugControlPanel
   - Check for console errors preventing debug initialization

3. **Log Output Missing**:
   - Verify debug is enabled: `DEBUG_ENABLED=true`
   - Check log level allows desired output
   - Ensure logging isn't filtered by container orchestration
   - Verify module names in `RUST_LOG` are correct

### Performance Issues

Verbose logging can impact performance:

1. **Development**: Use trace/debug levels freely
2. **Staging**: Limit to info level with selective debug modules
3. **Production**: Use warn/error only, keep `DEBUG_ENABLED=false`
4. **Targeted**: Debug specific modules rather than global trace

### Security Considerations

1. **Never log sensitive data** at any debug level
2. **Disable debug in production** environments
3. **Rotate logs regularly** with verbose debugging
4. **Monitor log file sizes** to prevent disk exhaustion
5. **Filter sensitive fields** from debug output

## Best Practices

### Development

1. **Use localStorage**: Enable client debug features as needed
2. **Granular Logging**: Target specific modules with `RUST_LOG`
3. **Clean State**: Clear debug localStorage when switching projects
4. **Document Discoveries**: Note useful debug combinations

### Production

1. **Minimal Logging**: Use warn/error levels only
2. **Disable Debug**: Set `DEBUG_ENABLED=false`
3. **Monitor Performance**: Watch for debug artifacts
4. **Secure Logs**: Ensure no sensitive data in output

### Team Collaboration

1. **Share Configurations**: Document useful debug setups
2. **Environment Consistency**: Use same debug levels for similar issues
3. **Clear Documentation**: Note debug requirements in PR descriptions
4. **Reset After Use**: Clean up debug settings when done

## Environment Variable Reference

| Variable | Description | Examples | Default |
|----------|-------------|----------|---------|
| `DEBUG_ENABLED` | Master debug toggle | `true`, `false` | `false` |
| `RUST_LOG` | Backend logging control | `info`, `debug`, `trace` | `info` |
| `VITE_DEBUG` | Frontend debug mode | `true`, `false` | Inherits from `DEBUG_ENABLED` |

## Future Enhancements

### Planned Improvements

1. **Dynamic Log Levels**: Adjust verbosity without restart
2. **Debug Profiles**: Predefined debug configurations
3. **Export/Import**: Share debug settings between developers
4. **Performance Profiling**: Integrated performance analysis
5. **Structured Logging**: JSON output for log analysis
6. **Debug Dashboard**: Real-time debug monitoring interface

### Integration Opportunities

1. **OpenTelemetry**: Distributed tracing integration
2. **Metrics Collection**: Prometheus/Grafana integration  
3. **Error Reporting**: Sentry/similar error tracking
4. **Performance Monitoring**: APM integration
5. **Log Aggregation**: ELK/similar log analysis stack

---

## Quick Reference

### Enable All Debug (Development)

```bash
# .env
DEBUG_ENABLED=true
RUST_LOG=debug

# Browser console
localStorage.setItem('debug.enabled', 'true');
```

### Disable All Debug (Production)

```bash
# .env
DEBUG_ENABLED=false
RUST_LOG=warn

# Browser console
localStorage.setItem('debug.enabled', 'false');
```

### Access Debug Controls

- **Quick Access**: `Ctrl+Shift+D` for DebugControlPanel
- **Full Controls**: Settings → Developer → Debug Options
- **Backend Config**: Edit `.env` file and restart services

This architecture provides optimal debug control with clean separation, high performance, and excellent developer experience across all environments.