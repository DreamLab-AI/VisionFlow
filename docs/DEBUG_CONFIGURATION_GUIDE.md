# Debug Configuration Guide

## Overview

The VisionFlow debug configuration system has been centralized to use environment variables, providing a single source of truth for all debugging settings across the backend, frontend, and Docker environments.

## Key Changes

### 1. Environment Variable Configuration

All debug settings are now controlled through two primary environment variables in the `.env` file:

- **`DEBUG_ENABLED`**: Master switch for all debug features (true/false)
- **`RUST_LOG`**: Controls backend logging verbosity levels

### 2. Removed Legacy Configuration

- **Removed**: Debug section from `data/settings.yaml`
- **Removed**: Hardcoded `RUST_LOG` values from Docker compose files
- **Removed**: Legacy `DEBUG_MODE` variable (replaced with `DEBUG_ENABLED`)

## Configuration Details

### Backend (Rust)

The backend now uses `env_logger` which automatically reads the `RUST_LOG` environment variable. This provides standardized logging configuration across the entire Rust ecosystem.

**Example RUST_LOG configurations:**
```bash
# Development - verbose logging
RUST_LOG=debug,webxr=trace,actix_web=info

# Production - minimal logging
RUST_LOG=info,webxr=warn

# Debug specific modules
RUST_LOG=warn,webxr::services::claude_flow=trace
```

**Logging levels** (from most to least verbose):
- `trace`: Very detailed debugging information
- `debug`: Debugging information
- `info`: Informational messages
- `warn`: Warnings
- `error`: Errors only

### Frontend (Vite/React)

Frontend debugging continues to use Vite environment variables, but now inherits the master `DEBUG_ENABLED` setting:

- `VITE_DEBUG=${DEBUG_ENABLED}`: Inherits from backend setting in Docker
- Other Vite debug variables remain unchanged

### Docker Environment

All Docker compose files now inherit debug settings from the `.env` file:

```yaml
env_file:
  - .env
environment:
  - VITE_DEBUG=${DEBUG_ENABLED:-true}
  # RUST_LOG inherited from .env
```

## Usage

### 1. Development Environment

Edit your `.env` file:
```bash
DEBUG_ENABLED=true
RUST_LOG=debug,webxr=trace,actix_web=info
```

### 2. Production Environment

Edit your `.env` file:
```bash
DEBUG_ENABLED=false
RUST_LOG=info
```

### 3. Debugging Specific Components

Target specific modules with RUST_LOG:
```bash
# Debug only Claude Flow service
RUST_LOG=warn,webxr::services::claude_flow=trace

# Debug GPU compute operations
RUST_LOG=info,webxr::gpu=debug

# Debug WebSocket connections
RUST_LOG=info,webxr::handlers::socket=debug
```

## Implementation Details

### Files Modified

1. **`.env_template`**: Updated with new debug variables
2. **`src/utils/logging.rs`**: Replaced `simplelog` with `env_logger`
3. **`src/main.rs`**: Simplified logging initialization
4. **`src/app_state.rs`**: Added `debug_enabled` field
5. **`data/settings.yaml`**: Maintains minimal debug section for struct compatibility (values ignored at runtime)
6. **`docker-compose.*.yml`**: Removed hardcoded RUST_LOG
7. **`Dockerfile.*`**: Added environment variable override support

### Important Note: Settings.yaml Debug Section

The debug section in `settings.yaml` exists only for structural compatibility with the Rust `SystemSettings` struct. These values are **not used at runtime** - the actual debug behavior is controlled entirely by environment variables:

- **`DEBUG_ENABLED`** environment variable controls all debug features
- **`RUST_LOG`** environment variable controls logging levels
- The settings.yaml values are placeholders that get overridden

### Accessing Debug State in Code

The debug state can be accessed in Rust code:

```rust
use webxr::utils::logging::is_debug_enabled;

if is_debug_enabled() {
    // Debug-only functionality
}
```

Or via AppState:
```rust
if app_state.debug_enabled {
    // Debug-only functionality
}
```

## Migration from Old System

If you're upgrading from the old system:

1. Copy your desired log level from `data/settings.yaml` to `.env`
2. Replace `DEBUG_MODE` with `DEBUG_ENABLED` in `.env`
3. Remove any custom RUST_LOG overrides in docker-compose files
4. Restart your containers to apply changes

## Troubleshooting

### Logs Not Appearing

1. Check `.env` file exists and is readable
2. Verify `RUST_LOG` syntax is correct
3. Ensure Docker containers were rebuilt after changes
4. Check that `env_file: .env` is present in docker-compose

### Debug Features Not Working

1. Verify `DEBUG_ENABLED=true` in `.env`
2. Check frontend receives `VITE_DEBUG` variable
3. Restart all services after changing `.env`

### Performance Issues with Verbose Logging

Verbose logging (trace/debug levels) can impact performance. For production:
```bash
DEBUG_ENABLED=false
RUST_LOG=warn
```

## Best Practices

1. **Development**: Use verbose logging for debugging
2. **Staging**: Use info level with selective debug modules
3. **Production**: Use warn/error only, DEBUG_ENABLED=false
4. **Troubleshooting**: Target specific modules rather than global trace
5. **Security**: Never log sensitive data regardless of debug level

## Environment Variable Reference

| Variable | Description | Example | Default |
|----------|-------------|---------|---------|
| `DEBUG_ENABLED` | Master debug switch | `true`, `false` | `false` |
| `RUST_LOG` | Backend log level | `debug`, `info`, `warn` | `info` |
| `VITE_DEBUG` | Frontend debug mode | `true`, `false` | Inherits from `DEBUG_ENABLED` |

## Future Improvements

- Dynamic log level adjustment without restart
- Per-component debug toggles
- Debug data export functionality
- Performance profiling integration
- Structured logging with JSON output