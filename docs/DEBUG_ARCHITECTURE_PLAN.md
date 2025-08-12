# Debug Architecture Plan

## Overview

Separate client and server debug controls for cleaner architecture and better developer experience.

## Architecture

### Client Debug (Browser/Desktop)
- **Storage**: localStorage only
- **Control**: Developer options panel in Settings + DebugControlPanel
- **State**: No server sync, purely client-side
- **Features**: Console logging, performance metrics, data flow debugging

### Server Debug (Rust Backend)
- **Storage**: settings.yaml (simple on/off)
- **Control**: Edit settings.yaml file
- **Verbosity**: RUST_LOG env var (unchanged)
- **Features**: Backend logging, API debug info

## Benefits

1. **Clean Separation**: Client and server debug are independent
2. **No Network Overhead**: Client debug changes are instant
3. **Simple Server Config**: Just enabled true/false in settings.yaml
4. **Flexible Verbosity**: RUST_LOG still controls log levels
5. **Developer Freedom**: Each developer controls their own client debug
6. **Environment Specific**: Server debug can differ per deployment

## Implementation

### Phase 1: Client-Side Changes
```typescript
// Remove backend sync
- path: 'system.debug.enabled'
+ localStorage.getItem('debug.enabled')

// Unify state management
DebugControlPanel + Settings Panel → Same localStorage keys
```

### Phase 2: Server-Side Changes
```yaml
# settings.yaml
system:
  debug:
    enabled: false  # Simple on/off switch
```

```rust
// Check debug state
pub fn is_debug_enabled() -> bool {
    // First check environment override
    if let Ok(val) = std::env::var("DEBUG_ENABLED") {
        return val.parse().unwrap_or(false);
    }
    
    // Then check settings.yaml
    let settings = load_settings();
    settings.system.debug.enabled
}
```

### Phase 3: Environment Variables
```bash
# .env remains simple
RUST_LOG=info  # Controls verbosity when debug is enabled
# DEBUG_ENABLED can override settings.yaml if needed
```

## Migration Path

1. **Client**: Stop trying to read/write system.debug to backend
2. **Server**: Add minimal debug section back to settings.yaml
3. **Both**: Update documentation
4. **Test**: Verify client and server debug work independently

## Final State

### Client Debug Flow
```
User toggles debug in UI → localStorage update → Immediate effect
```

### Server Debug Flow
```
Edit settings.yaml → Restart server → Debug enabled
RUST_LOG controls verbosity level
```

## Why This Works

- **Client debug** needs to be fast and responsive → localStorage
- **Server debug** needs to be persistent and environment-specific → settings.yaml
- **Log verbosity** is orthogonal to debug on/off → RUST_LOG env var
- **No coupling** between client and server debug states

This provides the best of both worlds: responsive client debugging and stable server configuration.