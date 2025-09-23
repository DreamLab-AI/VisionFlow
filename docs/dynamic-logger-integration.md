# Dynamic Logger Integration System

## Overview

The Dynamic Logger Integration System bridges the Control Center's debug settings with the logger system, enabling real-time log level changes without requiring page refresh or rebuild.

## Architecture

### Configuration Hierarchy

1. **Runtime Settings** (Highest Priority)
   - Control Center Developer Panel settings
   - Stored in `clientDebugState` (localStorage)
   - Real-time updates via storage events

2. **Environment Variables** (Medium Priority)
   - `VITE_LOG_LEVEL` for build-time defaults
   - Fallback when runtime settings unavailable

3. **System Defaults** (Lowest Priority)
   - Development: `debug` level
   - Production: `info` level

### Core Components

#### 1. Logger Registry (`loggerRegistry.ts`)

Tracks all active logger instances and enables bulk configuration updates.

```typescript
interface LoggerInstance {
  namespace: string;
  logger: any;
  updateLevel: (level: LogLevel) => void;
  setEnabled: (enabled: boolean) => void;
  isEnabled: () => boolean;
  getCurrentConfig: () => { level: LogLevel; enabled: boolean };
}
```

#### 2. Dynamic Configuration Manager (`dynamicLoggerConfig.ts`)

Manages configuration from multiple sources with real-time updates.

```typescript
class DynamicLoggerConfig {
  getCurrentConfig(): LoggerConfig;
  subscribe(callback: (config: LoggerConfig) => void): () => void;
  forceUpdate(): void;
  getConfigInfo(): ConfigurationInfo;
}
```

#### 3. Integration Bridge (`loggerDebugBridge.ts`)

Connects `clientDebugState` to the logger system.

```typescript
class LoggerDebugBridge {
  initialize(): void;
  synchronizeSettings(): void;
  getStatus(): IntegrationStatus;
  cleanup(): void;
}
```

#### 4. Enhanced Logger Core (`logger.ts`)

Updated core logger with dynamic configuration support.

```typescript
// New dynamic methods added to logger instances
interface DynamicLogger extends Logger {
  updateLevel(level: LogLevel): void;
  setEnabled(enabled: boolean): void;
  getCurrentConfig(): { level: LogLevel; enabled: boolean };
  isEnabled(): boolean;
  namespace: string;
}
```

#### 5. Auto-Initialization System (`loggerIntegrationInit.ts`)

Provides centralized initialization and management.

```typescript
// Auto-initializes on application load
initializeDynamicLoggers({
  enableBridgeDebug: import.meta.env?.DEV,
  logInitialization: import.meta.env?.DEV
});
```

## Usage

### Basic Usage

```typescript
// Import from loggerConfig (recommended)
import { createLogger, createAgentLogger } from '../utils/loggerConfig';

// Create loggers that respond to Control Center settings
const logger = createLogger('MyModule');
const agentLogger = createAgentLogger('MyAgent');

// Use normally - settings changes apply automatically
logger.info('This respects Control Center settings');
logger.debug('Debug level controlled by Control Center');
```

### Backward Compatibility

```typescript
// Existing imports continue to work
import { createLogger } from '../utils/logger'; // Still works

// But won't respect Control Center settings
// Recommended to update imports to '../utils/loggerConfig'
```

### Advanced Usage

```typescript
// Manual initialization with options
import { initializeDynamicLoggers } from '../utils/loggerIntegrationInit';

initializeDynamicLoggers({
  enableBridgeDebug: true,    // Enable debug logging for bridge
  autoStart: true,            // Auto-start integration
  logInitialization: true     // Log initialization process
});

// Get system status
import { getLoggerSystemStatus } from '../utils/loggerIntegrationInit';
const status = getLoggerSystemStatus();
```

### Control Center Integration

The system automatically responds to these Control Center settings:

- **`debug.consoleLogging`** - Master toggle for console logging
- **`debug.logLevel`** - Minimum log level (`error`, `warn`, `info`, `debug`)

Changes are applied immediately across all active logger instances.

## Configuration Sources

### Runtime Settings (Control Center)

```typescript
// Access current settings
import { clientDebugState } from '../utils/clientDebugState';

const consoleLogging = clientDebugState.get('consoleLogging');
const logLevel = clientDebugState.get('logLevel');

// Subscribe to changes
const unsubscribe = clientDebugState.subscribe('logLevel', (newLevel) => {
  console.log('Log level changed to:', newLevel);
});
```

### Environment Variables

```bash
# .env file
VITE_LOG_LEVEL=info    # Set default log level
```

### System Defaults

- **Development**: `debug` level, console logging enabled
- **Production**: `info` level, console logging based on settings

## Real-Time Updates

The system uses localStorage events for real-time synchronization:

1. User changes setting in Control Center
2. `clientDebugState` updates localStorage
3. Storage event triggers across all tabs/windows
4. `DynamicLoggerConfig` recomputes effective configuration
5. `LoggerRegistry` updates all active logger instances
6. Changes apply immediately without page refresh

## Migration Guide

### Updating Existing Code

```typescript
// OLD - Static configuration
import { createLogger } from '../utils/logger';
const logger = createLogger('Module', { level: 'debug' });

// NEW - Dynamic configuration
import { createLogger } from '../utils/loggerConfig';
const logger = createLogger('Module'); // Respects Control Center settings

// NEW - Override Control Center if needed
const logger = createLogger('Module', {
  respectRuntimeSettings: false,  // Ignore Control Center
  level: 'debug'                  // Use fixed level
});
```

### Compatibility Notes

- All existing logger method calls work unchanged
- `createLogger` and `createAgentLogger` imports should be updated
- Original logger exports marked as deprecated but still functional
- No breaking changes to existing logger functionality

## Testing and Debugging

### Demo Functions

```typescript
import { runLoggerDemo, testConfigurationChanges } from '../utils/loggerIntegrationDemo';

// Interactive demonstration
runLoggerDemo();

// Automated configuration testing
testConfigurationChanges();
```

### Browser Console

```javascript
// Available in browser console
runLoggerDemo();              // Interactive demo
testConfigurationChanges();   // Automated test
getLoggerSystemStatus();      // View system status
```

### Debug Information

```typescript
import { getLoggerSystemStatus } from '../utils/loggerIntegrationInit';

const status = getLoggerSystemStatus();
console.log('Logger System Status:', status);

// Returns:
// {
//   initialized: true,
//   bridge: { ... },
//   configuration: { ... },
//   registry: { total: 5, enabled: 3, byLevel: {...} },
//   timestamp: "..."
// }
```

## Error Handling

The system includes comprehensive error handling:

- **Graceful Degradation**: Falls back to environment/default settings if runtime settings fail
- **localStorage Errors**: Continues with in-memory state if localStorage unavailable
- **Circular Reference Protection**: Safe JSON serialization in logger output
- **Subscription Cleanup**: Automatic cleanup of event listeners

## Performance Considerations

- **Lazy Initialization**: Components initialize only when needed
- **Event Debouncing**: Configuration changes debounced to prevent excessive updates
- **Memory Management**: Automatic cleanup of unused logger references
- **Storage Optimization**: Minimal localStorage usage with efficient change detection

## Files Created/Modified

### New Files
- `/client/src/utils/loggerRegistry.ts` - Logger instance tracking
- `/client/src/utils/dynamicLoggerConfig.ts` - Configuration management
- `/client/src/utils/loggerDebugBridge.ts` - Integration bridge
- `/client/src/utils/loggerIntegrationInit.ts` - Initialization system
- `/client/src/utils/loggerIntegrationDemo.ts` - Demo and testing utilities

### Modified Files
- `/client/src/utils/logger.ts` - Added dynamic configuration methods
- `/client/src/utils/loggerConfig.ts` - Updated with full dynamic support

### Integration Points
- `/client/src/utils/clientDebugState.ts` - Existing debug state management
- Control Center Developer Panel - UI settings that control the system

## Summary

The Dynamic Logger Integration System provides seamless real-time control of logger behavior from the Control Center interface. It maintains full backward compatibility while adding powerful new capabilities for development and debugging workflows.

Key benefits:
- ✅ Real-time log level changes without page refresh
- ✅ Centralized control via Control Center interface
- ✅ Full backward compatibility with existing code
- ✅ Comprehensive error handling and fallback behavior
- ✅ Auto-initialization with minimal setup required
- ✅ Debug and monitoring capabilities built-in