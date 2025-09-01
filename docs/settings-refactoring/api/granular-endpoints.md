# Granular Settings API Specification

## Overview
This document specifies the new granular API endpoints that will replace the current monolithic settings transfer approach. These endpoints enable efficient, selective settings management with dot-notation path addressing.

## Design Philosophy

### Dot-Notation Path Addressing
Settings are addressed using dot-notation paths that correspond to the nested structure of the settings object:

```
visualisation.rendering.backgroundcolor
visualisation.graphs.logseq.physics.damping
system.network.port
xr.roomScale
```

### Minimal Data Transfer
- **GET requests**: Return only requested settings
- **POST requests**: Send only changed values
- **Response payloads**: Contain only affected settings

## New API Endpoints

### 1. Granular Settings Retrieval

#### `GET /api/settings/get`
Retrieve specific settings by path.

**Query Parameters:**
- `paths`: Comma-separated list of dot-notation paths
- `expand`: Optional. Include nested objects (`true`/`false`, default: `false`)

**Examples:**
```bash
# Get single setting
GET /api/settings/get?paths=visualisation.rendering.backgroundColor

# Get multiple settings
GET /api/settings/get?paths=system.network.port,xr.roomScale,visualisation.glow.enabled

# Get nested object
GET /api/settings/get?paths=visualisation.graphs.logseq&expand=true
```

**Response Format:**
```json
{
  "success": true,
  "data": {
    "visualisation": {
      "rendering": {
        "backgroundColor": "#000000"
      }
    },
    "system": {
      "network": {
        "port": 3000
      }
    },
    "xr": {
      "roomScale": 2.0
    }
  },
  "requestedPaths": ["visualisation.rendering.backgroundColor", "system.network.port", "xr.roomScale"],
  "timestamp": "2025-09-01T08:45:00Z"
}
```

### 2. Granular Settings Updates

#### `POST /api/settings/set`
Update specific settings using path-value pairs.

**Request Body:**
```json
{
  "updates": [
    {
      "path": "visualisation.rendering.backgroundColor",
      "value": "#FFFFFF"
    },
    {
      "path": "system.network.port", 
      "value": 8080
    },
    {
      "path": "visualisation.graphs.logseq.physics.damping",
      "value": 0.95
    }
  ],
  "validate": true  // Optional, default: true
}
```

**Response Format:**
```json
{
  "success": true,
  "updated": {
    "visualisation": {
      "rendering": {
        "backgroundColor": "#FFFFFF"
      },
      "graphs": {
        "logseq": {
          "physics": {
            "damping": 0.95
          }
        }
      }
    },
    "system": {
      "network": {
        "port": 8080
      }
    }
  },
  "errors": [],
  "timestamp": "2025-09-01T08:45:00Z",
  "affectedPaths": [
    "visualisation.rendering.backgroundColor",
    "system.network.port", 
    "visualisation.graphs.logseq.physics.damping"
  ]
}
```

### 3. Settings Schema Information

#### `GET /api/settings/schema`
Retrieve schema information for settings validation and UI generation.

**Query Parameters:**
- `path`: Optional. Get schema for specific path
- `type`: Optional. Filter by setting type (`string`, `number`, `boolean`, `object`)

**Response Format:**
```json
{
  "success": true,
  "schema": {
    "visualisation.rendering.backgroundColor": {
      "type": "string",
      "pattern": "^#[0-9A-Fa-f]{6}$",
      "default": "#000000",
      "description": "Background color for the rendering viewport"
    },
    "system.network.port": {
      "type": "integer",
      "minimum": 1024,
      "maximum": 65535,
      "default": 3000,
      "description": "HTTP server port number"
    }
  }
}
```

### 4. Settings Reset

#### `POST /api/settings/reset`
Reset specific settings to their default values.

**Request Body:**
```json
{
  "paths": ["visualisation.rendering.backgroundColor", "system.network.port"],
  "confirm": true  // Required to prevent accidental resets
}
```

**Response Format:**
```json
{
  "success": true,
  "reset": {
    "visualisation": {
      "rendering": {
        "backgroundColor": "#000000"
      }
    },
    "system": {
      "network": {
        "port": 3000
      }
    }
  },
  "resetPaths": ["visualisation.rendering.backgroundColor", "system.network.port"]
}
```

### 5. Bulk Operations

#### `POST /api/settings/bulk`
Perform multiple operations in a single request.

**Request Body:**
```json
{
  "operations": [
    {
      "type": "get",
      "paths": ["visualisation.glow.enabled", "xr.roomScale"]
    },
    {
      "type": "set",
      "updates": [
        {
          "path": "visualisation.rendering.backgroundColor",
          "value": "#222222"
        }
      ]
    },
    {
      "type": "reset",
      "paths": ["system.debug.enabled"]
    }
  ]
}
```

## Legacy Compatibility Endpoints

### Maintained During Transition

#### `GET /api/settings` (Legacy)
Returns complete settings object for backward compatibility.
- **Status**: Maintained during transition period
- **Performance**: Lower priority, may have longer response times
- **Deprecation**: Will be marked deprecated in v2.1, removed in v3.0

#### `POST /api/settings` (Legacy)
Accepts full or partial settings updates.
- **Status**: Maintained during transition period
- **Behavior**: Internally converts to granular operations
- **Deprecation**: Will be marked deprecated in v2.1, removed in v3.0

## Error Handling

### Error Response Format
```json
{
  "success": false,
  "error": {
    "code": "INVALID_PATH",
    "message": "Setting path 'invalid.path' does not exist",
    "details": {
      "invalidPath": "invalid.path",
      "suggestions": ["visualisation.path", "system.path"]
    }
  },
  "timestamp": "2025-09-01T08:45:00Z"
}
```

### Error Codes
- `INVALID_PATH`: Setting path does not exist
- `VALIDATION_ERROR`: Value fails validation rules
- `TYPE_MISMATCH`: Value type doesn't match expected type
- `PERMISSION_DENIED`: User lacks permission for this setting
- `READONLY_SETTING`: Attempting to modify read-only setting
- `RATE_LIMIT_EXCEEDED`: Too many requests

## Performance Characteristics

### Request/Response Sizes

#### Current (Monolithic)
- **GET /api/settings**: ~50-80KB response
- **POST /api/settings**: ~50-80KB request + response

#### New (Granular)
- **GET /api/settings/get**: ~0.1-5KB response (based on requested paths)
- **POST /api/settings/set**: ~0.1-2KB request + response

### Expected Performance Improvements
- **Network Usage**: 85-95% reduction in typical requests
- **Processing Time**: 60-70% faster for single setting changes
- **Cache Efficiency**: Much higher cache hit rates for selective requests

## Implementation Path Validation

### Valid Path Examples
```typescript
// Valid paths (dot-notation)
"visualisation.rendering.backgroundColor"
"visualisation.graphs.logseq.physics.damping"
"system.network.port"
"xr.controllers.leftHand.position.x"

// Array access
"visualisation.glow.sphereSizes.0"  // First element
"security.allowedOrigins.1"         // Second element

// Optional fields
"ragflow.apiKey"     // May be null/undefined
"whisper.timeout"    // Optional service setting
```

### Path Resolution Algorithm
1. **Split path** by '.' delimiter
2. **Navigate object** following path segments
3. **Handle arrays** using numeric indices
4. **Respect optional** fields (return null if not present)
5. **Type validation** against schema

## Frontend Integration

### New API Client Methods
```typescript
// New granular methods
async getSettings(paths: string[]): Promise<PartialSettings>
async setSettings(updates: SettingUpdate[]): Promise<PartialSettings>
async resetSettings(paths: string[]): Promise<PartialSettings>
async getSchema(path?: string): Promise<SettingsSchema>

// Bulk operations
async bulkOperation(operations: BulkOperation[]): Promise<BulkResponse>

// Utility methods
pathExists(path: string): boolean
validateValue(path: string, value: any): ValidationResult
getSuggestions(partialPath: string): string[]
```

### Integration with Zustand Store
```typescript
// Enhanced store methods
const useSettingsStore = create<SettingsState>((set, get) => ({
  // Selective loading
  loadSettings: async (paths: string[]) => {
    const settings = await settingsApi.getSettings(paths);
    set(state => ({ 
      settings: { ...state.settings, ...settings }
    }));
  },

  // Granular updates
  updateSetting: async (path: string, value: any) => {
    const updated = await settingsApi.setSettings([{ path, value }]);
    set(state => ({ 
      settings: mergeDeep(state.settings, updated)
    }));
  },

  // Batch operations
  batchUpdate: async (updates: SettingUpdate[]) => {
    const updated = await settingsApi.setSettings(updates);
    set(state => ({ 
      settings: mergeDeep(state.settings, updated)
    }));
  }
}));
```

## Migration Strategy

### Phase 1: Parallel Implementation
- Implement new granular endpoints alongside existing monolithic endpoints
- New endpoints return data in existing format for compatibility
- No breaking changes to existing API contracts

### Phase 2: Client Migration  
- Update frontend to use new granular endpoints
- Implement feature flags to control endpoint usage
- Add performance monitoring and comparison

### Phase 3: Optimization
- Enable compression on granular endpoints
- Implement intelligent caching strategies
- Add request bundling for multiple rapid updates

### Phase 4: Legacy Deprecation
- Mark monolithic endpoints as deprecated
- Add migration warnings to API responses
- Document migration paths for external clients

## Security Considerations

### Path Validation
- **Whitelist approach**: Only predefined paths are valid
- **Input sanitization**: Prevent path traversal attempts
- **Type enforcement**: Strict type checking on all values

### Permission System
```typescript
// Setting-level permissions
interface SettingPermission {
  path: string;
  read: boolean;
  write: boolean;
  reset: boolean;
  requiredRole?: string;
}

// Examples
const settingPermissions: SettingPermission[] = [
  {
    path: "system.network.port",
    read: true,
    write: false,  // Only admin can change
    reset: false,
    requiredRole: "admin"
  },
  {
    path: "visualisation.rendering.backgroundColor", 
    read: true,
    write: true,   // All users can change
    reset: true
  }
];
```

### Rate Limiting
- **Per-path limits**: Different limits for different setting categories
- **Burst protection**: Prevent rapid-fire setting changes
- **User-based limits**: Different limits for different user roles

## Testing Strategy

### Contract Testing
```typescript
// API contract tests
describe('Granular Settings API', () => {
  test('GET /api/settings/get returns requested paths only', async () => {
    const response = await request(app)
      .get('/api/settings/get?paths=system.network.port,xr.roomScale')
      .expect(200);
    
    expect(response.body.data).toHaveProperty('system.network.port');
    expect(response.body.data).toHaveProperty('xr.roomScale');
    expect(response.body.data).not.toHaveProperty('visualisation');
  });

  test('POST /api/settings/set updates only specified paths', async () => {
    const updates = [
      { path: 'visualisation.rendering.backgroundColor', value: '#FFFFFF' }
    ];
    
    const response = await request(app)
      .post('/api/settings/set')
      .send({ updates })
      .expect(200);
    
    expect(response.body.affectedPaths).toContain('visualisation.rendering.backgroundColor');
    expect(response.body.updated).toHaveProperty('visualisation.rendering.backgroundColor', '#FFFFFF');
  });
});
```

### Performance Testing
```typescript
// Performance benchmarks
describe('API Performance', () => {
  test('granular endpoint is faster than monolithic', async () => {
    const start = Date.now();
    
    // Test granular endpoint
    await request(app)
      .get('/api/settings/get?paths=system.network.port')
      .expect(200);
    
    const granularTime = Date.now() - start;
    
    // Test monolithic endpoint
    const monolithicStart = Date.now();
    await request(app)
      .get('/api/settings')
      .expect(200);
    
    const monolithicTime = Date.now() - monolithicStart;
    
    expect(granularTime).toBeLessThan(monolithicTime * 0.5); // At least 50% faster
  });
});
```

---

**Specification Version**: 1.0  
**Last Updated**: 2025-09-01  
**Next Review**: Before Phase 1 implementation