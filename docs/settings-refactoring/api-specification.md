# Granular Settings API Specification

## Overview

This document specifies the complete API for the new granular settings system, including all endpoints, request/response formats, validation rules, and error handling procedures. The API uses UK English spelling throughout and follows RESTful principles with dot-notation path addressing.

## API Principles

### Granular Access
- **Selective Loading**: Request only the settings you need
- **Minimal Payloads**: Transfer only changed or requested data
- **Path-Based Operations**: Use dot-notation for precise targeting

### Type Safety
- **Strong Validation**: All values validated against schema
- **Consistent Format**: CamelCase JSON, snake_case internal storage
- **Error Clarity**: Detailed validation error messages with suggestions

### Performance
- **Caching Strategy**: Intelligent server-side and client-side caching
- **Batch Operations**: Multiple operations in single requests
- **Compression**: Automatic response compression for efficiency

## Base URL

All API endpoints are prefixed with:
```
https://api.visionflow.com/api/settings
```

For development:
```
http://localhost:3001/api/settings
```

## Authentication

All requests require authentication via Bearer token:

```http
Authorization: Bearer <your-access-token>
```

## Core Endpoints

### 1. Get Settings by Path

Retrieve specific settings using dot-notation paths.

#### `GET /get`

**Query Parameters:**
- `paths` (required): Comma-separated list of setting paths
- `expand` (optional): Include nested objects (`true`/`false`, default: `false`)
- `format` (optional): Response format (`json`/`compact`, default: `json`)

**Request Examples:**
```bash
# Get single setting
GET /api/settings/get?paths=visualisation.rendering.backgroundColour

# Get multiple settings
GET /api/settings/get?paths=system.network.port,xr.roomScale,visualisation.glow.enabled

# Get nested object with expansion
GET /api/settings/get?paths=visualisation.graphs.logseq&expand=true

# Get compact response format
GET /api/settings/get?paths=system.debug.enabled&format=compact
```

**Response Format:**
```json
{
  "success": true,
  "data": {
    "visualisation": {
      "rendering": {
        "backgroundColour": "#000000"
      },
      "glow": {
        "enabled": true
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
  "requestedPaths": [
    "visualisation.rendering.backgroundColour",
    "system.network.port", 
    "xr.roomScale",
    "visualisation.glow.enabled"
  ],
  "metadata": {
    "responseSize": 1247,
    "processingTime": 23,
    "cacheHit": false
  },
  "timestamp": "2025-09-01T09:15:00.000Z"
}
```

**Compact Response Format:**
```json
{
  "success": true,
  "values": {
    "visualisation.rendering.backgroundColour": "#000000",
    "system.network.port": 3000,
    "xr.roomScale": 2.0,
    "visualisation.glow.enabled": true
  },
  "timestamp": "2025-09-01T09:15:00.000Z"
}
```

### 2. Update Settings by Path

Update specific settings using path-value pairs.

#### `POST /set`

**Request Body:**
```json
{
  "updates": [
    {
      "path": "visualisation.rendering.backgroundColour",
      "value": "#FFFFFF"
    },
    {
      "path": "system.network.port",
      "value": 8080
    },
    {
      "path": "visualisation.graphs.logseq.physics.dampingFactor",
      "value": 0.95
    }
  ],
  "validate": true,
  "persist": true,
  "notify": {
    "websocket": true,
    "webhook": false
  }
}
```

**Request Parameters:**
- `updates` (required): Array of setting updates
- `validate` (optional): Validate before applying (default: `true`)
- `persist` (optional): Save to storage (default: `true`)
- `notify` (optional): Notification settings

**Response Format:**
```json
{
  "success": true,
  "updated": {
    "visualisation": {
      "rendering": {
        "backgroundColour": "#FFFFFF"
      },
      "graphs": {
        "logseq": {
          "physics": {
            "dampingFactor": 0.95
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
  "changes": [
    {
      "path": "visualisation.rendering.backgroundColour",
      "oldValue": "#000000",
      "newValue": "#FFFFFF",
      "type": "modified"
    },
    {
      "path": "system.network.port",
      "oldValue": 3000,
      "newValue": 8080,
      "type": "modified"
    },
    {
      "path": "visualisation.graphs.logseq.physics.dampingFactor",
      "oldValue": 0.9,
      "newValue": 0.95,
      "type": "modified"
    }
  ],
  "affectedPaths": [
    "visualisation.rendering.backgroundColour",
    "system.network.port",
    "visualisation.graphs.logseq.physics.dampingFactor"
  ],
  "metadata": {
    "validationTime": 5,
    "updateTime": 12,
    "persistTime": 8
  },
  "timestamp": "2025-09-01T09:15:00.000Z"
}
```

### 3. Schema Information

Retrieve validation schema and metadata for settings.

#### `GET /schema`

**Query Parameters:**
- `path` (optional): Get schema for specific path
- `type` (optional): Filter by setting type (`string`, `number`, `boolean`, `object`)
- `category` (optional): Filter by category (`physics`, `rendering`, `system`)
- `format` (optional): Schema format (`jsonschema`, `openapi`, default: `jsonschema`)

**Request Examples:**
```bash
# Get complete schema
GET /api/settings/schema

# Get schema for specific path
GET /api/settings/schema?path=visualisation.rendering.backgroundColour

# Get schema for specific type
GET /api/settings/schema?type=number&category=physics

# Get OpenAPI format schema
GET /api/settings/schema?format=openapi
```

**Response Format:**
```json
{
  "success": true,
  "schema": {
    "visualisation.rendering.backgroundColour": {
      "type": "string",
      "pattern": "^#[0-9A-Fa-f]{6}$",
      "default": "#000000",
      "description": "Background colour for the rendering viewport",
      "category": "rendering",
      "uiComponent": "colorPicker",
      "examples": ["#000000", "#FFFFFF", "#FF0000"],
      "validation": {
        "required": false,
        "minLength": 7,
        "maxLength": 7
      }
    },
    "system.network.port": {
      "type": "integer",
      "minimum": 1024,
      "maximum": 65535,
      "default": 3000,
      "description": "HTTP server port number",
      "category": "system",
      "uiComponent": "numberInput",
      "validation": {
        "required": true,
        "customValidator": "validatePort"
      }
    },
    "visualisation.graphs.logseq.physics.dampingFactor": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "default": 0.9,
      "description": "Energy dissipation rate for physics simulation",
      "category": "physics", 
      "uiComponent": "slider",
      "step": 0.01,
      "precision": 3,
      "validation": {
        "required": false,
        "customValidator": "validatePhysicsParameter"
      }
    }
  },
  "metadata": {
    "totalPaths": 247,
    "categories": ["physics", "rendering", "system", "xr", "auth"],
    "types": ["string", "number", "boolean", "object", "array"],
    "schemaVersion": "2.0"
  },
  "timestamp": "2025-09-01T09:15:00.000Z"
}
```

### 4. Reset Settings

Reset specific settings to their default values.

#### `POST /reset`

**Request Body:**
```json
{
  "paths": [
    "visualisation.rendering.backgroundColour",
    "system.network.port",
    "visualisation.graphs.logseq.physics"
  ],
  "confirm": true,
  "scope": "specified",
  "backup": true
}
```

**Request Parameters:**
- `paths` (required): Array of setting paths to reset
- `confirm` (required): Must be `true` to prevent accidental resets
- `scope` (optional): Reset scope (`specified`, `category`, `all`, default: `specified`)
- `backup` (optional): Create backup before reset (default: `true`)

**Response Format:**
```json
{
  "success": true,
  "reset": {
    "visualisation": {
      "rendering": {
        "backgroundColour": "#000000"
      },
      "graphs": {
        "logseq": {
          "physics": {
            "springStrength": 0.3,
            "repulsionStrength": 150.0,
            "dampingFactor": 0.9,
            "centralForce": 10.0,
            "linkDistance": 50.0,
            "linkStrength": 0.7,
            "chargeStrength": -30.0,
            "collisionRadius": 5.0,
            "alphaDecay": 0.028,
            "velocityDecay": 0.4
          }
        }
      }
    },
    "system": {
      "network": {
        "port": 3000
      }
    }
  },
  "resetPaths": [
    "visualisation.rendering.backgroundColour",
    "system.network.port",
    "visualisation.graphs.logseq.physics.springStrength",
    "visualisation.graphs.logseq.physics.repulsionStrength",
    "visualisation.graphs.logseq.physics.dampingFactor",
    "visualisation.graphs.logseq.physics.centralForce",
    "visualisation.graphs.logseq.physics.linkDistance",
    "visualisation.graphs.logseq.physics.linkStrength",
    "visualisation.graphs.logseq.physics.chargeStrength",
    "visualisation.graphs.logseq.physics.collisionRadius",
    "visualisation.graphs.logseq.physics.alphaDecay",
    "visualisation.graphs.logseq.physics.velocityDecay"
  ],
  "backupId": "backup_2025_09_01_09_15_00",
  "metadata": {
    "resetCount": 12,
    "backupSize": 15847
  },
  "timestamp": "2025-09-01T09:15:00.000Z"
}
```

### 5. Bulk Operations

Perform multiple operations in a single request for efficiency.

#### `POST /bulk`

**Request Body:**
```json
{
  "operations": [
    {
      "type": "get",
      "paths": ["visualisation.glow.enabled", "xr.roomScale"],
      "expand": false
    },
    {
      "type": "set",
      "updates": [
        {
          "path": "visualisation.rendering.backgroundColour",
          "value": "#222222"
        },
        {
          "path": "system.debug.enabled",
          "value": true
        }
      ]
    },
    {
      "type": "reset",
      "paths": ["visualisation.graphs.logseq.physics.springStrength"],
      "confirm": true
    }
  ],
  "atomic": false,
  "validate": true
}
```

**Response Format:**
```json
{
  "success": true,
  "results": [
    {
      "type": "get",
      "success": true,
      "data": {
        "visualisation": {
          "glow": {
            "enabled": true
          }
        },
        "xr": {
          "roomScale": 2.0
        }
      }
    },
    {
      "type": "set", 
      "success": true,
      "updated": {
        "visualisation": {
          "rendering": {
            "backgroundColour": "#222222"
          }
        },
        "system": {
          "debug": {
            "enabled": true
          }
        }
      },
      "affectedPaths": [
        "visualisation.rendering.backgroundColour",
        "system.debug.enabled"
      ]
    },
    {
      "type": "reset",
      "success": true,
      "reset": {
        "visualisation": {
          "graphs": {
            "logseq": {
              "physics": {
                "springStrength": 0.3
              }
            }
          }
        }
      },
      "resetPaths": ["visualisation.graphs.logseq.physics.springStrength"]
    }
  ],
  "metadata": {
    "totalOperations": 3,
    "successCount": 3,
    "errorCount": 0,
    "processingTime": 45
  },
  "timestamp": "2025-09-01T09:15:00.000Z"
}
```

### 6. Validate Settings

Validate setting values without applying changes.

#### `POST /validate`

**Request Body:**
```json
{
  "validations": [
    {
      "path": "visualisation.rendering.backgroundColour",
      "value": "#ZZZZZZ"
    },
    {
      "path": "system.network.port",
      "value": 100
    },
    {
      "path": "visualisation.graphs.logseq.physics.springStrength",
      "value": 1.5
    }
  ]
}
```

**Response Format:**
```json
{
  "success": false,
  "validations": [
    {
      "path": "visualisation.rendering.backgroundColour",
      "value": "#ZZZZZZ",
      "valid": false,
      "errors": [
        {
          "code": "INVALID_FORMAT",
          "message": "Background colour must be a valid hex colour code",
          "expected": "^#[0-9A-Fa-f]{6}$",
          "suggestions": ["#000000", "#FFFFFF", "#FF0000"]
        }
      ]
    },
    {
      "path": "system.network.port",
      "value": 100,
      "valid": false,
      "errors": [
        {
          "code": "OUT_OF_RANGE",
          "message": "Port must be between 1024 and 65535",
          "minimum": 1024,
          "maximum": 65535,
          "suggestions": [3000, 8080, 8443]
        }
      ]
    },
    {
      "path": "visualisation.graphs.logseq.physics.springStrength",
      "value": 1.5,
      "valid": false,
      "errors": [
        {
          "code": "OUT_OF_RANGE",
          "message": "Spring strength must be between 0.0 and 1.0",
          "minimum": 0.0,
          "maximum": 1.0,
          "suggestions": [0.3, 0.5, 0.7, 1.0]
        }
      ]
    }
  ],
  "summary": {
    "totalValidations": 3,
    "validCount": 0,
    "invalidCount": 3
  },
  "timestamp": "2025-09-01T09:15:00.000Z"
}
```

## Path Specification

### Dot-Notation Syntax

Settings are addressed using hierarchical dot-notation paths:

```
visualisation.rendering.backgroundColour
visualisation.graphs.logseq.physics.dampingFactor
system.network.port
xr.controllers.leftHand.position.x
```

### Array Access

Array elements are accessed using numeric indices:

```
visualisation.glow.sphereSizes.0          # First element
security.allowedOrigins.1                 # Second element
visualisation.hologram.geometries.2.type  # Third hologram type
```

### Optional Fields

Optional settings may return `null` or be omitted:

```
ragflow.apiKey        # May be null if not configured
whisper.timeout       # Optional service setting
openai.model          # May not exist for all users
```

### Path Validation

All paths are validated against a whitelist:

```typescript
interface PathValidation {
  path: string;
  exists: boolean;
  readable: boolean;
  writable: boolean;
  type: 'string' | 'number' | 'boolean' | 'object' | 'array';
  required: boolean;
  deprecated?: boolean;
  alternatives?: string[];
}
```

## Validation Rules

### Data Type Validation

#### String Settings
```json
{
  "type": "string",
  "minLength": 1,
  "maxLength": 255,
  "pattern": "^#[0-9A-Fa-f]{6}$",  // For colour codes
  "enum": ["low", "medium", "high"]  // For quality settings
}
```

#### Numeric Settings
```json
{
  "type": "number",
  "minimum": 0.0,
  "maximum": 1.0,
  "multipleOf": 0.01,  // For precision control
  "exclusiveMinimum": true  // For ranges that exclude boundary
}
```

#### Boolean Settings
```json
{
  "type": "boolean",
  "default": false
}
```

#### Object Settings
```json
{
  "type": "object",
  "required": ["x", "y"],
  "additionalProperties": false,
  "properties": {
    "x": {"type": "number"},
    "y": {"type": "number"},
    "z": {"type": "number", "default": 0}
  }
}
```

### Physics Parameter Validation

Specific validation rules for physics simulation parameters:

```json
{
  "visualisation.graphs.*.physics.springStrength": {
    "type": "number",
    "minimum": 0.0,
    "maximum": 1.0,
    "default": 0.3,
    "description": "Controls the force of springs connecting nodes",
    "physicsCategory": "attraction",
    "performanceImpact": "medium",
    "validation": {
      "warning": {
        "threshold": 0.8,
        "message": "High spring strength may cause simulation instability"
      }
    }
  },
  "visualisation.graphs.*.physics.repulsionStrength": {
    "type": "number", 
    "minimum": 0.0,
    "maximum": 500.0,
    "default": 150.0,
    "description": "Repulsive force between all nodes",
    "physicsCategory": "repulsion",
    "performanceImpact": "high",
    "validation": {
      "warning": {
        "threshold": 400.0,
        "message": "Very high repulsion may impact performance"
      }
    }
  }
}
```

### System Setting Validation

Validation for system-level configuration:

```json
{
  "system.network.port": {
    "type": "integer",
    "minimum": 1024,
    "maximum": 65535,
    "default": 3000,
    "description": "HTTP server port number",
    "validation": {
      "custom": "validatePort",
      "requirements": [
        "Must not conflict with system ports (< 1024)",
        "Must not be currently in use",
        "Must be accessible to the application"
      ]
    }
  },
  "system.websocket.maxConnections": {
    "type": "integer",
    "minimum": 1,
    "maximum": 10000,
    "default": 1000,
    "description": "Maximum concurrent WebSocket connections",
    "validation": {
      "warning": {
        "threshold": 5000,
        "message": "High connection limits may impact server performance"
      }
    }
  }
}
```

## Error Handling

### Error Response Format

All errors follow a consistent structure:

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Multiple validation errors occurred",
    "details": {
      "errors": [
        {
          "path": "visualisation.rendering.backgroundColour",
          "code": "INVALID_FORMAT",
          "message": "Background colour must be a valid hex colour code",
          "value": "#ZZZZZZ",
          "expected": "^#[0-9A-Fa-f]{6}$",
          "suggestions": ["#000000", "#FFFFFF", "#FF0000"]
        }
      ],
      "requestId": "req_2025_09_01_12_34_56_789",
      "context": {
        "endpoint": "/api/settings/set",
        "userId": "user_123",
        "timestamp": "2025-09-01T09:15:00.000Z"
      }
    }
  },
  "timestamp": "2025-09-01T09:15:00.000Z"
}
```

### Error Codes

#### Client Errors (4xx)

| Code | HTTP Status | Description | Resolution |
|------|-------------|-------------|------------|
| `INVALID_PATH` | 400 | Setting path does not exist | Check path spelling and schema |
| `VALIDATION_ERROR` | 400 | Value fails validation rules | Review value against constraints |
| `TYPE_MISMATCH` | 400 | Value type doesn't match expected | Check schema for correct type |
| `MISSING_REQUIRED` | 400 | Required field is missing | Provide all required fields |
| `MALFORMED_REQUEST` | 400 | Request body is invalid JSON | Validate JSON syntax |
| `READONLY_SETTING` | 403 | Attempting to modify read-only setting | Use read-only settings for display only |
| `PERMISSION_DENIED` | 403 | User lacks permission for this setting | Check user permissions |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests | Implement request throttling |

#### Server Errors (5xx)

| Code | HTTP Status | Description | Resolution |
|------|-------------|-------------|------------|
| `ACTOR_ERROR` | 500 | Settings actor communication failure | Retry request, check server status |
| `PERSISTENCE_ERROR` | 500 | Failed to save settings | Check disk space and permissions |
| `VALIDATION_SERVICE_ERROR` | 500 | Validation service unavailable | Retry with validation disabled |
| `SCHEMA_ERROR` | 500 | Schema loading failure | Check schema file integrity |
| `INTERNAL_ERROR` | 500 | Unexpected server error | Contact support with request ID |

### Error Handling Examples

#### Validation Error Response
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Physics parameter out of range",
    "details": {
      "path": "visualisation.graphs.logseq.physics.springStrength",
      "value": 1.5,
      "constraints": {
        "minimum": 0.0,
        "maximum": 1.0,
        "type": "number"
      },
      "suggestion": "Use a value between 0.0 and 1.0. Common values are 0.3 (loose), 0.5 (medium), 0.7 (tight)"
    }
  }
}
```

#### Path Not Found Error
```json
{
  "success": false,
  "error": {
    "code": "INVALID_PATH", 
    "message": "Setting path does not exist",
    "details": {
      "invalidPath": "visualisation.rendering.backgroundColor",
      "suggestions": [
        "visualisation.rendering.backgroundColour",
        "visualisation.rendering.ambientLightIntensity",
        "visualisation.rendering.quality"
      ],
      "category": "rendering",
      "didYouMean": "visualisation.rendering.backgroundColour"
    }
  }
}
```

#### Permission Denied Error
```json
{
  "success": false,
  "error": {
    "code": "PERMISSION_DENIED",
    "message": "Insufficient permissions to modify system settings",
    "details": {
      "path": "system.network.port",
      "requiredRole": "admin",
      "userRole": "user",
      "requiredPermissions": ["SYSTEM_SETTINGS_WRITE"],
      "userPermissions": ["VISUALISATION_SETTINGS_WRITE", "PHYSICS_SETTINGS_WRITE"]
    }
  }
}
```

## Performance Characteristics

### Response Time Targets

| Operation | Target Response Time | Maximum Response Time |
|-----------|---------------------|----------------------|
| Single GET | < 50ms | < 200ms |
| Multiple GET (< 10 paths) | < 100ms | < 300ms |
| Single SET | < 100ms | < 500ms |
| Multiple SET (< 10 updates) | < 200ms | < 800ms |
| Schema GET | < 30ms | < 100ms |
| Bulk operations | < 300ms | < 1000ms |

### Payload Size Comparison

| Operation | Legacy API | Granular API | Reduction |
|-----------|------------|--------------|-----------|
| Get single setting | 50-80KB | 0.1-1KB | 98-99% |
| Update single setting | 50-80KB | 0.1-2KB | 95-98% |
| Get physics settings | 50-80KB | 2-5KB | 90-95% |
| Get all rendering settings | 50-80KB | 5-15KB | 70-85% |

### Caching Strategy

#### Server-Side Caching
- **Memory Cache**: Frequently accessed settings cached for 5 minutes
- **Redis Cache**: Distributed cache for multi-instance deployments
- **Schema Cache**: Setting schemas cached until restart

#### Client-Side Caching
- **Browser Cache**: ETag-based HTTP caching for 2 minutes
- **Application Cache**: Zustand persist middleware for session storage
- **Service Worker**: Background cache updates for offline support

#### Cache Invalidation
```json
{
  "cacheInvalidation": {
    "immediate": ["system.network", "system.security"],
    "delayed": ["visualisation.rendering", "visualisation.animations"],
    "manual": ["auth.tokens", "auth.permissions"]
  }
}
```

## Rate Limiting

### Rate Limit Rules

| Endpoint | Rate Limit | Burst Limit | Window |
|----------|------------|-------------|--------|
| GET /get | 100 req/min | 20 req/10s | 1 minute |
| POST /set | 60 req/min | 10 req/10s | 1 minute |
| POST /bulk | 20 req/min | 5 req/10s | 1 minute |
| GET /schema | 30 req/min | 10 req/10s | 1 minute |
| POST /validate | 100 req/min | 20 req/10s | 1 minute |
| POST /reset | 5 req/min | 2 req/10s | 1 minute |

### Rate Limit Headers

Responses include rate limit information:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1693555200
X-RateLimit-RetryAfter: 60
```

### Rate Limit Error Response

```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Too many requests",
    "details": {
      "limit": 100,
      "remaining": 0,
      "resetTime": "2025-09-01T09:16:00.000Z",
      "retryAfter": 60
    }
  }
}
```

## Monitoring and Analytics

### Performance Metrics

The API collects comprehensive metrics:

```json
{
  "metrics": {
    "responseTime": {
      "p50": 45,
      "p95": 120,
      "p99": 300
    },
    "throughput": {
      "requestsPerSecond": 150,
      "requestsPerMinute": 9000
    },
    "errorRate": {
      "total": 2.3,
      "by_code": {
        "VALIDATION_ERROR": 1.8,
        "INVALID_PATH": 0.3,
        "RATE_LIMIT_EXCEEDED": 0.2
      }
    },
    "cacheMetrics": {
      "hitRate": 0.85,
      "missRate": 0.15,
      "evictionRate": 0.02
    }
  }
}
```

### Health Check Endpoint

#### `GET /health`

```json
{
  "status": "healthy",
  "version": "2.0.0",
  "components": {
    "settings_actor": "healthy",
    "validation_service": "healthy", 
    "persistence_layer": "healthy",
    "cache_service": "healthy"
  },
  "metrics": {
    "uptime": 86400,
    "memoryUsage": 0.65,
    "cpuUsage": 0.23
  },
  "timestamp": "2025-09-01T09:15:00.000Z"
}
```

---

**API Specification Version**: 2.0  
**Last Updated**: 2025-09-01  
**Maintained By**: API Documentation Team