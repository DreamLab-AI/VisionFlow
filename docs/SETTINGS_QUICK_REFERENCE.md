# Settings Quick Reference Guide

## Overview

After Migration 001, the database now contains **78 total settings** (5 original + 73 new), organized into 7 functional categories.

## Quick Access Queries

### Get All Settings by Category

```sql
-- Analytics Settings
SELECT key, value_type,
       COALESCE(value_text, CAST(value_integer AS TEXT),
                CAST(value_float AS TEXT),
                CASE value_boolean WHEN 1 THEN 'true' ELSE 'false' END) as value,
       description
FROM settings
WHERE key LIKE 'analytics.%'
ORDER BY key;

-- Agent Settings
SELECT key, value_type,
       COALESCE(value_text, CAST(value_integer AS TEXT),
                CAST(value_float AS TEXT),
                CASE value_boolean WHEN 1 THEN 'true' ELSE 'false' END) as value,
       description
FROM settings
WHERE key LIKE 'agents.%'
ORDER BY key;
```

### Get Settings by Value Type

```sql
-- All boolean toggles
SELECT key, value_boolean, description
FROM settings
WHERE value_type = 'boolean'
ORDER BY key;

-- All numeric settings with ranges
SELECT key, value_integer, value_float, description
FROM settings
WHERE value_type IN ('integer', 'float')
ORDER BY key;
```

## Category Reference

### 1. Analytics (11 settings)

**Prefix**: `analytics.*`

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `enableMetrics` | boolean | true | Master toggle for analytics |
| `updateInterval` | integer | 30 | Update frequency in seconds |
| `showDegreeDistribution` | boolean | false | Show degree distribution graph |
| `showClusteringCoefficient` | boolean | false | Show clustering metrics |
| `showCentrality` | boolean | false | Show centrality analysis |
| `clustering.algorithm` | string | kmeans | Algorithm (kmeans/louvain/spectral) |
| `clustering.clusterCount` | integer | 8 | Number of clusters (2-50) |
| `clustering.resolution` | float | 1.0 | Resolution parameter (0.1-5.0) |
| `clustering.iterations` | integer | 50 | Max iterations (10-500) |
| `clustering.exportEnabled` | boolean | false | Enable cluster export |
| `clustering.importEnabled` | boolean | false | Enable distance matrix import |

### 2. Dashboard (8 settings)

**Prefix**: `dashboard.*`

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `showStatus` | boolean | true | Show status panel |
| `autoRefresh` | boolean | true | Auto-refresh data |
| `refreshInterval` | integer | 5 | Refresh interval (seconds) |
| `computeMode` | string | "Basic Force-Directed" | Current GPU compute mode |
| `iterationCount` | integer | 0 | Current iteration count |
| `showConvergence` | boolean | true | Show convergence indicator |
| `activeConstraints` | integer | 0 | Active constraints count |
| `clusteringActive` | boolean | false | Clustering status |

### 3. Performance (11 settings)

**Prefix**: `performance.*`

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `showFPS` | boolean | false | Show FPS counter |
| `targetFPS` | integer | 60 | Target frame rate (30-144) |
| `gpuMemoryLimit` | integer | 4096 | GPU memory limit (MB) |
| `levelOfDetail` | string | "high" | Quality preset |
| `enableAdaptiveQuality` | boolean | true | Adaptive quality scaling |
| `warmupDuration` | integer | 2 | Physics warmup (seconds) |
| `convergenceThreshold` | float | 0.01 | Physics convergence (0.001-0.1) |
| `enableAdaptiveCooling` | boolean | true | Adaptive cooling strategy |
| `gpuBlockSize` | integer | 256 | CUDA block size (64-1024) |
| `enableMemoryCoalescing` | boolean | true | Memory coalescing |
| `iterationLimit` | integer | 1000 | Max physics iterations |

### 4. GPU Visualization (8 settings)

**Prefix**: `gpu.visualization.*`

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `heatmap.enabled` | boolean | false | GPU utilization heatmap |
| `heatmap.updateInterval` | integer | 100 | Update interval (ms) |
| `heatmap.colorScheme` | string | "viridis" | Color scheme |
| `heatmap.showLegend` | boolean | true | Show color legend |
| `particleTrails.enabled` | boolean | false | Particle motion trails |
| `particleTrails.length` | integer | 20 | Trail length (5-100) |
| `particleTrails.fadeRate` | float | 0.95 | Fade rate (0.5-0.99) |
| `particleTrails.colorMode` | string | "velocity" | Color mode |

### 5. Bloom Effects (4 settings)

**Prefix**: `effects.bloom.*`

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `threshold` | float | 0.8 | Brightness threshold (0.0-1.0) |
| `radius` | float | 0.5 | Effect radius (0.0-2.0) |
| `softness` | float | 0.3 | Edge softness (0.0-1.0) |
| `adaptiveThreshold` | boolean | false | Adaptive threshold mode |

### 6. Developer (11 settings)

**Prefix**: `dev.*`

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `debugMode` | boolean | false | Enable debug mode |
| `showBoundingBoxes` | boolean | false | Show bounding boxes |
| `showForceVectors` | boolean | false | Show force vectors |
| `enablePerformanceProfiling` | boolean | false | Performance profiling |
| `logLevel` | string | "info" | Logging level |
| `captureMetrics` | boolean | false | Capture metrics |
| `exportMetrics` | boolean | false | Export metrics |
| `metricsInterval` | integer | 1000 | Metrics interval (ms) |
| `validateData` | boolean | true | Data validation |
| `strictMode` | boolean | false | Strict validation |
| `showMemoryUsage` | boolean | false | Memory usage stats |

### 7. Agents (20 settings)

**Prefix**: `agents.*`

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `maxConcurrent` | integer | 4 | Max concurrent agents (1-16) |
| `coordinationMode` | string | "hierarchical" | Topology type |
| `enableMemory` | boolean | true | Memory storage |
| `memoryPersistence` | string | "auto" | Persistence mode |
| `enableLearning` | boolean | false | Autonomous learning |
| `learningRate` | float | 0.01 | Learning rate (0.001-0.1) |
| `cognitivePattern` | string | "adaptive" | Thinking pattern |
| `enableCoordination` | boolean | true | Peer coordination |
| `taskTimeout` | integer | 300 | Task timeout (seconds) |
| `retryAttempts` | integer | 3 | Retry attempts (0-10) |
| `healthCheckInterval` | integer | 30 | Health check interval |
| `enableMetrics` | boolean | true | Performance metrics |
| `metricsExport` | boolean | false | Metrics export |
| `knowledgeSharing` | boolean | true | Knowledge sharing |
| `workflowStrategy` | string | "adaptive" | Workflow strategy |
| `enableHooks` | boolean | true | Coordination hooks |
| `hookTimeout` | integer | 10 | Hook timeout (seconds) |
| `sessionPersistence` | boolean | true | Session persistence |
| `neuralEnabled` | boolean | false | Neural features |
| `neuralTrainingInterval` | integer | 100 | Training interval |

## Common Operations

### Update a Setting

```sql
-- Update boolean setting
UPDATE settings
SET value_boolean = 1, updated_at = CURRENT_TIMESTAMP
WHERE key = 'analytics.enableMetrics';

-- Update integer setting
UPDATE settings
SET value_integer = 16, updated_at = CURRENT_TIMESTAMP
WHERE key = 'agents.maxConcurrent';

-- Update string setting
UPDATE settings
SET value_text = 'mesh', updated_at = CURRENT_TIMESTAMP
WHERE key = 'agents.coordinationMode';

-- Update float setting
UPDATE settings
SET value_float = 0.05, updated_at = CURRENT_TIMESTAMP
WHERE key = 'agents.learningRate';
```

### Get Modified Settings

```sql
-- Settings modified in last hour
SELECT key, value_type, updated_at, description
FROM settings
WHERE updated_at >= datetime('now', '-1 hour')
ORDER BY updated_at DESC;

-- Recently changed settings with values
SELECT key,
       COALESCE(value_text, CAST(value_integer AS TEXT),
                CAST(value_float AS TEXT),
                CASE value_boolean WHEN 1 THEN 'true' ELSE 'false' END) as value,
       updated_at
FROM settings
WHERE updated_at > created_at
ORDER BY updated_at DESC;
```

### Export Settings

```sql
-- Export all settings as JSON-ready format
SELECT json_object(
    'key', key,
    'value', COALESCE(value_text, CAST(value_integer AS TEXT),
                      CAST(value_float AS TEXT),
                      CASE value_boolean WHEN 1 THEN 'true' ELSE 'false' END),
    'type', value_type,
    'description', description
) as setting_json
FROM settings
WHERE parent_key = 'app_full_settings'
ORDER BY key;
```

## Value Constraints

### String Enumerations

```javascript
// Analytics
analytics.clustering.algorithm: ['kmeans', 'louvain', 'spectral']

// Performance
performance.levelOfDetail: ['low', 'medium', 'high', 'ultra']

// GPU Visualization
gpu.visualization.heatmap.colorScheme: ['viridis', 'plasma', 'inferno', 'magma']
gpu.visualization.particleTrails.colorMode: ['velocity', 'acceleration', 'energy']

// Developer
dev.logLevel: ['debug', 'info', 'warn', 'error']

// Agents
agents.coordinationMode: ['mesh', 'hierarchical', 'ring', 'star']
agents.memoryPersistence: ['auto', 'memory', 'disk']
agents.cognitivePattern: ['convergent', 'divergent', 'lateral', 'systems', 'critical', 'adaptive']
agents.workflowStrategy: ['parallel', 'sequential', 'adaptive']
```

### Numeric Ranges

```javascript
// Analytics
analytics.updateInterval: 1-300 seconds
analytics.clustering.clusterCount: 2-50
analytics.clustering.resolution: 0.1-5.0
analytics.clustering.iterations: 10-500

// Performance
performance.targetFPS: 30-144
performance.gpuMemoryLimit: 256-16384 MB
performance.warmupDuration: 0-10 seconds
performance.convergenceThreshold: 0.001-0.1
performance.gpuBlockSize: 64-1024
performance.iterationLimit: 100-10000

// GPU Visualization
gpu.visualization.heatmap.updateInterval: 50-1000 ms
gpu.visualization.particleTrails.length: 5-100
gpu.visualization.particleTrails.fadeRate: 0.5-0.99

// Bloom Effects
effects.bloom.threshold: 0.0-1.0
effects.bloom.radius: 0.0-2.0
effects.bloom.softness: 0.0-1.0

// Developer
dev.metricsInterval: 100-10000 ms

// Agents
agents.maxConcurrent: 1-16
agents.taskTimeout: 10-3600 seconds
agents.retryAttempts: 0-10
agents.healthCheckInterval: 5-300 seconds
agents.hookTimeout: 1-60 seconds
agents.learningRate: 0.001-0.1
agents.neuralTrainingInterval: 10-1000
```

## Migration Info

- **Migration File**: `scripts/migrations/001_add_missing_settings.sql`
- **Migration Runner**: `scripts/run_migration.sh`
- **Results Documentation**: `docs/MIGRATION_001_RESULTS.md`
- **Settings Added**: 73
- **Total Settings**: 78
- **Execution Date**: 2025-10-22

## Database Schema

```sql
CREATE TABLE settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    parent_key TEXT,
    value_type TEXT NOT NULL CHECK(value_type IN ('string', 'integer', 'float', 'boolean', 'json')),
    value_text TEXT,
    value_integer INTEGER,
    value_float REAL,
    value_boolean INTEGER CHECK(value_boolean IN (0, 1)),
    value_json TEXT,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_key) REFERENCES settings(key) ON DELETE CASCADE
);
```

---

**Last Updated**: 2025-10-22
**Version**: 1.0.0 (Post-Migration 001)
