-- Database Seed Script: Complete Default Settings (camelCase only)
-- This script populates the settings table with comprehensive default configuration
-- All keys use camelCase format as per the application standard

-- Clear existing settings (optional - uncomment if needed)
-- DELETE FROM settings WHERE key = 'app_full_settings';

-- Insert complete default settings
INSERT INTO settings (key, value_type, value_json, description, created_at, updated_at)
VALUES (
  'app_full_settings',
  'json',
  '{
    "visualisation": {
      "rendering": {
        "backend": "webgpu",
        "quality": "high",
        "antialiasing": "msaa4x",
        "shadows": true,
        "ambientOcclusion": false,
        "postProcessing": true
      },
      "animations": {
        "enabled": true,
        "enableMotionBlur": false,
        "duration": 300,
        "easing": "easeInOut",
        "particleEffects": true,
        "transitionSpeed": "normal"
      },
      "sync": {
        "enabled": false,
        "camera": true,
        "selection": true,
        "viewport": false,
        "filters": true,
        "autoSync": false
      },
      "effects": {
        "bloom": false,
        "glow": true,
        "edgeHighlight": true,
        "depthOfField": false,
        "colorGrading": false,
        "vignette": false
      },
      "camera": {
        "type": "perspective",
        "fov": 60,
        "near": 0.1,
        "far": 1000,
        "defaultPosition": [0, 5, 10],
        "defaultTarget": [0, 0, 0],
        "enablePan": true,
        "enableZoom": true,
        "enableRotate": true,
        "zoomSpeed": 1.0,
        "panSpeed": 1.0,
        "rotateSpeed": 1.0
      },
      "graph": {
        "layout": "force",
        "nodeSize": 1.0,
        "edgeWidth": 0.1,
        "showLabels": true,
        "labelFontSize": 12,
        "maxNodes": 10000,
        "clusteringEnabled": false,
        "physicsEnabled": true
      }
    },
    "performance": {
      "autoOptimize": false,
      "simplifyEdges": true,
      "cullDistance": 50,
      "maxFrameRate": 60,
      "enableLOD": true,
      "lodLevels": 3,
      "chunkSize": 1000,
      "enableOcclusion": true,
      "memoryLimit": 2048,
      "gpuMemoryLimit": 1024,
      "workerThreads": 4,
      "enableWebWorkers": true,
      "cacheSize": 256,
      "preloadDistance": 20
    },
    "interaction": {
      "enableHover": true,
      "enableClick": true,
      "enableDrag": true,
      "enableSelect": true,
      "hoverDelay": 300,
      "clickDelay": 200,
      "dragThreshold": 5,
      "multiSelectKey": "ctrl",
      "contextMenuKey": "rightClick",
      "tooltipEnabled": true,
      "tooltipDelay": 500,
      "gesturesEnabled": false,
      "touchEnabled": true
    },
    "export": {
      "format": "json",
      "includeMetadata": true,
      "includeSettings": false,
      "compression": "none",
      "imageFormat": "png",
      "imageQuality": 0.9,
      "imageWidth": 1920,
      "imageHeight": 1080,
      "videoFormat": "mp4",
      "videoFPS": 30,
      "videoBitrate": 5000,
      "exportPath": "./exports",
      "filenameTemplate": "{name}_{timestamp}"
    },
    "ui": {
      "theme": "dark",
      "language": "en",
      "fontSize": 14,
      "iconSize": 24,
      "panelPosition": "right",
      "toolbarPosition": "top",
      "showMinimap": true,
      "showStats": false,
      "showGrid": false,
      "showAxes": false,
      "compactMode": false,
      "highContrastMode": false,
      "reducedMotion": false
    },
    "data": {
      "cacheEnabled": true,
      "cacheDuration": 3600,
      "maxCacheSize": 100,
      "autoRefresh": false,
      "refreshInterval": 60,
      "lazyLoad": true,
      "pagination": 100,
      "filterMode": "client",
      "sortMode": "client",
      "validateSchema": true
    },
    "network": {
      "timeout": 30000,
      "retries": 3,
      "retryDelay": 1000,
      "batchRequests": true,
      "batchSize": 10,
      "enableCompression": true,
      "enableCaching": true,
      "cacheStrategy": "staleWhileRevalidate",
      "maxConnections": 6,
      "keepAlive": true
    },
    "logging": {
      "level": "info",
      "enableConsole": true,
      "enableFile": false,
      "filePath": "./logs/app.log",
      "maxFileSize": 10485760,
      "maxFiles": 5,
      "format": "json",
      "includeTimestamp": true,
      "includeStackTrace": true,
      "logPerformance": false,
      "logNetworkRequests": false
    },
    "security": {
      "enableCSP": true,
      "enableCORS": false,
      "allowedOrigins": [],
      "enableXSSProtection": true,
      "enableClickjacking": true,
      "sessionTimeout": 3600,
      "maxLoginAttempts": 5,
      "lockoutDuration": 900,
      "requireStrongPassword": true,
      "passwordMinLength": 8,
      "enableMFA": false
    },
    "notifications": {
      "enabled": true,
      "position": "topRight",
      "duration": 5000,
      "showProgress": true,
      "sound": false,
      "desktop": false,
      "groupSimilar": true,
      "maxVisible": 3,
      "persistence": false
    }
  }',
  'Complete application settings with all namespaces in camelCase format',
  CURRENT_TIMESTAMP,
  CURRENT_TIMESTAMP
);

-- Insert individual namespace settings for quick access (optional)
INSERT INTO settings (key, value_type, value_json, description, created_at, updated_at)
VALUES
  ('visualisation_settings', 'json', (
    SELECT json_extract(value_json, '$.visualisation')
    FROM settings
    WHERE key = 'app_full_settings'
  ), 'Visualisation-specific settings', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),

  ('performance_settings', 'json', (
    SELECT json_extract(value_json, '$.performance')
    FROM settings
    WHERE key = 'app_full_settings'
  ), 'Performance optimization settings', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),

  ('interaction_settings', 'json', (
    SELECT json_extract(value_json, '$.interaction')
    FROM settings
    WHERE key = 'app_full_settings'
  ), 'User interaction settings', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),

  ('export_settings', 'json', (
    SELECT json_extract(value_json, '$.export')
    FROM settings
    WHERE key = 'app_full_settings'
  ), 'Export configuration settings', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);

-- Verify insertion
SELECT
  key,
  value_type,
  json_extract(value_json, '$.visualisation.sync.enabled') as sync_enabled,
  json_extract(value_json, '$.performance.autoOptimize') as auto_optimize,
  json_extract(value_json, '$.interaction.hoverDelay') as hover_delay,
  json_extract(value_json, '$.export.format') as export_format,
  description,
  created_at
FROM settings
WHERE key = 'app_full_settings';

-- Create indexes for faster JSON queries (PostgreSQL/SQLite compatible)
CREATE INDEX IF NOT EXISTS idx_settings_key ON settings(key);
CREATE INDEX IF NOT EXISTS idx_settings_value_type ON settings(value_type);

-- Comments for documentation
COMMENT ON TABLE settings IS 'Application configuration stored in camelCase format';
COMMENT ON COLUMN settings.value_json IS 'JSON object with camelCase keys for all configuration values';
