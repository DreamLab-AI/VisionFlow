-- Migration 001: Add 73 Missing Settings
-- Date: 2025-10-22
-- Description: Extends settings database with analytics, dashboard, performance, GPU visualization, bloom effects, developer tools, and agent control settings
-- Schema: Uses value_type with separate columns (value_text, value_integer, value_float, value_boolean, value_json)

-- ============================================================================
-- ANALYTICS SETTINGS (11 settings)
-- ============================================================================
INSERT INTO settings (key, parent_key, value_type, value_boolean, description) VALUES
  ('analytics.enableMetrics', 'app_full_settings', 'boolean', 1, 'Enable analytics metrics collection'),
  ('analytics.showDegreeDistribution', 'app_full_settings', 'boolean', 0, 'Show degree distribution graph'),
  ('analytics.showClusteringCoefficient', 'app_full_settings', 'boolean', 0, 'Show clustering coefficient'),
  ('analytics.showCentrality', 'app_full_settings', 'boolean', 0, 'Show centrality metrics'),
  ('analytics.clustering.exportEnabled', 'app_full_settings', 'boolean', 0, 'Enable cluster export functionality'),
  ('analytics.clustering.importEnabled', 'app_full_settings', 'boolean', 0, 'Enable distance matrix import');

INSERT INTO settings (key, parent_key, value_type, value_integer, description) VALUES
  ('analytics.updateInterval', 'app_full_settings', 'integer', 30, 'Analytics update interval in seconds'),
  ('analytics.clustering.clusterCount', 'app_full_settings', 'integer', 8, 'Number of clusters'),
  ('analytics.clustering.iterations', 'app_full_settings', 'integer', 50, 'Maximum clustering iterations');

INSERT INTO settings (key, parent_key, value_type, value_float, description) VALUES
  ('analytics.clustering.resolution', 'app_full_settings', 'float', 1.0, 'Clustering resolution parameter');

INSERT INTO settings (key, parent_key, value_type, value_text, description) VALUES
  ('analytics.clustering.algorithm', 'app_full_settings', 'string', 'kmeans', 'Clustering algorithm (kmeans, louvain, spectral)');

-- ============================================================================
-- DASHBOARD SETTINGS (8 settings)
-- ============================================================================
INSERT INTO settings (key, parent_key, value_type, value_boolean, description) VALUES
  ('dashboard.showStatus', 'app_full_settings', 'boolean', 1, 'Show graph status panel'),
  ('dashboard.autoRefresh', 'app_full_settings', 'boolean', 1, 'Auto-refresh dashboard data'),
  ('dashboard.showConvergence', 'app_full_settings', 'boolean', 1, 'Show convergence indicator'),
  ('dashboard.clusteringActive', 'app_full_settings', 'boolean', 0, 'Clustering currently active');

INSERT INTO settings (key, parent_key, value_type, value_integer, description) VALUES
  ('dashboard.refreshInterval', 'app_full_settings', 'integer', 5, 'Dashboard refresh interval in seconds'),
  ('dashboard.iterationCount', 'app_full_settings', 'integer', 0, 'Current physics iteration count'),
  ('dashboard.activeConstraints', 'app_full_settings', 'integer', 0, 'Number of active constraints');

INSERT INTO settings (key, parent_key, value_type, value_text, description) VALUES
  ('dashboard.computeMode', 'app_full_settings', 'string', 'Basic Force-Directed', 'Current GPU compute mode');

-- ============================================================================
-- PERFORMANCE SETTINGS (11 settings)
-- ============================================================================
INSERT INTO settings (key, parent_key, value_type, value_boolean, description) VALUES
  ('performance.showFPS', 'app_full_settings', 'boolean', 0, 'Show FPS counter in UI'),
  ('performance.enableAdaptiveQuality', 'app_full_settings', 'boolean', 1, 'Enable adaptive quality scaling'),
  ('performance.enableAdaptiveCooling', 'app_full_settings', 'boolean', 1, 'Enable adaptive cooling strategy'),
  ('performance.enableMemoryCoalescing', 'app_full_settings', 'boolean', 1, 'Enable GPU memory coalescing optimization');

INSERT INTO settings (key, parent_key, value_type, value_integer, description) VALUES
  ('performance.targetFPS', 'app_full_settings', 'integer', 60, 'Target frames per second'),
  ('performance.gpuMemoryLimit', 'app_full_settings', 'integer', 4096, 'GPU memory limit in megabytes'),
  ('performance.warmupDuration', 'app_full_settings', 'integer', 2, 'Physics warmup duration in seconds'),
  ('performance.gpuBlockSize', 'app_full_settings', 'integer', 256, 'CUDA block size for GPU operations'),
  ('performance.iterationLimit', 'app_full_settings', 'integer', 1000, 'Maximum physics iterations');

INSERT INTO settings (key, parent_key, value_type, value_float, description) VALUES
  ('performance.convergenceThreshold', 'app_full_settings', 'float', 0.01, 'Physics convergence threshold');

INSERT INTO settings (key, parent_key, value_type, value_text, description) VALUES
  ('performance.levelOfDetail', 'app_full_settings', 'string', 'high', 'Quality preset (low, medium, high, ultra)');

-- ============================================================================
-- GPU VISUALIZATION SETTINGS (8 settings)
-- ============================================================================
INSERT INTO settings (key, parent_key, value_type, value_boolean, description) VALUES
  ('gpu.visualization.heatmap.enabled', 'app_full_settings', 'boolean', 0, 'Enable GPU utilization heatmap'),
  ('gpu.visualization.heatmap.showLegend', 'app_full_settings', 'boolean', 1, 'Show heatmap color legend'),
  ('gpu.visualization.particleTrails.enabled', 'app_full_settings', 'boolean', 0, 'Enable particle motion trails');

INSERT INTO settings (key, parent_key, value_type, value_integer, description) VALUES
  ('gpu.visualization.heatmap.updateInterval', 'app_full_settings', 'integer', 100, 'Heatmap update interval in milliseconds'),
  ('gpu.visualization.particleTrails.length', 'app_full_settings', 'integer', 20, 'Particle trail length');

INSERT INTO settings (key, parent_key, value_type, value_float, description) VALUES
  ('gpu.visualization.particleTrails.fadeRate', 'app_full_settings', 'float', 0.95, 'Trail fade rate');

INSERT INTO settings (key, parent_key, value_type, value_text, description) VALUES
  ('gpu.visualization.heatmap.colorScheme', 'app_full_settings', 'string', 'viridis', 'Heatmap color scheme (viridis, plasma, inferno, magma)'),
  ('gpu.visualization.particleTrails.colorMode', 'app_full_settings', 'string', 'velocity', 'Trail coloring mode (velocity, acceleration, energy)');

-- ============================================================================
-- BLOOM EFFECT SETTINGS (4 settings)
-- ============================================================================
INSERT INTO settings (key, parent_key, value_type, value_boolean, description) VALUES
  ('effects.bloom.adaptiveThreshold', 'app_full_settings', 'boolean', 0, 'Enable adaptive bloom threshold');

INSERT INTO settings (key, parent_key, value_type, value_float, description) VALUES
  ('effects.bloom.threshold', 'app_full_settings', 'float', 0.8, 'Bloom brightness threshold'),
  ('effects.bloom.radius', 'app_full_settings', 'float', 0.5, 'Bloom effect radius'),
  ('effects.bloom.softness', 'app_full_settings', 'float', 0.3, 'Bloom edge softness');

-- ============================================================================
-- DEVELOPER SETTINGS (11 settings)
-- ============================================================================
INSERT INTO settings (key, parent_key, value_type, value_boolean, description) VALUES
  ('dev.debugMode', 'app_full_settings', 'boolean', 0, 'Enable debug mode'),
  ('dev.showBoundingBoxes', 'app_full_settings', 'boolean', 0, 'Show bounding boxes in visualization'),
  ('dev.showForceVectors', 'app_full_settings', 'boolean', 0, 'Show force vectors'),
  ('dev.enablePerformanceProfiling', 'app_full_settings', 'boolean', 0, 'Enable performance profiling'),
  ('dev.captureMetrics', 'app_full_settings', 'boolean', 0, 'Capture detailed performance metrics'),
  ('dev.exportMetrics', 'app_full_settings', 'boolean', 0, 'Enable metrics export'),
  ('dev.validateData', 'app_full_settings', 'boolean', 1, 'Enable data validation checks'),
  ('dev.strictMode', 'app_full_settings', 'boolean', 0, 'Enable strict validation mode'),
  ('dev.showMemoryUsage', 'app_full_settings', 'boolean', 0, 'Show memory usage statistics');

INSERT INTO settings (key, parent_key, value_type, value_integer, description) VALUES
  ('dev.metricsInterval', 'app_full_settings', 'integer', 1000, 'Metrics capture interval in milliseconds');

INSERT INTO settings (key, parent_key, value_type, value_text, description) VALUES
  ('dev.logLevel', 'app_full_settings', 'string', 'info', 'Console logging level (debug, info, warn, error)');

-- ============================================================================
-- AGENT CONTROL SETTINGS (20 settings)
-- ============================================================================
INSERT INTO settings (key, parent_key, value_type, value_boolean, description) VALUES
  ('agents.enableMemory', 'app_full_settings', 'boolean', 1, 'Enable agent memory storage'),
  ('agents.enableLearning', 'app_full_settings', 'boolean', 0, 'Enable autonomous learning'),
  ('agents.enableCoordination', 'app_full_settings', 'boolean', 1, 'Enable peer coordination'),
  ('agents.knowledgeSharing', 'app_full_settings', 'boolean', 1, 'Enable knowledge sharing between agents'),
  ('agents.enableHooks', 'app_full_settings', 'boolean', 1, 'Enable agent coordination hooks'),
  ('agents.sessionPersistence', 'app_full_settings', 'boolean', 1, 'Enable session state persistence'),
  ('agents.neuralEnabled', 'app_full_settings', 'boolean', 0, 'Enable neural network features'),
  ('agents.enableMetrics', 'app_full_settings', 'boolean', 1, 'Enable agent performance metrics'),
  ('agents.metricsExport', 'app_full_settings', 'boolean', 0, 'Export agent metrics');

INSERT INTO settings (key, parent_key, value_type, value_integer, description) VALUES
  ('agents.maxConcurrent', 'app_full_settings', 'integer', 4, 'Maximum concurrent agents'),
  ('agents.taskTimeout', 'app_full_settings', 'integer', 300, 'Agent task timeout in seconds'),
  ('agents.retryAttempts', 'app_full_settings', 'integer', 3, 'Number of retry attempts for failed tasks'),
  ('agents.healthCheckInterval', 'app_full_settings', 'integer', 30, 'Health check interval in seconds'),
  ('agents.hookTimeout', 'app_full_settings', 'integer', 10, 'Hook execution timeout in seconds'),
  ('agents.neuralTrainingInterval', 'app_full_settings', 'integer', 100, 'Neural training interval in iterations');

INSERT INTO settings (key, parent_key, value_type, value_float, description) VALUES
  ('agents.learningRate', 'app_full_settings', 'float', 0.01, 'Agent learning rate');

INSERT INTO settings (key, parent_key, value_type, value_text, description) VALUES
  ('agents.coordinationMode', 'app_full_settings', 'string', 'hierarchical', 'Agent coordination topology (mesh, hierarchical, ring, star)'),
  ('agents.memoryPersistence', 'app_full_settings', 'string', 'auto', 'Memory persistence mode (auto, memory, disk)'),
  ('agents.cognitivePattern', 'app_full_settings', 'string', 'adaptive', 'Cognitive thinking pattern (convergent, divergent, lateral, systems, critical, adaptive)'),
  ('agents.workflowStrategy', 'app_full_settings', 'string', 'adaptive', 'Workflow execution strategy (parallel, sequential, adaptive)');

-- ============================================================================
-- MIGRATION VALIDATION
-- ============================================================================
-- Expected totals:
-- Analytics: 11 settings
-- Dashboard: 8 settings
-- Performance: 11 settings
-- GPU Visualization: 8 settings
-- Bloom Effects: 4 settings
-- Developer: 11 settings
-- Agents: 20 settings
-- TOTAL: 73 settings
