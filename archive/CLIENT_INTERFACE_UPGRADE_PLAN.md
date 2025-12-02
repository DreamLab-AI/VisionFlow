# VisionFlow Client Interface Upgrade Plan

**Version:** 2.0
**Date:** 2025-11-06 (Updated)
**Original Date:** 2025-11-05
**Branch:** `claude/audit-stubs-disconnected-011CUpLF5w9noyxx5uQBepeV`
**Based On:** CLIENT_SERVER_INTEGRATION_AUDIT.md
**Status:** ⚠️ SPRINT 1 & 2 COMPLETE, SPRINT 3 REMAINING

---

## ⚠️ STATUS UPDATE (November 6, 2025)

**Progress Achieved:**
- ✅ **Sprint 1 Complete** - Physics and Semantic control panels implemented
- ✅ **Sprint 2 Complete** - Advanced features and real-time monitoring
- ⏳ **Sprint 3 Remaining** - Polish, optimization, and final integration

**What's Been Built:**
- ✅ Physics Control Panel (~400 lines TypeScript)
- ✅ Semantic Analysis Panel (~350 lines)
- ✅ Inference Engine Panel (~300 lines)
- ✅ Multi-Graph Load interface
- ✅ Real-time monitoring dashboard
- **Total:** ~3,600 lines of new client code

**Remaining Work (Sprint 3):**
- [ ] Performance optimization
- [ ] Comprehensive testing
- [ ] Polish and UX refinements
- [ ] Documentation updates

**Current Status:** ~67% complete (Sprint 1 & 2 done)
**Estimated Effort Remaining:** 8-10 days for Sprint 3

**Reference:** CLIENT_MIGRATION_COMPLETE.md (archived), 

---

## Executive Summary

This document provides a comprehensive, step-by-step implementation plan to upgrade the VisionFlow client interface to achieve 100% feature parity with the server. The plan is organized into 3 sprints over 2-3 months.

**Original State:**
- Client feature parity: 79%
- Server endpoints: 85+
- Client API calls: 67
- Missing features: 18+ endpoints, 3 major feature sets

**Current State (Post Sprint 1 & 2):**
- Client feature parity: ~85-90%
- New UI panels: 4 major panels added
- Integration endpoints: +15 new endpoints used

**Target State:**
- Client feature parity: 100%
- All server features exposed in UI
- Modern, maintainable architecture
- Real-time monitoring and control

**Original Effort:** 26-40 developer days (2 developers, 2-3 sprints)
**Remaining Effort:** 8-10 days (Sprint 3)

---

## Sprint 1: Critical Features (Days 1-15)

**Goal:** Expose Physics and Semantic APIs to users
**Priority:** 🔴 CRITICAL
**Effort:** 11-15 days

---

### Task 1.1: Physics Control Panel (Days 1-5)

**Feature:** Complete physics simulation control from client UI

#### Files to Create:

**1. `client/src/features/physics/components/PhysicsControlPanel.tsx`** (300-400 lines)

```typescript
/**
 * Physics Control Panel
 *
 * Provides comprehensive controls for the physics simulation engine:
 * - Start/stop/pause simulation
 * - Adjust simulation parameters in real-time
 * - Pin/unpin nodes for manual layout
 * - Apply custom forces
 * - Optimize layout with different algorithms
 */

import React, { useState, useEffect } from 'react';
import { usePhysicsService } from '../hooks/usePhysicsService';
import { Button, Slider, Select, Card, Divider } from '@/components/ui';
import { Play, Pause, Square, RefreshCw, Pin, Zap } from 'lucide-react';

interface PhysicsControlPanelProps {
  visible?: boolean;
  onClose?: () => void;
}

export const PhysicsControlPanel: React.FC<PhysicsControlPanelProps> = ({
  visible = true,
  onClose,
}) => {
  const {
    status,
    parameters,
    startSimulation,
    stopSimulation,
    updateParameters,
    resetSimulation,
    optimizeLayout,
    applyForces,
    pinNodes,
    unpinNodes,
    performStep,
    loading,
    error,
  } = usePhysicsService();

  // State for parameter controls
  const [localParams, setLocalParams] = useState(parameters);

  // Sync local params with server
  useEffect(() => {
    setLocalParams(parameters);
  }, [parameters]);

  const handleParameterChange = (key: string, value: number) => {
    setLocalParams((prev) => ({ ...prev, [key]: value }));
  };

  const handleApplyParameters = async () => {
    await updateParameters(localParams);
  };

  return (
    <Card className="physics-control-panel">
      <div className="panel-header">
        <h3>Physics Simulation</h3>
        <StatusIndicator running={status?.running} />
      </div>

      <Divider />

      {/* Simulation Controls */}
      <div className="simulation-controls">
        <h4>Controls</h4>
        <div className="button-group">
          <Button
            icon={<Play />}
            onClick={startSimulation}
            disabled={status?.running || loading}
          >
            Start
          </Button>
          <Button
            icon={<Square />}
            onClick={stopSimulation}
            disabled={!status?.running || loading}
          >
            Stop
          </Button>
          <Button
            icon={<RefreshCw />}
            onClick={performStep}
            disabled={loading}
          >
            Step
          </Button>
          <Button
            icon={<RefreshCw />}
            onClick={resetSimulation}
            disabled={loading}
            variant="danger"
          >
            Reset
          </Button>
        </div>
      </div>

      <Divider />

      {/* Parameters */}
      <div className="parameters">
        <h4>Parameters</h4>

        <div className="parameter">
          <label>Spring Constant</label>
          <Slider
            min={0.1}
            max={2.0}
            step={0.1}
            value={localParams.spring_constant}
            onChange={(v) => handleParameterChange('spring_constant', v)}
          />
          <span>{localParams.spring_constant.toFixed(2)}</span>
        </div>

        <div className="parameter">
          <label>Damping</label>
          <Slider
            min={0.1}
            max={1.0}
            step={0.05}
            value={localParams.damping}
            onChange={(v) => handleParameterChange('damping', v)}
          />
          <span>{localParams.damping.toFixed(2)}</span>
        </div>

        <div className="parameter">
          <label>Repulsion Strength</label>
          <Slider
            min={0.5}
            max={3.0}
            step={0.1}
            value={localParams.repulsion_strength}
            onChange={(v) => handleParameterChange('repulsion_strength', v)}
          />
          <span>{localParams.repulsion_strength.toFixed(2)}</span>
        </div>

        <div className="parameter">
          <label>Attraction Strength</label>
          <Slider
            min={0.1}
            max={2.0}
            step={0.1}
            value={localParams.attraction_strength}
            onChange={(v) => handleParameterChange('attraction_strength', v)}
          />
          <span>{localParams.attraction_strength.toFixed(2)}</span>
        </div>

        <Button onClick={handleApplyParameters} disabled={loading}>
          Apply Parameters
        </Button>
      </div>

      <Divider />

      {/* Layout Optimization */}
      <div className="layout-optimization">
        <h4>Layout Optimization</h4>
        <Select
          value={optimizationAlgorithm}
          onChange={setOptimizationAlgorithm}
          options={[
            { label: 'Force-Directed', value: 'force_directed' },
            { label: 'Spring Embedder', value: 'spring_embedder' },
            { label: 'Fruchterman-Reingold', value: 'fruchterman_reingold' },
            { label: 'Kamada-Kawai', value: 'kamada_kawai' },
          ]}
        />
        <Button
          onClick={() => optimizeLayout(optimizationAlgorithm)}
          disabled={loading}
        >
          Optimize Layout
        </Button>
      </div>

      <Divider />

      {/* Statistics */}
      {status?.statistics && (
        <div className="statistics">
          <h4>Statistics</h4>
          <div className="stat">
            <span>Total Steps:</span>
            <span>{status.statistics.total_steps}</span>
          </div>
          <div className="stat">
            <span>Avg Step Time:</span>
            <span>{status.statistics.average_step_time_ms.toFixed(2)}ms</span>
          </div>
          <div className="stat">
            <span>Avg Energy:</span>
            <span>{status.statistics.average_energy.toFixed(4)}</span>
          </div>
          <div className="stat">
            <span>GPU Memory:</span>
            <span>{status.statistics.gpu_memory_used_mb.toFixed(1)}MB</span>
          </div>
        </div>
      )}

      {error && (
        <div className="error-message">
          <span>Error: {error}</span>
        </div>
      )}
    </Card>
  );
};
```

**2. `client/src/features/physics/hooks/usePhysicsService.ts`** (200-250 lines)

```typescript
/**
 * Physics Service Hook
 *
 * Provides access to physics simulation API endpoints
 */

import { useState, useEffect, useCallback } from 'react';
import { unifiedApiClient } from '@/services/api/UnifiedApiClient';

export interface PhysicsStatus {
  simulation_id?: string;
  running: boolean;
  gpu_status?: {
    device_name: string;
    compute_capability: string;
    total_memory_mb: number;
    free_memory_mb: number;
  };
  statistics?: {
    total_steps: number;
    average_step_time_ms: number;
    average_energy: number;
    gpu_memory_used_mb: number;
  };
}

export interface SimulationParameters {
  time_step: number;
  damping: number;
  spring_constant: number;
  repulsion_strength: number;
  attraction_strength: number;
  max_velocity: number;
  convergence_threshold: number;
  max_iterations: number;
  auto_stop_on_convergence: boolean;
}

export function usePhysicsService() {
  const [status, setStatus] = useState<PhysicsStatus | null>(null);
  const [parameters, setParameters] = useState<SimulationParameters>({
    time_step: 0.016,
    damping: 0.8,
    spring_constant: 1.0,
    repulsion_strength: 1.5,
    attraction_strength: 1.0,
    max_velocity: 100.0,
    convergence_threshold: 0.01,
    max_iterations: 1000,
    auto_stop_on_convergence: false,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Poll status every 1 second
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const response = await unifiedApiClient.get('/api/physics/status');
        setStatus(response.data);
      } catch (err) {
        console.error('Failed to fetch physics status:', err);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const startSimulation = useCallback(async (customParams?: Partial<SimulationParameters>) => {
    setLoading(true);
    setError(null);
    try {
      const response = await unifiedApiClient.post('/api/physics/start', {
        ...parameters,
        ...customParams,
      });
      setStatus(response.data);
    } catch (err: any) {
      setError(err.message || 'Failed to start simulation');
      throw err;
    } finally {
      setLoading(false);
    }
  }, [parameters]);

  const stopSimulation = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      await unifiedApiClient.post('/api/physics/stop');
      await fetchStatus(); // Refresh status
    } catch (err: any) {
      setError(err.message || 'Failed to stop simulation');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const updateParameters = useCallback(async (newParams: Partial<SimulationParameters>) => {
    setLoading(true);
    setError(null);
    try {
      await unifiedApiClient.post('/api/physics/parameters', newParams);
      setParameters((prev) => ({ ...prev, ...newParams }));
    } catch (err: any) {
      setError(err.message || 'Failed to update parameters');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const optimizeLayout = useCallback(async (algorithm: string) => {
    setLoading(true);
    setError(null);
    try {
      const response = await unifiedApiClient.post('/api/physics/optimize', {
        algorithm,
        max_iterations: 1000,
        target_energy: 0.01,
      });
      return response.data;
    } catch (err: any) {
      setError(err.message || 'Failed to optimize layout');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const performStep = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      await unifiedApiClient.post('/api/physics/step');
      await fetchStatus();
    } catch (err: any) {
      setError(err.message || 'Failed to perform step');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const resetSimulation = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      await unifiedApiClient.post('/api/physics/reset');
      await fetchStatus();
    } catch (err: any) {
      setError(err.message || 'Failed to reset simulation');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const pinNodes = useCallback(async (nodes: Array<{ node_id: number; x: number; y: number; z: number }>) => {
    setLoading(true);
    setError(null);
    try {
      await unifiedApiClient.post('/api/physics/nodes/pin', { nodes });
    } catch (err: any) {
      setError(err.message || 'Failed to pin nodes');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const unpinNodes = useCallback(async (nodeIds: number[]) => {
    setLoading(true);
    setError(null);
    try {
      await unifiedApiClient.post('/api/physics/nodes/unpin', { node_ids: nodeIds });
    } catch (err: any) {
      setError(err.message || 'Failed to unpin nodes');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const applyForces = useCallback(async (forces: Array<{ node_id: number; force_x: number; force_y: number; force_z: number }>) => {
    setLoading(true);
    setError(null);
    try {
      await unifiedApiClient.post('/api/physics/forces/apply', { forces });
    } catch (err: any) {
      setError(err.message || 'Failed to apply forces');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchStatus = async () => {
    try {
      const response = await unifiedApiClient.get('/api/physics/status');
      setStatus(response.data);
    } catch (err) {
      console.error('Failed to fetch status:', err);
    }
  };

  return {
    status,
    parameters,
    startSimulation,
    stopSimulation,
    updateParameters,
    resetSimulation,
    optimizeLayout,
    performStep,
    pinNodes,
    unpinNodes,
    applyForces,
    loading,
    error,
  };
}
```

**3. Modify `client/src/features/visualisation/components/ControlPanel/index.tsx`**

Add new tab for Physics Control:
```typescript
<Tab label="Physics" icon={<Zap />}>
  <PhysicsControlPanel />
</Tab>
```

#### Testing Plan:

1. **Unit Tests**
   - Test parameter updates
   - Test start/stop simulation
   - Test error handling

2. **Integration Tests**
   - Test API endpoint connectivity
   - Test real-time status updates
   - Test parameter synchronization

3. **Manual Testing**
   - Verify all buttons work
   - Verify sliders update parameters
   - Verify statistics display correctly
   - Verify error messages display

#### Acceptance Criteria:

- [ ] Physics control panel visible in UI
- [ ] Can start/stop simulation from client
- [ ] Parameters update in real-time
- [ ] Statistics display correctly
- [ ] GPU status visible
- [ ] Error handling works
- [ ] All 10 physics endpoints integrated

---

### Task 1.2: Semantic Analysis Panel (Days 6-10)

**Feature:** Expose semantic analysis tools to users

#### Files to Create:

**1. `client/src/features/analytics/components/SemanticAnalysisPanel.tsx`** (350-450 lines)

```typescript
/**
 * Semantic Analysis Panel
 *
 * Provides advanced graph analytics:
 * - Community detection (Louvain, Label Propagation, etc.)
 * - Centrality computation (PageRank, Betweenness, Closeness)
 * - Shortest path finding
 * - Semantic constraint generation
 * - Performance statistics
 */

import React, { useState } from 'react';
import { useSemanticService } from '../hooks/useSemanticService';
import { Button, Select, Input, Card, Divider, Tabs } from '@/components/ui';
import { Network, TrendingUp, Route, Settings } from 'lucide-react';

export const SemanticAnalysisPanel: React.FC = () => {
  const {
    detectCommunities,
    computeCentrality,
    computeShortestPath,
    generateConstraints,
    statistics,
    loading,
    error,
  } = useSemanticService();

  const [activeTab, setActiveTab] = useState('communities');

  return (
    <Card className="semantic-analysis-panel">
      <h3>Semantic Analysis</h3>

      <Tabs activeTab={activeTab} onChange={setActiveTab}>
        <Tab id="communities" label="Communities" icon={<Network />}>
          <CommunityDetectionTab
            onDetect={detectCommunities}
            loading={loading}
          />
        </Tab>

        <Tab id="centrality" label="Centrality" icon={<TrendingUp />}>
          <CentralityAnalysisTab
            onCompute={computeCentrality}
            loading={loading}
          />
        </Tab>

        <Tab id="pathfinding" label="Paths" icon={<Route />}>
          <ShortestPathTab
            onCompute={computeShortestPath}
            loading={loading}
          />
        </Tab>

        <Tab id="constraints" label="Constraints" icon={<Settings />}>
          <ConstraintsTab
            onGenerate={generateConstraints}
            loading={loading}
          />
        </Tab>
      </Tabs>

      {/* Statistics Footer */}
      {statistics && (
        <div className="statistics-footer">
          <Divider />
          <h4>Performance Statistics</h4>
          <div className="stats-grid">
            <div className="stat">
              <span>Total Analyses:</span>
              <span>{statistics.total_analyses}</span>
            </div>
            <div className="stat">
              <span>Avg Clustering Time:</span>
              <span>{statistics.average_clustering_time_ms.toFixed(1)}ms</span>
            </div>
            <div className="stat">
              <span>Avg Pathfinding Time:</span>
              <span>{statistics.average_pathfinding_time_ms.toFixed(1)}ms</span>
            </div>
            <div className="stat">
              <span>Cache Hit Rate:</span>
              <span>{(statistics.cache_hit_rate * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      )}

      {error && <div className="error-message">{error}</div>}
    </Card>
  );
};

const CommunityDetectionTab: React.FC<{ onDetect: Function; loading: boolean }> = ({
  onDetect,
  loading,
}) => {
  const [algorithm, setAlgorithm] = useState('louvain');
  const [minClusterSize, setMinClusterSize] = useState(5);
  const [results, setResults] = useState<any>(null);

  const handleDetect = async () => {
    const result = await onDetect({ algorithm, min_cluster_size: minClusterSize });
    setResults(result);
  };

  return (
    <div className="community-detection">
      <div className="controls">
        <label>Algorithm</label>
        <Select
          value={algorithm}
          onChange={setAlgorithm}
          options={[
            { label: 'Louvain', value: 'louvain' },
            { label: 'Label Propagation', value: 'label_propagation' },
            { label: 'Connected Components', value: 'connected_components' },
            { label: 'Hierarchical', value: 'hierarchical' },
          ]}
        />

        {algorithm === 'hierarchical' && (
          <>
            <label>Min Cluster Size</label>
            <Input
              type="number"
              value={minClusterSize}
              onChange={(e) => setMinClusterSize(parseInt(e.target.value))}
              min={1}
            />
          </>
        )}

        <Button onClick={handleDetect} disabled={loading}>
          Detect Communities
        </Button>
      </div>

      {results && (
        <div className="results">
          <h4>Results</h4>
          <div className="result-stat">
            <span>Total Clusters:</span>
            <span>{Object.keys(results.cluster_sizes).length}</span>
          </div>
          <div className="result-stat">
            <span>Modularity:</span>
            <span>{results.modularity.toFixed(4)}</span>
          </div>
          <div className="result-stat">
            <span>Computation Time:</span>
            <span>{results.computation_time_ms.toFixed(1)}ms</span>
          </div>

          <h5>Cluster Sizes</h5>
          <div className="cluster-list">
            {Object.entries(results.cluster_sizes).map(([clusterId, size]) => (
              <div key={clusterId} className="cluster-item">
                <span>Cluster {clusterId}:</span>
                <span>{size as number} nodes</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// Similar implementations for CentralityAnalysisTab, ShortestPathTab, ConstraintsTab
```

**2. `client/src/features/analytics/hooks/useSemanticService.ts`** (200-250 lines)

Similar pattern to usePhysicsService.

**3. Modify analytics section UI to add new panel**

#### Acceptance Criteria:

- [ ] Semantic analysis panel visible in UI
- [ ] Community detection works for all algorithms
- [ ] Centrality computation displays results
- [ ] Shortest path finder functional
- [ ] Constraint generation works
- [ ] Statistics display correctly
- [ ] All 6 semantic endpoints integrated

---

### Task 1.3: Integration Testing & Bug Fixes (Days 11-15)

**Activities:**
- End-to-end testing of Physics + Semantic panels
- Performance testing (ensure UI doesn't block)
- Error handling improvements
- UI polish and accessibility
- Documentation updates

**Deliverables:**
- [ ] All tests passing
- [ ] No critical bugs
- [ ] User documentation for new features
- [ ] Sprint 1 demo ready

---

## Sprint 2: High Priority Features (Days 16-25)

**Goal:** Add inference tools, health monitoring, and NL query
**Priority:** 🟡 HIGH
**Effort:** 7-10 days

---

### Task 2.1: Inference Tools UI (Days 16-18)

**Feature:** Ontology reasoning and validation

#### Files to Create:

**1. `client/src/features/ontology/components/InferencePanel.tsx`** (250-300 lines)

```typescript
/**
 * Inference Panel
 *
 * Provides ontology inference tools:
 * - Run inference on loaded ontology
 * - Validate ontology consistency
 * - Batch inference for multiple ontologies
 * - View inference results and explanations
 * - Classification and consistency reports
 */

import React, { useState } from 'react';
import { useInferenceService } from '../hooks/useInferenceService';
import { Button, Card, Checkbox, Input, Divider } from '@/components/ui';
import { Brain, CheckCircle, AlertTriangle } from 'lucide-react';

export const InferencePanel: React.FC<{ ontologyId: string }> = ({ ontologyId }) => {
  const {
    runInference,
    validateOntology,
    getResults,
    getClassification,
    getConsistencyReport,
    invalidateCache,
    loading,
    error,
  } = useInferenceService();

  const [force, setForce] = useState(false);
  const [results, setResults] = useState<any>(null);

  const handleRunInference = async () => {
    const result = await runInference({ ontology_id: ontologyId, force });
    setResults(result);
  };

  return (
    <Card className="inference-panel">
      <div className="header">
        <Brain size={24} />
        <h3>Ontology Inference</h3>
      </div>

      <div className="controls">
        <p>Current Ontology: <strong>{ontologyId}</strong></p>

        <Checkbox
          checked={force}
          onChange={setForce}
          label="Force re-inference (ignore cache)"
        />

        <div className="button-group">
          <Button
            onClick={handleRunInference}
            disabled={loading}
            variant="primary"
          >
            Run Inference
          </Button>
          <Button
            onClick={() => validateOntology({ ontology_id: ontologyId })}
            disabled={loading}
          >
            Validate
          </Button>
          <Button
            onClick={() => invalidateCache(ontologyId)}
            disabled={loading}
            variant="secondary"
          >
            Clear Cache
          </Button>
        </div>
      </div>

      {results && (
        <div className="results">
          <Divider />
          <h4>
            {results.success ? (
              <><CheckCircle color="green" /> Inference Complete</>
            ) : (
              <><AlertTriangle color="red" /> Inference Failed</>
            )}
          </h4>

          <div className="result-grid">
            <div className="stat">
              <span>Inferred Axioms:</span>
              <span>{results.inferred_axioms_count}</span>
            </div>
            <div className="stat">
              <span>Inference Time:</span>
              <span>{results.inference_time_ms}ms</span>
            </div>
            <div className="stat">
              <span>Reasoner:</span>
              <span>{results.reasoner_version}</span>
            </div>
          </div>

          {results.error && (
            <div className="error-box">
              <AlertTriangle />
              <span>{results.error}</span>
            </div>
          )}

          <div className="actions">
            <Button onClick={() => getClassification(ontologyId)}>
              View Classification
            </Button>
            <Button onClick={() => getConsistencyReport(ontologyId)}>
              Consistency Report
            </Button>
          </div>
        </div>
      )}

      {error && <div className="error-message">{error}</div>}
    </Card>
  );
};
```

**2. `client/src/features/ontology/hooks/useInferenceService.ts`** (150-200 lines)

**3. Integrate into ontology mode UI**

#### Acceptance Criteria:

- [ ] Inference panel visible in ontology mode
- [ ] Can run inference on current ontology
- [ ] Validation works
- [ ] Results display correctly
- [ ] Error handling works
- [ ] All 7 inference endpoints integrated

---

### Task 2.2: System Health Dashboard (Days 19-21)

**Feature:** Real-time system health monitoring

#### Files to Create:

**1. `client/src/features/monitoring/components/HealthDashboard.tsx`** (200-250 lines)

```typescript
/**
 * System Health Dashboard
 *
 * Monitors system health:
 * - Overall system status
 * - Component health (database, graph, physics, websocket)
 * - MCP relay status and control
 * - Real-time health updates
 */

import React, { useState, useEffect } from 'react';
import { useHealthService } from '../hooks/useHealthService';
import { Card, Button, Badge, Divider } from '@/components/ui';
import { Activity, CheckCircle, AlertCircle, XCircle } from 'lucide-react';

export const HealthDashboard: React.FC = () => {
  const {
    overallHealth,
    physicsHealth,
    startMCPRelay,
    getMCPLogs,
    loading,
    error,
  } = useHealthService();

  const [logs, setLogs] = useState<string>('');

  const getStatusIcon = (healthy: boolean) => {
    return healthy ? (
      <CheckCircle color="green" size={20} />
    ) : (
      <XCircle color="red" size={20} />
    );
  };

  return (
    <Card className="health-dashboard">
      <div className="header">
        <Activity size={24} />
        <h3>System Health Monitor</h3>
      </div>

      {/* Overall Status */}
      <div className="overall-status">
        <h4>Overall Status</h4>
        {overallHealth && (
          <div className="status">
            {getStatusIcon(overallHealth.healthy)}
            <Badge variant={overallHealth.healthy ? 'success' : 'danger'}>
              {overallHealth.healthy ? 'HEALTHY' : 'UNHEALTHY'}
            </Badge>
          </div>
        )}
      </div>

      <Divider />

      {/* Component Health */}
      <div className="components">
        <h4>Components</h4>
        {overallHealth?.components && Object.entries(overallHealth.components).map(([name, healthy]) => (
          <div key={name} className="component">
            {getStatusIcon(healthy as boolean)}
            <span>{name}</span>
          </div>
        ))}
      </div>

      <Divider />

      {/* Physics Simulation */}
      {physicsHealth && (
        <div className="physics-health">
          <h4>Physics Simulation</h4>
          <div className="stat">
            <span>Running:</span>
            <span>{physicsHealth.running ? 'Yes' : 'No'}</span>
          </div>
          {physicsHealth.statistics && (
            <>
              <div className="stat">
                <span>Total Steps:</span>
                <span>{physicsHealth.statistics.total_steps}</span>
              </div>
              <div className="stat">
                <span>Avg Step Time:</span>
                <span>{physicsHealth.statistics.average_step_time_ms.toFixed(1)}ms</span>
              </div>
            </>
          )}
        </div>
      )}

      <Divider />

      {/* MCP Relay */}
      <div className="mcp-relay">
        <h4>MCP Relay</h4>
        <Button onClick={startMCPRelay} disabled={loading}>
          Start Relay
        </Button>
        <Button onClick={async () => setLogs(await getMCPLogs())} disabled={loading}>
          View Logs
        </Button>

        {logs && (
          <pre className="logs">{logs}</pre>
        )}
      </div>

      {error && <div className="error-message">{error}</div>}
    </Card>
  );
};
```

**2. `client/src/features/monitoring/hooks/useHealthService.ts`** (100-150 lines)

**3. Add to main navigation or settings panel**

#### Acceptance Criteria:

- [ ] Health dashboard visible
- [ ] Overall health displays correctly
- [ ] Component statuses accurate
- [ ] MCP relay controls work
- [ ] Logs display correctly
- [ ] All 4 health endpoints integrated

---

### Task 2.3: Natural Language Query Integration (Days 22-23)

**Feature:** Natural language search for graph

#### Files to Modify:

**1. `client/src/components/SearchBar.tsx`** (Add NL query support)

**2. Create `client/src/hooks/useNaturalLanguageQuery.ts`**

#### Acceptance Criteria:

- [ ] Search bar supports NL queries
- [ ] Results display correctly
- [ ] Fast response time (<1s)

---

### Task 2.4: Sprint 2 Testing & Polish (Days 24-25)

**Activities:**
- Integration testing
- Bug fixes
- UI polish
- Documentation

---

## Sprint 3: Architecture & Nice-to-Have Features (Days 26-40)

**Goal:** Refactor client architecture, add monitoring dashboards
**Priority:** 🟢 MEDIUM
**Effort:** 8-15 days

---

### Task 3.1: H4 Message Metrics Dashboard (Days 26-29)

**Prerequisites:** Server must expose H4 metrics endpoint

**Feature:** Real-time message acknowledgment monitoring

#### Files to Create:

**1. Backend: Expose metrics endpoint**

`src/handlers/api_handler/metrics/mod.rs`:
```rust
pub async fn get_message_metrics(
    orchestrator_addr: web::Data<Addr<PhysicsOrchestratorActor>>,
) -> ActixResult<HttpResponse> {
    // Get metrics from orchestrator's MessageTracker
    // Return JSON summary
}
```

**2. `client/src/features/monitoring/components/MessageMetricsDashboard.tsx`**

Real-time metrics with charts showing:
- Success rate over time
- Per-message-kind success rates
- Latency distribution
- Retry counts
- Failure rates

**3. Use Chart.js or Recharts for visualization**

#### Acceptance Criteria:

- [ ] Metrics endpoint created on server
- [ ] Dashboard displays real-time metrics
- [ ] Charts update smoothly
- [ ] Can filter by message kind

---

### Task 3.2: MCP Integration UI (Days 30-36)

**Feature:** MCP server management and monitoring

**Note:** Only implement if MCP is being actively used

#### Files to Create:

**1. `client/src/features/mcp/components/MCPConnectionManager.tsx`**

- List MCP servers
- Add/remove connections
- View connection status
- Real-time message viewer

**2. `client/src/features/mcp/hooks/useMCPWebSocket.ts`**

- WebSocket connection to `/mcp/ws`
- Message handling
- Auto-reconnect

**3. `client/src/features/mcp/components/MCPMessageViewer.tsx`**

- Real-time message stream
- Message filtering
- Message inspector

#### Acceptance Criteria:

- [ ] Can connect to MCP servers
- [ ] Messages display in real-time
- [ ] Can send messages to servers
- [ ] WebSocket reconnects automatically

---

### Task 3.3: Service Layer Refactor (Days 37-39)

**Goal:** Improve client architecture

**Changes:**

1. **Create service layer** (`client/src/services/`)
   - PhysicsService
   - SemanticService
   - InferenceService
   - HealthService
   - etc.

2. **Add feature flags** (`client/src/config/features.ts`)

3. **Implement API version negotiation**

4. **Add circuit breaker pattern** for retry logic

#### Files to Create/Modify:

- 10+ service files
- 5+ configuration files
- Update all components to use services

#### Acceptance Criteria:

- [ ] Service layer abstraction complete
- [ ] Feature flags functional
- [ ] API version checked on startup
- [ ] Circuit breaker prevents cascading failures

---

### Task 3.4: Final Testing & Documentation (Day 40)

**Activities:**
- Full regression testing
- Performance testing
- User documentation updates
- Developer documentation
- Sprint 3 demo

**Deliverables:**
- [ ] All tests passing
- [ ] Documentation complete
- [ ] 100% feature parity achieved
- [ ] Production ready

---

## Testing Strategy

### Unit Tests

For each new component:
- Props validation
- Event handler testing
- Error state testing
- Loading state testing

**Tools:** Jest, React Testing Library

### Integration Tests

For each API integration:
- API endpoint connectivity
- Response parsing
- Error handling
- Retry logic

**Tools:** Jest, MSW (Mock Service Worker)

### E2E Tests

Critical user flows:
1. Start physics simulation → Adjust parameters → Stop simulation
2. Run community detection → View results → Export data
3. Run inference → View results → Download report
4. Check system health → Start MCP relay → View logs

**Tools:** Playwright or Cypress

### Performance Tests

- Initial load time (<3s)
- API response time (<1s for most endpoints)
- UI responsiveness (60fps)
- Memory usage (<200MB)

**Tools:** Lighthouse, Chrome DevTools

---

## Deployment Strategy

### Phase 1: Feature Flags

Deploy all new features behind feature flags:
```typescript
if (FEATURES.PHYSICS_CONTROL) {
  // Show physics panel
}
```

Enable for:
- Internal users: Week 1
- Beta users: Week 2-3
- All users: Week 4

### Phase 2: Gradual Rollout

1. Deploy to staging
2. Internal testing (1 week)
3. Beta release (2 weeks)
4. Production rollout (phased, 25% → 50% → 100%)

### Phase 3: Monitoring

Track metrics:
- New feature usage rates
- Error rates
- Performance metrics
- User feedback

---

## Risk Management

### Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| API changes break client | Medium | High | API version negotiation, feature flags |
| Performance issues with polling | Medium | Medium | Optimize polling intervals, use WebSockets |
| Complex UIs confuse users | High | Medium | User testing, progressive disclosure |
| Integration takes longer than expected | Medium | High | Agile sprints, frequent demos |

---

## Success Metrics

### Technical Metrics

- [ ] Feature parity: 100% (from 79%)
- [ ] Test coverage: >80%
- [ ] API response time: <1s (p95)
- [ ] UI responsiveness: 60fps
- [ ] Zero critical bugs

### User Metrics

- [ ] Feature discovery rate: >50% of users try new features
- [ ] Feature adoption: >30% regular usage
- [ ] User satisfaction: >4.0/5.0
- [ ] Support tickets: <5% increase

---

## Resource Requirements

### Team

- 2 Frontend developers
- 1 Backend developer (for metrics endpoint)
- 1 QA engineer
- 1 Designer (UI review)

### Tools & Infrastructure

- Development environment
- Staging environment
- Feature flag system
- Analytics/monitoring
- Testing tools

---

## Timeline Summary

| Sprint | Duration | Features | Status |
|--------|----------|----------|--------|
| **Sprint 1** | Days 1-15 | Physics + Semantic APIs | 🔴 Critical |
| **Sprint 2** | Days 16-25 | Inference + Health + NL Query | 🟡 High |
| **Sprint 3** | Days 26-40 | Metrics + MCP + Refactor | 🟢 Medium |

**Total Duration:** 40 days (8 weeks with 2 developers)

---

## Conclusion

This comprehensive upgrade plan will bring the VisionFlow client to 100% feature parity with the server, exposing powerful physics simulation, semantic analysis, and ontology reasoning tools to users. The phased approach ensures critical features are delivered first, with progressive enhancement and architectural improvements following in later sprints.

**Key Deliverables:**
- Physics Control Panel
- Semantic Analysis Tools
- Inference UI
- Health Monitoring Dashboard
- Natural Language Query
- Message Metrics Dashboard (optional)
- MCP Integration UI (optional)
- Improved Client Architecture

**Production Readiness:** Will increase from 79% to 100% feature parity.

---

**Plan Status:** ✅ COMPLETE
**Ready for:** Sprint planning and team assignment
**Next Step:** Kick off Sprint 1 - Physics & Semantic APIs
