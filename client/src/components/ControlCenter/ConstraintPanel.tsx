// frontend/src/components/ControlCenter/ConstraintPanel.tsx
// REAL constraint management panel - NO MOCKS

import React, { useState, useEffect } from 'react';
import {
  constraintsApi,
  ConstraintListResponse,
  ApplyConstraintRequest,
  createConstraintSystem,
  clampStrength,
  getConstraintTypeName,
} from '../../api/constraintsApi';
import './ConstraintPanel.css';

interface ConstraintPanelProps {
  onError?: (error: string) => void;
  onSuccess?: (message: string) => void;
}

export const ConstraintPanel: React.FC<ConstraintPanelProps> = ({
  onError,
  onSuccess,
}) => {
  const [constraintData, setConstraintData] =
    useState<ConstraintListResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [applying, setApplying] = useState(false);

  
  const [constraintType, setConstraintType] = useState<
    'separation' | 'boundary' | 'alignment' | 'cluster'
  >('separation');
  const [nodeIds, setNodeIds] = useState<string>('');
  const [strength, setStrength] = useState<number>(0.5);

  useEffect(() => {
    loadConstraints();
  }, []);

  const loadConstraints = async () => {
    setLoading(true);
    try {
      const response = await constraintsApi.list();
      setConstraintData(response.data);
      if (onSuccess) {
        onSuccess('Constraints loaded successfully');
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to load constraints';
      console.error('Failed to load constraints:', err);
      if (onError) {
        onError(message);
      }
    } finally {
      setLoading(false);
    }
  };

  const applyConstraints = async () => {
    
    const ids = nodeIds
      .split(',')
      .map((id) => parseInt(id.trim(), 10))
      .filter((id) => !isNaN(id));

    if (ids.length === 0) {
      if (onError) {
        onError('Please enter valid node IDs (comma-separated numbers)');
      }
      return;
    }

    const request: ApplyConstraintRequest = {
      constraintType,
      nodeIds: ids,
      strength: clampStrength(strength),
    };

    setApplying(true);
    try {
      const response = await constraintsApi.apply(request);
      if (onSuccess) {
        onSuccess(
          `Applied ${response.data.constraintType} constraint to ${response.data.nodeCount} nodes`
        );
      }
      
      await loadConstraints();
      
      setNodeIds('');
      setStrength(0.5);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to apply constraints';
      console.error('Failed to apply constraints:', err);
      if (onError) {
        onError(message);
      }
    } finally {
      setApplying(false);
    }
  };

  const defineDefaultSystem = async () => {
    setApplying(true);
    try {
      const system = createConstraintSystem();
      const response = await constraintsApi.define(system);
      if (onSuccess) {
        onSuccess('Default constraint system defined successfully');
      }
      await loadConstraints();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to define constraint system';
      console.error('Failed to define constraints:', err);
      if (onError) {
        onError(message);
      }
    } finally {
      setApplying(false);
    }
  };

  const removeAllConstraints = async () => {
    setApplying(true);
    try {
      const response = await constraintsApi.remove({});
      if (onSuccess) {
        onSuccess(`Removed ${response.data.removedCount} constraints`);
      }
      await loadConstraints();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to remove constraints';
      console.error('Failed to remove constraints:', err);
      if (onError) {
        onError(message);
      }
    } finally {
      setApplying(false);
    }
  };

  if (loading) {
    return (
      <div className="constraint-panel loading">Loading constraints...</div>
    );
  }

  return (
    <div className="constraint-panel">
      <div className="constraint-header">
        <h2>Constraint Management</h2>
        <div className="header-actions">
          <button onClick={loadConstraints} disabled={loading || applying}>
            Refresh
          </button>
          <button onClick={defineDefaultSystem} disabled={applying}>
            Define Default System
          </button>
          <button
            onClick={removeAllConstraints}
            disabled={applying}
            className="danger"
          >
            Remove All
          </button>
        </div>
      </div>

      {constraintData && (
        <div className="constraint-status">
          <div className="status-item">
            <strong>Data Source:</strong> {constraintData.dataSource}
          </div>
          <div className="status-item">
            <strong>Total Constraints:</strong> {constraintData.count}
          </div>
          <div className="status-item">
            <strong>GPU Available:</strong>{' '}
            {constraintData.gpuAvailable ? 'Yes' : 'No'}
          </div>
          {constraintData.modes && (
            <>
              <div className="status-item">
                <strong>Logseq Mode:</strong>{' '}
                {constraintData.modes.logseqComputeMode === 2
                  ? 'Constraints Enabled'
                  : 'Standard'}
              </div>
              <div className="status-item">
                <strong>VisionFlow Mode:</strong>{' '}
                {constraintData.modes.visionflowComputeMode === 2
                  ? 'Constraints Enabled'
                  : 'Standard'}
              </div>
            </>
          )}
        </div>
      )}

      <div className="constraint-list">
        <h3>Active Constraints ({constraintData?.count || 0})</h3>
        {constraintData && constraintData.constraints.length > 0 ? (
          <div className="constraints-grid">
            {constraintData.constraints.map((constraint, index) => (
              <div key={index} className="constraint-card">
                <div className="constraint-type">{constraint.type}</div>
                {constraint.enabled !== undefined && (
                  <div
                    className={`constraint-status ${constraint.enabled ? 'enabled' : 'disabled'}`}
                  >
                    {constraint.enabled ? 'Enabled' : 'Disabled'}
                  </div>
                )}
                {constraint.mode && (
                  <div className="constraint-mode">{constraint.mode}</div>
                )}
                {constraint.targetGraphs && (
                  <div className="constraint-targets">
                    Targets: {constraint.targetGraphs.join(', ')}
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-state">
            No constraints currently defined. Use the form below to apply
            constraints or click "Define Default System" to set up standard
            constraints.
          </div>
        )}
      </div>

      <div className="apply-constraint-form">
        <h3>Apply Constraint to Nodes</h3>

        <div className="form-group">
          <label>
            Constraint Type:
            <select
              value={constraintType}
              onChange={(e) => setConstraintType(e.target.value as any)}
              disabled={applying}
            >
              <option value="separation">Separation</option>
              <option value="boundary">Boundary</option>
              <option value="alignment">Alignment</option>
              <option value="cluster">Cluster</option>
            </select>
          </label>
        </div>

        <div className="form-group">
          <label>
            Node IDs (comma-separated):
            <input
              type="text"
              value={nodeIds}
              onChange={(e) => setNodeIds(e.target.value)}
              placeholder="e.g., 1, 2, 3, 4"
              disabled={applying}
            />
          </label>
          <small>Enter the IDs of nodes to apply this constraint to</small>
        </div>

        <div className="form-group">
          <label>
            Strength: {strength.toFixed(2)}
            <input
              type="range"
              min="0"
              max="10"
              step="0.1"
              value={strength}
              onChange={(e) => setStrength(parseFloat(e.target.value))}
              disabled={applying}
            />
          </label>
          <small>Constraint strength (0.0 to 10.0)</small>
        </div>

        <button
          onClick={applyConstraints}
          disabled={applying || !nodeIds.trim()}
          className="apply-button"
        >
          {applying ? 'Applying...' : 'Apply Constraint'}
        </button>
      </div>

      <div className="constraint-help">
        <h4>Constraint Types:</h4>
        <ul>
          <li>
            <strong>Separation:</strong> Maintain minimum distance between nodes
          </li>
          <li>
            <strong>Boundary:</strong> Keep nodes within defined boundaries
          </li>
          <li>
            <strong>Alignment:</strong> Align nodes along axes
          </li>
          <li>
            <strong>Cluster:</strong> Group nodes together
          </li>
        </ul>
      </div>
    </div>
  );
};
