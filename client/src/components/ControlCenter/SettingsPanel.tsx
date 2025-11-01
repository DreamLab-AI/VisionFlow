// frontend/src/components/ControlCenter/SettingsPanel.tsx
// REAL settings panel with database integration - NO MOCKS

import React, { useState, useEffect } from 'react';
import {
  settingsApi,
  PhysicsSettings,
  ConstraintSettings,
  RenderingSettings,
  clamp,
  validatePhysicsSettings,
  validateConstraintSettings,
} from '../../api/settingsApi';
import './SettingsPanel.css';

interface SettingsPanelProps {
  onError?: (error: string) => void;
  onSuccess?: (message: string) => void;
}

export const SettingsPanel: React.FC<SettingsPanelProps> = ({
  onError,
  onSuccess,
}) => {
  const [physics, setPhysics] = useState<PhysicsSettings | null>(null);
  const [constraints, setConstraints] = useState<ConstraintSettings | null>(
    null
  );
  const [rendering, setRendering] = useState<RenderingSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [activeTab, setActiveTab] = useState<
    'physics' | 'constraints' | 'rendering'
  >('physics');

  
  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    setLoading(true);
    try {
      const [physicsRes, constraintsRes, renderingRes] = await Promise.all([
        settingsApi.getPhysics(),
        settingsApi.getConstraints(),
        settingsApi.getRendering(),
      ]);

      setPhysics(physicsRes.data);
      setConstraints(constraintsRes.data);
      setRendering(renderingRes.data);

      if (onSuccess) {
        onSuccess('Settings loaded successfully');
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load settings';
      console.error('Failed to load settings:', err);
      if (onError) {
        onError(message);
      }
    } finally {
      setLoading(false);
    }
  };

  const updatePhysics = async (updates: Partial<PhysicsSettings>) => {
    if (!physics) return;

    const validation = validatePhysicsSettings(updates);
    if (validation) {
      if (onError) onError(validation);
      return;
    }

    setSaving(true);
    try {
      await settingsApi.updatePhysics(updates);
      setPhysics({ ...physics, ...updates });
      if (onSuccess) {
        onSuccess('Physics settings updated');
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to update physics';
      console.error('Failed to update physics:', err);
      if (onError) {
        onError(message);
      }
    } finally {
      setSaving(false);
    }
  };

  const updateConstraints = async (updates: Partial<ConstraintSettings>) => {
    if (!constraints) return;

    const validation = validateConstraintSettings(updates);
    if (validation) {
      if (onError) onError(validation);
      return;
    }

    setSaving(true);
    try {
      await settingsApi.updateConstraints(updates);
      setConstraints({ ...constraints, ...updates });
      if (onSuccess) {
        onSuccess('Constraint settings updated');
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to update constraints';
      console.error('Failed to update constraints:', err);
      if (onError) {
        onError(message);
      }
    } finally {
      setSaving(false);
    }
  };

  const updateRendering = async (updates: Partial<RenderingSettings>) => {
    if (!rendering) return;

    setSaving(true);
    try {
      await settingsApi.updateRendering(updates);
      setRendering({ ...rendering, ...updates });
      if (onSuccess) {
        onSuccess('Rendering settings updated');
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to update rendering';
      console.error('Failed to update rendering:', err);
      if (onError) {
        onError(message);
      }
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return <div className="settings-panel loading">Loading settings...</div>;
  }

  if (!physics || !constraints || !rendering) {
    return (
      <div className="settings-panel error">
        Failed to load settings. Please refresh.
      </div>
    );
  }

  return (
    <div className="settings-panel">
      <div className="settings-header">
        <h2>Settings Control Panel</h2>
        <button onClick={loadSettings} disabled={loading || saving}>
          Refresh
        </button>
      </div>

      <div className="settings-tabs">
        <button
          className={activeTab === 'physics' ? 'active' : ''}
          onClick={() => setActiveTab('physics')}
        >
          Physics
        </button>
        <button
          className={activeTab === 'constraints' ? 'active' : ''}
          onClick={() => setActiveTab('constraints')}
        >
          Constraints
        </button>
        <button
          className={activeTab === 'rendering' ? 'active' : ''}
          onClick={() => setActiveTab('rendering')}
        >
          Rendering
        </button>
      </div>

      {activeTab === 'physics' && (
        <div className="settings-section">
          <h3>Physics Settings</h3>

          <div className="setting-group">
            <label>
              <input
                type="checkbox"
                checked={physics.enabled}
                onChange={(e) => updatePhysics({ enabled: e.target.checked })}
                disabled={saving}
              />
              Enable Physics Simulation
            </label>
          </div>

          <div className="setting-group">
            <label>
              Damping: {physics.damping.toFixed(2)}
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={physics.damping}
                onChange={(e) =>
                  updatePhysics({ damping: parseFloat(e.target.value) })
                }
                disabled={saving}
              />
            </label>
          </div>

          <div className="setting-group">
            <label>
              Max Velocity: {physics.maxVelocity.toFixed(1)}
              <input
                type="range"
                min="0"
                max="200"
                step="5"
                value={physics.maxVelocity}
                onChange={(e) =>
                  updatePhysics({ maxVelocity: parseFloat(e.target.value) })
                }
                disabled={saving}
              />
            </label>
          </div>

          <div className="setting-group">
            <label>
              Separation Radius: {physics.separationRadius.toFixed(1)}
              <input
                type="range"
                min="0"
                max="500"
                step="10"
                value={physics.separationRadius}
                onChange={(e) =>
                  updatePhysics({
                    separationRadius: parseFloat(e.target.value),
                  })
                }
                disabled={saving}
              />
            </label>
          </div>

          <div className="setting-group">
            <label>
              Iterations: {physics.iterations}
              <input
                type="number"
                min="1"
                max="100"
                value={physics.iterations}
                onChange={(e) =>
                  updatePhysics({ iterations: parseInt(e.target.value, 10) })
                }
                disabled={saving}
              />
            </label>
          </div>

          <div className="setting-group">
            <label>
              <input
                type="checkbox"
                checked={physics.enableBounds}
                onChange={(e) =>
                  updatePhysics({ enableBounds: e.target.checked })
                }
                disabled={saving}
              />
              Enable Boundary Constraints
            </label>
          </div>

          {physics.enableBounds && (
            <div className="setting-group">
              <label>
                Bounds Size: {physics.boundsSize.toFixed(0)}
                <input
                  type="range"
                  min="100"
                  max="2000"
                  step="50"
                  value={physics.boundsSize}
                  onChange={(e) =>
                    updatePhysics({ boundsSize: parseFloat(e.target.value) })
                  }
                  disabled={saving}
                />
              </label>
            </div>
          )}
        </div>
      )}

      {activeTab === 'constraints' && (
        <div className="settings-section">
          <h3>Constraint LOD Settings</h3>

          <div className="setting-group">
            <label>
              <input
                type="checkbox"
                checked={constraints.lodEnabled}
                onChange={(e) =>
                  updateConstraints({ lodEnabled: e.target.checked })
                }
                disabled={saving}
              />
              Enable Level-of-Detail Constraint Culling
            </label>
          </div>

          {constraints.lodEnabled && (
            <>
              <div className="setting-group">
                <label>
                  Far Threshold: {constraints.farThreshold.toFixed(0)}
                  <input
                    type="range"
                    min="100"
                    max="5000"
                    step="100"
                    value={constraints.farThreshold}
                    onChange={(e) =>
                      updateConstraints({
                        farThreshold: parseFloat(e.target.value),
                      })
                    }
                    disabled={saving}
                  />
                </label>
                <small>Distance for priority 1-3 constraints only</small>
              </div>

              <div className="setting-group">
                <label>
                  Medium Threshold: {constraints.mediumThreshold.toFixed(0)}
                  <input
                    type="range"
                    min="10"
                    max="1000"
                    step="10"
                    value={constraints.mediumThreshold}
                    onChange={(e) =>
                      updateConstraints({
                        mediumThreshold: parseFloat(e.target.value),
                      })
                    }
                    disabled={saving}
                  />
                </label>
                <small>Distance for priority 1-5 constraints</small>
              </div>

              <div className="setting-group">
                <label>
                  Near Threshold: {constraints.nearThreshold.toFixed(0)}
                  <input
                    type="range"
                    min="1"
                    max="100"
                    step="1"
                    value={constraints.nearThreshold}
                    onChange={(e) =>
                      updateConstraints({
                        nearThreshold: parseFloat(e.target.value),
                      })
                    }
                    disabled={saving}
                  />
                </label>
                <small>Distance for all constraints</small>
              </div>

              <div className="setting-group">
                <label>
                  Priority Weighting:
                  <select
                    value={constraints.priorityWeighting}
                    onChange={(e) =>
                      updateConstraints({
                        priorityWeighting: e.target.value as any,
                      })
                    }
                    disabled={saving}
                  >
                    <option value="linear">Linear</option>
                    <option value="exponential">Exponential</option>
                    <option value="quadratic">Quadratic</option>
                  </select>
                </label>
              </div>
            </>
          )}

          <div className="setting-group">
            <label>
              <input
                type="checkbox"
                checked={constraints.progressiveActivation}
                onChange={(e) =>
                  updateConstraints({
                    progressiveActivation: e.target.checked,
                  })
                }
                disabled={saving}
              />
              Progressive Constraint Activation
            </label>
          </div>

          {constraints.progressiveActivation && (
            <div className="setting-group">
              <label>
                Activation Frames: {constraints.activationFrames}
                <input
                  type="number"
                  min="1"
                  max="600"
                  value={constraints.activationFrames}
                  onChange={(e) =>
                    updateConstraints({
                      activationFrames: parseInt(e.target.value, 10),
                    })
                  }
                  disabled={saving}
                />
              </label>
              <small>Number of frames to fully activate constraints</small>
            </div>
          )}
        </div>
      )}

      {activeTab === 'rendering' && (
        <div className="settings-section">
          <h3>Rendering Settings</h3>

          <div className="setting-group">
            <label>
              Ambient Light Intensity:{' '}
              {rendering.ambientLightIntensity.toFixed(2)}
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={rendering.ambientLightIntensity}
                onChange={(e) =>
                  updateRendering({
                    ambientLightIntensity: parseFloat(e.target.value),
                  })
                }
                disabled={saving}
              />
            </label>
          </div>

          <div className="setting-group">
            <label>
              Directional Light Intensity:{' '}
              {rendering.directionalLightIntensity.toFixed(2)}
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={rendering.directionalLightIntensity}
                onChange={(e) =>
                  updateRendering({
                    directionalLightIntensity: parseFloat(e.target.value),
                  })
                }
                disabled={saving}
              />
            </label>
          </div>

          <div className="setting-group">
            <label>
              <input
                type="checkbox"
                checked={rendering.enableAntialiasing}
                onChange={(e) =>
                  updateRendering({ enableAntialiasing: e.target.checked })
                }
                disabled={saving}
              />
              Enable Antialiasing
            </label>
          </div>

          <div className="setting-group">
            <label>
              <input
                type="checkbox"
                checked={rendering.enableShadows}
                onChange={(e) =>
                  updateRendering({ enableShadows: e.target.checked })
                }
                disabled={saving}
              />
              Enable Shadows
            </label>
          </div>

          <div className="setting-group">
            <label>
              <input
                type="checkbox"
                checked={rendering.enableAmbientOcclusion}
                onChange={(e) =>
                  updateRendering({
                    enableAmbientOcclusion: e.target.checked,
                  })
                }
                disabled={saving}
              />
              Enable Ambient Occlusion
            </label>
          </div>

          <div className="setting-group">
            <label>
              Background Color:
              <input
                type="color"
                value={rendering.backgroundColor}
                onChange={(e) =>
                  updateRendering({ backgroundColor: e.target.value })
                }
                disabled={saving}
              />
            </label>
          </div>
        </div>
      )}

      {saving && <div className="saving-indicator">Saving...</div>}
    </div>
  );
};
