// frontend/src/components/ControlCenter/QualityGatePanel.tsx
// Quality Gate settings panel for feature toggles and performance thresholds

import React, { useState, useEffect, useCallback } from 'react';
import { settingsApi, QualityGateSettings } from '../../api/settingsApi';
import { useSettingsStore } from '../../store/settingsStore';
import { createLogger } from '../../utils/loggerConfig';

const logger = createLogger('QualityGatePanel');

interface QualityGatePanelProps {
  onError: (message: string) => void;
  onSuccess: (message: string) => void;
}

// Toggles that persist to server but have no client-side consumer yet
const INERT_TOGGLES: Record<string, string> = {
  gpuAcceleration: 'GPU acceleration is always enabled when available',
  ontologyPhysics: 'Managed by physics engine automatically',
  semanticForces: 'Semantic forces not yet wired to renderer',
  layoutMode: 'Layout mode not yet consumed by physics worker',
  gnnPhysics: 'GNN physics model not yet available',
  showClusters: 'Analytics overlay not yet implemented',
  showAnomalies: 'Analytics overlay not yet implemented',
  showCommunities: 'Analytics overlay not yet implemented',
  ruvectorEnabled: 'Server-side HNSW indexing - no client toggle needed',
  autoAdjust: 'Performance auto-tuning not yet implemented',
  minFpsThreshold: 'Performance auto-tuning not yet implemented',
};

const defaultSettings: QualityGateSettings = {
  gpuAcceleration: true,
  ontologyPhysics: false,
  semanticForces: false,
  layoutMode: 'force-directed',
  showClusters: true,
  showAnomalies: true,
  showCommunities: false,
  ruvectorEnabled: false,
  gnnPhysics: false,
  minFpsThreshold: 30,
  maxNodeCount: 500000,  // High default - show all nodes by default
  autoAdjust: true,
};

export const QualityGatePanel: React.FC<QualityGatePanelProps> = ({
  onError,
  onSuccess,
}) => {
  const [settings, setSettings] = useState<QualityGateSettings>(defaultSettings);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  // Load settings on mount
  useEffect(() => {
    const loadSettings = async () => {
      try {
        const response = await settingsApi.getQualityGates();
        setSettings(response.data);
        // Sync to Zustand store for graphDataManager
        useSettingsStore.getState().updateSettings((draft: any) => {
          draft.qualityGates = response.data;
        });
      } catch (error) {
        logger.error('Failed to load quality gate settings:', error);
        // Use defaults on error
        setSettings(defaultSettings);
        useSettingsStore.getState().updateSettings((draft: any) => {
          draft.qualityGates = defaultSettings;
        });
      } finally {
        setLoading(false);
      }
    };
    loadSettings();
  }, []);

  // Save settings with debounce
  const saveSettings = useCallback(async (newSettings: QualityGateSettings) => {
    setSaving(true);
    try {
      await settingsApi.updateQualityGates(newSettings);
      // Sync to Zustand store for graphDataManager filtering
      useSettingsStore.getState().updateSettings((draft: any) => {
        draft.qualityGates = newSettings;
      });
      logger.debug('Updated qualityGates in store:', newSettings);
      onSuccess('Quality gate settings updated');
    } catch (error) {
      logger.error('Failed to save quality gate settings:', error);
      onError('Failed to save quality gate settings');
    } finally {
      setSaving(false);
    }
  }, [onError, onSuccess]);

  const handleToggle = (key: keyof QualityGateSettings) => {
    const newSettings = { ...settings, [key]: !settings[key] };
    setSettings(newSettings);
    saveSettings(newSettings);
  };

  const handleSliderChange = (key: keyof QualityGateSettings, value: number) => {
    const newSettings = { ...settings, [key]: value };
    setSettings(newSettings);
  };

  const handleSliderCommit = () => {
    saveSettings(settings);
  };

  const handleSelectChange = (key: keyof QualityGateSettings, value: string) => {
    const newSettings = { ...settings, [key]: value };
    setSettings(newSettings);
    saveSettings(newSettings);
  };

  if (loading) {
    return <div className="panel-loading">Loading quality gate settings...</div>;
  }

  return (
    <div className="quality-gate-panel">
      {/* Compute Mode Section */}
      <section className="settings-section">
        <h3>Compute Mode</h3>
        <div className="setting-row" style={{ opacity: 0.45, cursor: 'not-allowed' }} title={INERT_TOGGLES.gpuAcceleration}>
          <label>
            <span className="setting-label">GPU Acceleration</span>
            <span className="setting-description">Enable GPU-accelerated physics (20-50x faster)</span>
            <em style={{ fontSize: '0.75em', display: 'block', color: '#888' }}>{INERT_TOGGLES.gpuAcceleration}</em>
          </label>
          <input type="checkbox" checked={settings.gpuAcceleration} disabled />
        </div>
        <div className="setting-row" style={{ opacity: 0.45, cursor: 'not-allowed' }} title={INERT_TOGGLES.autoAdjust}>
          <label>
            <span className="setting-label">Auto-Adjust Quality</span>
            <span className="setting-description">Automatically disable expensive features if FPS drops</span>
            <em style={{ fontSize: '0.75em', display: 'block', color: '#888' }}>{INERT_TOGGLES.autoAdjust}</em>
          </label>
          <input type="checkbox" checked={settings.autoAdjust} disabled />
        </div>
        <div className="setting-row" style={{ opacity: 0.45, cursor: 'not-allowed' }} title={INERT_TOGGLES.minFpsThreshold}>
          <label>
            <span className="setting-label">Min FPS Threshold</span>
            <span className="setting-description">Disable features if FPS falls below: {settings.minFpsThreshold} fps</span>
            <em style={{ fontSize: '0.75em', display: 'block', color: '#888' }}>{INERT_TOGGLES.minFpsThreshold}</em>
          </label>
          <input type="range" min={15} max={60} step={5} value={settings.minFpsThreshold} disabled />
        </div>
        <div className="setting-row">
          <label>
            <span className="setting-label">Max Node Count</span>
            <span className="setting-description">Aggressive filtering above: {settings.maxNodeCount.toLocaleString()} nodes</span>
          </label>
          <input
            type="range"
            min={100}
            max={500000}
            step={5000}
            value={settings.maxNodeCount}
            onChange={(e) => handleSliderChange('maxNodeCount', parseInt(e.target.value))}
            onMouseUp={handleSliderCommit}
            onTouchEnd={handleSliderCommit}
            disabled={saving}
          />
        </div>
      </section>

      {/* Physics Features Section */}
      <section className="settings-section">
        <h3>Physics Features</h3>
        <div className="setting-row" style={{ opacity: 0.45, cursor: 'not-allowed' }} title={INERT_TOGGLES.ontologyPhysics}>
          <label>
            <span className="setting-label">Ontology Physics Forces</span>
            <span className="setting-description">Apply SubClassOf, DisjointWith, EquivalentClasses constraints</span>
            <em style={{ fontSize: '0.75em', display: 'block', color: '#888' }}>{INERT_TOGGLES.ontologyPhysics}</em>
          </label>
          <input type="checkbox" checked={settings.ontologyPhysics} disabled />
        </div>
        <div className="setting-row" style={{ opacity: 0.45, cursor: 'not-allowed' }} title={INERT_TOGGLES.semanticForces}>
          <label>
            <span className="setting-label">Semantic Forces</span>
            <span className="setting-description">Enable DAG layout and type clustering (experimental)</span>
            <em style={{ fontSize: '0.75em', display: 'block', color: '#888' }}>{INERT_TOGGLES.semanticForces}</em>
          </label>
          <input type="checkbox" checked={settings.semanticForces} disabled />
        </div>
        <div className="setting-row" style={{ opacity: 0.45, cursor: 'not-allowed' }} title={INERT_TOGGLES.layoutMode}>
          <label>
            <span className="setting-label">Layout Mode</span>
            <span className="setting-description">Graph layout algorithm</span>
            <em style={{ fontSize: '0.75em', display: 'block', color: '#888' }}>{INERT_TOGGLES.layoutMode}</em>
          </label>
          <select value={settings.layoutMode} disabled>
            <option value="force-directed">Force-Directed (Classic)</option>
            <option value="dag-topdown">DAG Top-Down</option>
            <option value="dag-radial">DAG Radial</option>
            <option value="dag-leftright">DAG Left-Right</option>
            <option value="type-clustering">Type Clustering</option>
          </select>
        </div>
        <div className="setting-row" style={{ opacity: 0.45, cursor: 'not-allowed' }} title={INERT_TOGGLES.gnnPhysics}>
          <label>
            <span className="setting-label">GNN-Enhanced Physics</span>
            <span className="setting-description">Use Graph Neural Network learned weights (advanced)</span>
            <em style={{ fontSize: '0.75em', display: 'block', color: '#888' }}>{INERT_TOGGLES.gnnPhysics}</em>
          </label>
          <input type="checkbox" checked={settings.gnnPhysics} disabled />
        </div>
      </section>

      {/* Analytics Visualization Section */}
      <section className="settings-section">
        <h3>Analytics Visualization</h3>
        <div className="setting-row" style={{ opacity: 0.45, cursor: 'not-allowed' }} title={INERT_TOGGLES.showClusters}>
          <label>
            <span className="setting-label">Show Clusters</span>
            <span className="setting-description">Color-code nodes by cluster assignment</span>
            <em style={{ fontSize: '0.75em', display: 'block', color: '#888' }}>{INERT_TOGGLES.showClusters}</em>
          </label>
          <input type="checkbox" checked={settings.showClusters} disabled />
        </div>
        <div className="setting-row" style={{ opacity: 0.45, cursor: 'not-allowed' }} title={INERT_TOGGLES.showAnomalies}>
          <label>
            <span className="setting-label">Show Anomalies</span>
            <span className="setting-description">Highlight anomalous nodes with red glow</span>
            <em style={{ fontSize: '0.75em', display: 'block', color: '#888' }}>{INERT_TOGGLES.showAnomalies}</em>
          </label>
          <input type="checkbox" checked={settings.showAnomalies} disabled />
        </div>
        <div className="setting-row" style={{ opacity: 0.45, cursor: 'not-allowed' }} title={INERT_TOGGLES.showCommunities}>
          <label>
            <span className="setting-label">Show Communities</span>
            <span className="setting-description">Visualize Louvain community detection results</span>
            <em style={{ fontSize: '0.75em', display: 'block', color: '#888' }}>{INERT_TOGGLES.showCommunities}</em>
          </label>
          <input type="checkbox" checked={settings.showCommunities} disabled />
        </div>
      </section>

      {/* Advanced Features Section */}
      <section className="settings-section">
        <h3>Advanced Features</h3>
        <div className="setting-row" style={{ opacity: 0.45, cursor: 'not-allowed' }} title={INERT_TOGGLES.ruvectorEnabled}>
          <label>
            <span className="setting-label">RuVector Integration</span>
            <span className="setting-description">Enable HNSW index for 150x faster similarity search</span>
            <em style={{ fontSize: '0.75em', display: 'block', color: '#888' }}>{INERT_TOGGLES.ruvectorEnabled}</em>
          </label>
          <input type="checkbox" checked={settings.ruvectorEnabled} disabled />
        </div>
      </section>

      {saving && <div className="saving-indicator">Saving...</div>}
    </div>
  );
};
