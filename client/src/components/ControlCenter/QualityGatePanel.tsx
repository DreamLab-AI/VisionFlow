// frontend/src/components/ControlCenter/QualityGatePanel.tsx
// Quality Gate settings panel for feature toggles and performance thresholds

import React, { useState, useEffect, useCallback } from 'react';
import { settingsApi, QualityGateSettings } from '../../api/settingsApi';
import { useSettingsStore } from '../../store/settingsStore';

interface QualityGatePanelProps {
  onError: (message: string) => void;
  onSuccess: (message: string) => void;
}

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
  maxNodeCount: 500,  // Reduced for better debugging velocity
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
        console.error('Failed to load quality gate settings:', error);
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
      console.log('[QualityGatePanel] Updated qualityGates in store:', newSettings);
      onSuccess('Quality gate settings updated');
    } catch (error) {
      console.error('Failed to save quality gate settings:', error);
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
        <div className="setting-row">
          <label>
            <span className="setting-label">GPU Acceleration</span>
            <span className="setting-description">Enable GPU-accelerated physics (20-50x faster)</span>
          </label>
          <input
            type="checkbox"
            checked={settings.gpuAcceleration}
            onChange={() => handleToggle('gpuAcceleration')}
            disabled={saving}
          />
        </div>
        <div className="setting-row">
          <label>
            <span className="setting-label">Auto-Adjust Quality</span>
            <span className="setting-description">Automatically disable expensive features if FPS drops</span>
          </label>
          <input
            type="checkbox"
            checked={settings.autoAdjust}
            onChange={() => handleToggle('autoAdjust')}
            disabled={saving}
          />
        </div>
        <div className="setting-row">
          <label>
            <span className="setting-label">Min FPS Threshold</span>
            <span className="setting-description">Disable features if FPS falls below: {settings.minFpsThreshold} fps</span>
          </label>
          <input
            type="range"
            min={15}
            max={60}
            step={5}
            value={settings.minFpsThreshold}
            onChange={(e) => handleSliderChange('minFpsThreshold', parseInt(e.target.value))}
            onMouseUp={handleSliderCommit}
            onTouchEnd={handleSliderCommit}
            disabled={saving}
          />
        </div>
        <div className="setting-row">
          <label>
            <span className="setting-label">Max Node Count</span>
            <span className="setting-description">Aggressive filtering above: {settings.maxNodeCount.toLocaleString()} nodes</span>
          </label>
          <input
            type="range"
            min={100}
            max={10000}
            step={100}
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
        <div className="setting-row">
          <label>
            <span className="setting-label">Ontology Physics Forces</span>
            <span className="setting-description">Apply SubClassOf, DisjointWith, EquivalentClasses constraints</span>
          </label>
          <input
            type="checkbox"
            checked={settings.ontologyPhysics}
            onChange={() => handleToggle('ontologyPhysics')}
            disabled={saving}
          />
        </div>
        <div className="setting-row">
          <label>
            <span className="setting-label">Semantic Forces</span>
            <span className="setting-description">Enable DAG layout and type clustering (experimental)</span>
          </label>
          <input
            type="checkbox"
            checked={settings.semanticForces}
            onChange={() => handleToggle('semanticForces')}
            disabled={saving}
          />
        </div>
        <div className="setting-row">
          <label>
            <span className="setting-label">Layout Mode</span>
            <span className="setting-description">Graph layout algorithm</span>
          </label>
          <select
            value={settings.layoutMode}
            onChange={(e) => handleSelectChange('layoutMode', e.target.value)}
            disabled={saving}
          >
            <option value="force-directed">Force-Directed (Classic)</option>
            <option value="dag-topdown">DAG Top-Down</option>
            <option value="dag-radial">DAG Radial</option>
            <option value="dag-leftright">DAG Left-Right</option>
            <option value="type-clustering">Type Clustering</option>
          </select>
        </div>
        <div className="setting-row">
          <label>
            <span className="setting-label">GNN-Enhanced Physics</span>
            <span className="setting-description">Use Graph Neural Network learned weights (advanced)</span>
          </label>
          <input
            type="checkbox"
            checked={settings.gnnPhysics}
            onChange={() => handleToggle('gnnPhysics')}
            disabled={saving}
          />
        </div>
      </section>

      {/* Analytics Visualization Section */}
      <section className="settings-section">
        <h3>Analytics Visualization</h3>
        <div className="setting-row">
          <label>
            <span className="setting-label">Show Clusters</span>
            <span className="setting-description">Color-code nodes by cluster assignment</span>
          </label>
          <input
            type="checkbox"
            checked={settings.showClusters}
            onChange={() => handleToggle('showClusters')}
            disabled={saving}
          />
        </div>
        <div className="setting-row">
          <label>
            <span className="setting-label">Show Anomalies</span>
            <span className="setting-description">Highlight anomalous nodes with red glow</span>
          </label>
          <input
            type="checkbox"
            checked={settings.showAnomalies}
            onChange={() => handleToggle('showAnomalies')}
            disabled={saving}
          />
        </div>
        <div className="setting-row">
          <label>
            <span className="setting-label">Show Communities</span>
            <span className="setting-description">Visualize Louvain community detection results</span>
          </label>
          <input
            type="checkbox"
            checked={settings.showCommunities}
            onChange={() => handleToggle('showCommunities')}
            disabled={saving}
          />
        </div>
      </section>

      {/* Advanced Features Section */}
      <section className="settings-section">
        <h3>Advanced Features</h3>
        <div className="setting-row">
          <label>
            <span className="setting-label">RuVector Integration</span>
            <span className="setting-description">Enable HNSW index for 150x faster similarity search</span>
          </label>
          <input
            type="checkbox"
            checked={settings.ruvectorEnabled}
            onChange={() => handleToggle('ruvectorEnabled')}
            disabled={saving}
          />
        </div>
      </section>

      {saving && <div className="saving-indicator">Saving...</div>}
    </div>
  );
};
