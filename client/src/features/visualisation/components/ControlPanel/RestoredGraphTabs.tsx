

import React, { useState, useCallback } from 'react';
import { Eye, Zap, TrendingUp, MousePointer2, Download } from 'lucide-react';
import { useSettingsStore } from '../../../../store/settingsStore';
import type { GraphNode, GraphEdge } from '@/features/graph/types/graphTypes';

interface GraphTabProps {
  graphId?: string;
  graphData?: {
    nodes: GraphNode[];
    edges: GraphEdge[];
  };
  otherGraphData?: any;
  onExport?: (format: string, options: any) => void;
}

// Toggle Switch Component
const Toggle: React.FC<{
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
}> = ({ checked, onChange, label }) => (
  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '4px 0' }}>
    <label style={{ fontSize: '9px', color: 'white' }}>{label}</label>
    <button
      onClick={() => onChange(!checked)}
      style={{
        width: '32px',
        height: '16px',
        borderRadius: '8px',
        border: 'none',
        background: checked ? '#10b981' : '#4b5563',
        position: 'relative',
        cursor: 'pointer',
        transition: 'background 0.2s',
        flexShrink: 0
      }}
    >
      <div style={{
        width: '12px',
        height: '12px',
        borderRadius: '50%',
        background: 'white',
        position: 'absolute',
        top: '2px',
        left: checked ? '18px' : '2px',
        transition: 'left 0.2s'
      }} />
    </button>
  </div>
);

// Section Header
const SectionHeader: React.FC<{ icon: React.ComponentType<any>; title: string; color: string }> = ({ icon: Icon, title, color }) => (
  <div style={{
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    padding: '6px 0',
    borderBottom: `1px solid ${color}30`,
    marginBottom: '6px'
  }}>
    <Icon size={12} style={{ color }} />
    <span style={{ fontSize: '10px', fontWeight: '600', color }}>{title}</span>
  </div>
);

export const RestoredGraphVisualisationTab: React.FC<GraphTabProps> = () => {
  const settings = useSettingsStore(state => state.settings);
  const updateSettings = useSettingsStore(state => state.updateSettings);

  // Type assertion for extended visualisation settings
  const visSettings = (settings?.visualisation ?? {}) as any;
  const syncEnabled = visSettings?.sync?.enabled ?? false;
  const cameraSync = visSettings?.sync?.camera ?? true;
  const selectionSync = visSettings?.sync?.selection ?? true;
  const animationsEnabled = visSettings?.animations?.enabled ?? true;
  const bloomEffect = visSettings?.effects?.bloom ?? false;
  const glowEffect = visSettings?.effects?.glow ?? true;

  return (
    <div style={{ padding: '4px', color: 'white' }}>
      <SectionHeader icon={Eye} title="Synchronisation" color="#a78bfa" />
      <Toggle
        checked={syncEnabled}
        onChange={(val) => updateSettings((draft) => {
          if (!draft.visualisation) draft.visualisation = {} as any;
          if (!(draft.visualisation as any).sync) (draft.visualisation as any).sync = {};
          (draft.visualisation as any).sync.enabled = val;
        })}
        label="Enable Sync"
      />
      {syncEnabled && (
        <div style={{ marginLeft: '6px', paddingLeft: '6px', borderLeft: '1px solid rgba(167,139,250,0.3)' }}>
          <Toggle
            checked={cameraSync}
            onChange={(val) => updateSettings((draft) => {
              if (!draft.visualisation) draft.visualisation = {} as any;
              if (!(draft.visualisation as any).sync) (draft.visualisation as any).sync = {};
              (draft.visualisation as any).sync.camera = val;
            })}
            label="Camera"
          />
          <Toggle
            checked={selectionSync}
            onChange={(val) => updateSettings((draft) => {
              if (!draft.visualisation) draft.visualisation = {} as any;
              if (!(draft.visualisation as any).sync) (draft.visualisation as any).sync = {};
              (draft.visualisation as any).sync.selection = val;
            })}
            label="Selection"
          />
        </div>
      )}

      <SectionHeader icon={Zap} title="Animations" color="#fbbf24" />
      <Toggle
        checked={animationsEnabled}
        onChange={(val) => updateSettings((draft) => {
          if (!draft.visualisation) draft.visualisation = {} as any;
          if (!(draft.visualisation as any).animations) (draft.visualisation as any).animations = {};
          (draft.visualisation as any).animations.enabled = val;
        })}
        label="Enable Animations"
      />

      <SectionHeader icon={Zap} title="Visual Effects" color="#ec4899" />
      <Toggle
        checked={bloomEffect}
        onChange={(val) => updateSettings((draft) => {
          if (!draft.visualisation) draft.visualisation = {} as any;
          if (!(draft.visualisation as any).effects) (draft.visualisation as any).effects = {};
          (draft.visualisation as any).effects.bloom = val;
        })}
        label="Bloom"
      />
      <Toggle
        checked={glowEffect}
        onChange={(val) => updateSettings((draft) => {
          if (!draft.visualisation) draft.visualisation = {} as any;
          if (!(draft.visualisation as any).effects) (draft.visualisation as any).effects = {};
          (draft.visualisation as any).effects.glow = val;
        })}
        label="Glow"
      />
    </div>
  );
};

export const RestoredGraphOptimisationTab: React.FC<GraphTabProps> = ({ graphData }) => {
  const settings = useSettingsStore(state => state.settings);
  const updateSettings = useSettingsStore(state => state.updateSettings);

  // Type assertion for extended performance settings
  const perfSettings = (settings as any)?.performance || {};
  const autoOptimize = perfSettings?.autoOptimize ?? false;
  const simplifyEdges = perfSettings?.simplifyEdges ?? true;
  const cullDistance = perfSettings?.cullDistance ?? 50;

  return (
    <div style={{ padding: '4px', color: 'white' }}>
      <SectionHeader icon={TrendingUp} title="Performance" color="#f59e0b" />

      <div style={{ fontSize: '9px', color: 'rgba(255,255,255,0.6)', marginBottom: '4px' }}>
        Nodes: {graphData?.nodes?.length || 0} | Edges: {graphData?.edges?.length || 0}
      </div>

      <Toggle
        checked={autoOptimize}
        onChange={(val) => updateSettings((draft) => {
          if (!(draft as any).performance) (draft as any).performance = {};
          (draft as any).performance.autoOptimize = val;
        })}
        label="Auto Optimize"
      />
      <Toggle
        checked={simplifyEdges}
        onChange={(val) => updateSettings((draft) => {
          if (!(draft as any).performance) (draft as any).performance = {};
          (draft as any).performance.simplifyEdges = val;
        })}
        label="Simplify Edges"
      />

      <div style={{ padding: '6px 0' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
          <label style={{ fontSize: '9px', color: 'white' }}>Cull Distance</label>
          <span style={{ fontSize: '9px', color: 'rgba(255,255,255,0.7)' }}>{cullDistance}</span>
        </div>
        <input
          type="range"
          value={cullDistance}
          onChange={(e) => updateSettings((draft) => {
            if (!(draft as any).performance) (draft as any).performance = {};
            (draft as any).performance.cullDistance = Number(e.target.value);
          })}
          min={10}
          max={100}
          step={5}
          style={{
            width: '100%',
            height: '3px',
            borderRadius: '2px',
            background: 'rgba(245,158,11,0.3)',
            outline: 'none',
            cursor: 'pointer'
          }}
        />
      </div>

      <button
        onClick={() => console.log('Optimize triggered')}
        style={{
          width: '100%',
          background: 'linear-gradient(to right, #f59e0b, #d97706)',
          color: 'white',
          padding: '6px',
          borderRadius: '3px',
          fontSize: '10px',
          fontWeight: '600',
          border: 'none',
          cursor: 'pointer',
          marginTop: '4px'
        }}
      >
        Optimize Now
      </button>
    </div>
  );
};

export const RestoredGraphInteractionTab: React.FC<GraphTabProps> = () => {
  const settings = useSettingsStore(state => state.settings);
  const updateSettings = useSettingsStore(state => state.updateSettings);

  // Type assertion for extended interaction settings
  const interactionSettings = (settings as any)?.interaction || {};
  const enableHover = interactionSettings?.enableHover ?? true;
  const enableClick = interactionSettings?.enableClick ?? true;
  const enableDrag = interactionSettings?.enableDrag ?? true;
  const hoverDelay = interactionSettings?.hoverDelay ?? 200;

  return (
    <div style={{ padding: '4px', color: 'white' }}>
      <SectionHeader icon={MousePointer2} title="Interaction" color="#3b82f6" />

      <Toggle
        checked={enableHover}
        onChange={(val) => updateSettings((draft) => {
          if (!(draft as any).interaction) (draft as any).interaction = {};
          (draft as any).interaction.enableHover = val;
        })}
        label="Hover Effects"
      />
      <Toggle
        checked={enableClick}
        onChange={(val) => updateSettings((draft) => {
          if (!(draft as any).interaction) (draft as any).interaction = {};
          (draft as any).interaction.enableClick = val;
        })}
        label="Click to Select"
      />
      <Toggle
        checked={enableDrag}
        onChange={(val) => updateSettings((draft) => {
          if (!(draft as any).interaction) (draft as any).interaction = {};
          (draft as any).interaction.enableDrag = val;
        })}
        label="Drag Nodes"
      />

      <div style={{ padding: '6px 0' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
          <label style={{ fontSize: '9px', color: 'white' }}>Hover Delay</label>
          <span style={{ fontSize: '9px', color: 'rgba(255,255,255,0.7)' }}>{hoverDelay}ms</span>
        </div>
        <input
          type="range"
          value={hoverDelay}
          onChange={(e) => updateSettings((draft) => {
            if (!(draft as any).interaction) (draft as any).interaction = {};
            (draft as any).interaction.hoverDelay = Number(e.target.value);
          })}
          min={0}
          max={500}
          step={50}
          style={{
            width: '100%',
            height: '3px',
            borderRadius: '2px',
            background: 'rgba(59,130,246,0.3)',
            outline: 'none',
            cursor: 'pointer'
          }}
        />
      </div>
    </div>
  );
};

export const RestoredGraphExportTab: React.FC<GraphTabProps> = ({ graphData }) => {
  const settings = useSettingsStore(state => state.settings);
  const updateSettings = useSettingsStore(state => state.updateSettings);

  // Type assertion for extended export settings
  const exportSettings = (settings as any)?.export || {};
  const format = exportSettings?.format ?? 'json';
  const includeMetadata = exportSettings?.includeMetadata ?? true;

  const handleExport = () => {
    console.log('Exporting as', format, 'with metadata:', includeMetadata);
    
  };

  return (
    <div style={{ padding: '4px', color: 'white' }}>
      <SectionHeader icon={Download} title="Export" color="#ec4899" />

      <div style={{ fontSize: '9px', color: 'rgba(255,255,255,0.6)', marginBottom: '4px' }}>
        Ready to export: {graphData?.nodes?.length || 0} nodes, {graphData?.edges?.length || 0} edges
      </div>

      <div style={{ padding: '4px 0' }}>
        <label style={{ fontSize: '9px', display: 'block', marginBottom: '4px', color: 'white' }}>
          Format
        </label>
        <select
          value={format}
          onChange={(e) => updateSettings((draft) => {
            if (!(draft as any).export) (draft as any).export = {};
            (draft as any).export.format = e.target.value;
          })}
          style={{
            width: '100%',
            background: 'rgba(255,255,255,0.05)',
            border: '1px solid rgba(236,72,153,0.3)',
            borderRadius: '3px',
            fontSize: '10px',
            color: 'white',
            padding: '4px 6px',
            cursor: 'pointer'
          }}
        >
          <option value="json" style={{ background: '#1f2937' }}>JSON</option>
          <option value="csv" style={{ background: '#1f2937' }}>CSV</option>
          <option value="graphml" style={{ background: '#1f2937' }}>GraphML</option>
          <option value="gexf" style={{ background: '#1f2937' }}>GEXF</option>
        </select>
      </div>

      <Toggle
        checked={includeMetadata}
        onChange={(val) => updateSettings((draft) => {
          if (!(draft as any).export) (draft as any).export = {};
          (draft as any).export.includeMetadata = val;
        })}
        label="Include Metadata"
      />

      <button
        onClick={handleExport}
        style={{
          width: '100%',
          background: 'linear-gradient(to right, #ec4899, #db2777)',
          color: 'white',
          padding: '6px',
          borderRadius: '3px',
          fontSize: '10px',
          fontWeight: '600',
          border: 'none',
          cursor: 'pointer',
          marginTop: '4px'
        }}
      >
        Export Graph
      </button>
    </div>
  );
};

export const RestoredGraphAnalysisTab: React.FC<GraphTabProps> = ({ graphData }) => {
  const [analyzing, setAnalyzing] = useState(false);

  const runAnalysis = () => {
    setAnalyzing(true);
    setTimeout(() => setAnalyzing(false), 1000);
  };

  return (
    <div style={{ padding: '4px', color: 'white' }}>
      <SectionHeader icon={TrendingUp} title="Graph Analysis" color="#10b981" />

      <div style={{ fontSize: '9px', color: 'rgba(255,255,255,0.6)', marginBottom: '4px' }}>
        <div>Nodes: {graphData?.nodes?.length || 0}</div>
        <div>Edges: {graphData?.edges?.length || 0}</div>
        <div>Density: {graphData?.nodes?.length ?
          ((graphData?.edges?.length || 0) / (graphData.nodes.length * (graphData.nodes.length - 1)) * 100).toFixed(1)
          : 0}%</div>
      </div>

      <button
        onClick={runAnalysis}
        disabled={analyzing}
        style={{
          width: '100%',
          background: analyzing ? 'rgba(16,185,129,0.5)' : 'linear-gradient(to right, #10b981, #059669)',
          color: 'white',
          padding: '6px',
          borderRadius: '3px',
          fontSize: '10px',
          fontWeight: '600',
          border: 'none',
          cursor: analyzing ? 'not-allowed' : 'pointer'
        }}
      >
        {analyzing ? 'Analyzing...' : 'Run Full Analysis'}
      </button>

      <div style={{
        marginTop: '4px',
        padding: '6px',
        background: 'rgba(16,185,129,0.1)',
        border: '1px solid rgba(16,185,129,0.3)',
        borderRadius: '3px',
        fontSize: '9px'
      }}>
        <div style={{ fontWeight: '600', marginBottom: '2px' }}>Available Analyses:</div>
        <div>• Centrality Measures</div>
        <div>• Community Detection</div>
        <div>• Shortest Paths</div>
        <div>• Clustering Coefficient</div>
      </div>
    </div>
  );
};
