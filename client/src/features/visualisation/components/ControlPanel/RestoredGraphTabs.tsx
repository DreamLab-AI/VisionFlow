/**
 * Restored Graph Feature Tabs - Inline Styles Version
 */

import React, { useState, useCallback } from 'react';
import { Eye, Zap, TrendingUp, MousePointer2, Download } from 'lucide-react';
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
  const [syncEnabled, setSyncEnabled] = useState(false);
  const [cameraSync, setCameraSync] = useState(true);
  const [selectionSync, setSelectionSync] = useState(true);
  const [animationsEnabled, setAnimationsEnabled] = useState(true);
  const [bloomEffect, setBloomEffect] = useState(false);
  const [glowEffect, setGlowEffect] = useState(true);

  return (
    <div style={{ padding: '8px', color: 'white' }}>
      <SectionHeader icon={Eye} title="Synchronisation" color="#a78bfa" />
      <Toggle checked={syncEnabled} onChange={setSyncEnabled} label="Enable Sync" />
      {syncEnabled && (
        <div style={{ marginLeft: '8px', paddingLeft: '8px', borderLeft: '1px solid rgba(167,139,250,0.3)' }}>
          <Toggle checked={cameraSync} onChange={setCameraSync} label="Camera" />
          <Toggle checked={selectionSync} onChange={setSelectionSync} label="Selection" />
        </div>
      )}

      <SectionHeader icon={Zap} title="Animations" color="#fbbf24" />
      <Toggle checked={animationsEnabled} onChange={setAnimationsEnabled} label="Enable Animations" />

      <SectionHeader icon={Zap} title="Visual Effects" color="#ec4899" />
      <Toggle checked={bloomEffect} onChange={setBloomEffect} label="Bloom" />
      <Toggle checked={glowEffect} onChange={setGlowEffect} label="Glow" />
    </div>
  );
};

export const RestoredGraphOptimisationTab: React.FC<GraphTabProps> = ({ graphData }) => {
  const [autoOptimize, setAutoOptimize] = useState(false);
  const [simplifyEdges, setSimplifyEdges] = useState(true);
  const [cullDistance, setCullDistance] = useState(50);

  return (
    <div style={{ padding: '8px', color: 'white' }}>
      <SectionHeader icon={TrendingUp} title="Performance" color="#f59e0b" />

      <div style={{ fontSize: '9px', color: 'rgba(255,255,255,0.6)', marginBottom: '6px' }}>
        Nodes: {graphData?.nodes?.length || 0} | Edges: {graphData?.edges?.length || 0}
      </div>

      <Toggle checked={autoOptimize} onChange={setAutoOptimize} label="Auto Optimize" />
      <Toggle checked={simplifyEdges} onChange={setSimplifyEdges} label="Simplify Edges" />

      <div style={{ padding: '6px 0' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
          <label style={{ fontSize: '9px', color: 'white' }}>Cull Distance</label>
          <span style={{ fontSize: '9px', color: 'rgba(255,255,255,0.7)' }}>{cullDistance}</span>
        </div>
        <input
          type="range"
          value={cullDistance}
          onChange={(e) => setCullDistance(Number(e.target.value))}
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
          marginTop: '6px'
        }}
      >
        Optimize Now
      </button>
    </div>
  );
};

export const RestoredGraphInteractionTab: React.FC<GraphTabProps> = () => {
  const [enableHover, setEnableHover] = useState(true);
  const [enableClick, setEnableClick] = useState(true);
  const [enableDrag, setEnableDrag] = useState(true);
  const [hoverDelay, setHoverDelay] = useState(200);

  return (
    <div style={{ padding: '8px', color: 'white' }}>
      <SectionHeader icon={MousePointer2} title="Interaction" color="#3b82f6" />

      <Toggle checked={enableHover} onChange={setEnableHover} label="Hover Effects" />
      <Toggle checked={enableClick} onChange={setEnableClick} label="Click to Select" />
      <Toggle checked={enableDrag} onChange={setEnableDrag} label="Drag Nodes" />

      <div style={{ padding: '6px 0' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
          <label style={{ fontSize: '9px', color: 'white' }}>Hover Delay</label>
          <span style={{ fontSize: '9px', color: 'rgba(255,255,255,0.7)' }}>{hoverDelay}ms</span>
        </div>
        <input
          type="range"
          value={hoverDelay}
          onChange={(e) => setHoverDelay(Number(e.target.value))}
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
  const [format, setFormat] = useState('json');
  const [includeMetadata, setIncludeMetadata] = useState(true);

  const handleExport = () => {
    console.log('Exporting as', format, 'with metadata:', includeMetadata);
    // Export logic would go here
  };

  return (
    <div style={{ padding: '8px', color: 'white' }}>
      <SectionHeader icon={Download} title="Export" color="#ec4899" />

      <div style={{ fontSize: '9px', color: 'rgba(255,255,255,0.6)', marginBottom: '8px' }}>
        Ready to export: {graphData?.nodes?.length || 0} nodes, {graphData?.edges?.length || 0} edges
      </div>

      <div style={{ padding: '4px 0' }}>
        <label style={{ fontSize: '9px', display: 'block', marginBottom: '4px', color: 'white' }}>
          Format
        </label>
        <select
          value={format}
          onChange={(e) => setFormat(e.target.value)}
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

      <Toggle checked={includeMetadata} onChange={setIncludeMetadata} label="Include Metadata" />

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
          marginTop: '8px'
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
    <div style={{ padding: '8px', color: 'white' }}>
      <SectionHeader icon={TrendingUp} title="Graph Analysis" color="#10b981" />

      <div style={{ fontSize: '9px', color: 'rgba(255,255,255,0.6)', marginBottom: '8px' }}>
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
        marginTop: '8px',
        padding: '6px',
        background: 'rgba(16,185,129,0.1)',
        border: '1px solid rgba(16,185,129,0.3)',
        borderRadius: '3px',
        fontSize: '9px'
      }}>
        <div style={{ fontWeight: '600', marginBottom: '4px' }}>Available Analyses:</div>
        <div>• Centrality Measures</div>
        <div>• Community Detection</div>
        <div>• Shortest Paths</div>
        <div>• Clustering Coefficient</div>
      </div>
    </div>
  );
};
