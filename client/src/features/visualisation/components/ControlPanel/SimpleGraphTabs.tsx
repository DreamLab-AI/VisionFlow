

import React from 'react';
import type { GraphNode, GraphEdge } from '@/features/graph/types/graphTypes';

interface SimpleGraphTabProps {
  graphId?: string;
  graphData?: {
    nodes: GraphNode[];
    edges: GraphEdge[];
  };
  otherGraphData?: any;
  onExport?: (format: string, options: any) => void;
}

export const SimpleGraphAnalysisTab: React.FC<SimpleGraphTabProps> = ({ graphData }) => (
  <div style={{ padding: '8px', color: 'white' }}>
    <h4 style={{ fontSize: '11px', fontWeight: '600', marginBottom: '6px', color: '#10b981' }}>
      Graph Analysis
    </h4>
    <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.7)', marginBottom: '4px' }}>
      Nodes: {graphData?.nodes?.length || 0}
    </div>
    <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.7)', marginBottom: '6px' }}>
      Edges: {graphData?.edges?.length || 0}
    </div>
    <div style={{
      padding: '8px',
      background: 'rgba(16,185,129,0.1)',
      border: '1px solid rgba(16,185,129,0.3)',
      borderRadius: '3px',
      fontSize: '9px',
      color: 'rgba(255,255,255,0.8)'
    }}>
      Full analysis features will be restored once styling system is fixed.
    </div>
  </div>
);

export const SimpleGraphVisualisationTab: React.FC<SimpleGraphTabProps> = () => (
  <div style={{ padding: '8px', color: 'white' }}>
    <h4 style={{ fontSize: '11px', fontWeight: '600', marginBottom: '6px', color: '#a78bfa' }}>
      Visualisation
    </h4>
    <div style={{
      padding: '8px',
      background: 'rgba(147,51,234,0.1)',
      border: '1px solid rgba(147,51,234,0.3)',
      borderRadius: '3px',
      fontSize: '9px',
      color: 'rgba(255,255,255,0.8)'
    }}>
      Visualisation controls will be restored once styling system is fixed.
    </div>
  </div>
);

export const SimpleGraphOptimisationTab: React.FC<SimpleGraphTabProps> = ({ graphData }) => (
  <div style={{ padding: '8px', color: 'white' }}>
    <h4 style={{ fontSize: '11px', fontWeight: '600', marginBottom: '6px', color: '#f59e0b' }}>
      Optimisation
    </h4>
    <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.7)', marginBottom: '6px' }}>
      Graph nodes: {graphData?.nodes?.length || 0}
    </div>
    <div style={{
      padding: '8px',
      background: 'rgba(245,158,11,0.1)',
      border: '1px solid rgba(245,158,11,0.3)',
      borderRadius: '3px',
      fontSize: '9px',
      color: 'rgba(255,255,255,0.8)'
    }}>
      Optimisation controls will be restored once styling system is fixed.
    </div>
  </div>
);

export const SimpleGraphInteractionTab: React.FC<SimpleGraphTabProps> = () => (
  <div style={{ padding: '8px', color: 'white' }}>
    <h4 style={{ fontSize: '11px', fontWeight: '600', marginBottom: '6px', color: '#3b82f6' }}>
      Interaction
    </h4>
    <div style={{
      padding: '8px',
      background: 'rgba(59,130,246,0.1)',
      border: '1px solid rgba(59,130,246,0.3)',
      borderRadius: '3px',
      fontSize: '9px',
      color: 'rgba(255,255,255,0.8)'
    }}>
      Interaction controls will be restored once styling system is fixed.
    </div>
  </div>
);

export const SimpleGraphExportTab: React.FC<SimpleGraphTabProps> = ({ graphData }) => (
  <div style={{ padding: '8px', color: 'white' }}>
    <h4 style={{ fontSize: '11px', fontWeight: '600', marginBottom: '6px', color: '#ec4899' }}>
      Export
    </h4>
    <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.7)', marginBottom: '6px' }}>
      Ready to export: {graphData?.nodes?.length || 0} nodes
    </div>
    <div style={{
      padding: '8px',
      background: 'rgba(236,72,153,0.1)',
      border: '1px solid rgba(236,72,153,0.3)',
      borderRadius: '3px',
      fontSize: '9px',
      color: 'rgba(255,255,255,0.8)'
    }}>
      Export functionality will be restored once styling system is fixed.
    </div>
  </div>
);
