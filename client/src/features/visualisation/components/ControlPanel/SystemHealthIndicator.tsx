import React, { useEffect, useState } from 'react';
import { Activity, Wifi, Database, Server, Check, AlertCircle, Loader, Filter } from 'lucide-react';
import { webSocketService } from '../../../../store/websocketStore';

interface ConnectionStatus {
  websocket: 'connected' | 'connecting' | 'disconnected';
  metadata: 'loaded' | 'loading' | 'error' | 'none';
  nodes: number;
  edges: number;
  mcpSwarm: 'connected' | 'disconnected' | 'unknown';
  visibleNodes?: number;
  totalNodes?: number;
}

interface SystemHealthIndicatorProps {
  graphData?: {
    nodes: any[];
    edges: any[];
  };
  mcpConnected?: boolean;
  websocketStatus?: 'connected' | 'connecting' | 'disconnected';
  metadataStatus?: 'loaded' | 'loading' | 'error' | 'none';
}

export const SystemHealthIndicator: React.FC<SystemHealthIndicatorProps> = ({
  graphData,
  mcpConnected = false,
  websocketStatus = 'disconnected',
  metadataStatus = 'none'
}) => {
  const [status, setStatus] = useState<ConnectionStatus>({
    websocket: websocketStatus,
    metadata: metadataStatus,
    nodes: graphData?.nodes?.length || 0,
    edges: graphData?.edges?.length || 0,
    mcpSwarm: mcpConnected ? 'connected' : 'disconnected'
  });

  useEffect(() => {
    setStatus({
      websocket: websocketStatus,
      metadata: metadataStatus,
      nodes: graphData?.nodes?.length || 0,
      edges: graphData?.edges?.length || 0,
      mcpSwarm: mcpConnected ? 'connected' : 'disconnected'
    });
  }, [graphData, mcpConnected, websocketStatus, metadataStatus]);

  // Listen for filter updates from WebSocket
  useEffect(() => {
    if (webSocketService) {
      const unsubscribe = webSocketService.on('filterApplied', (data: unknown) => {
        const filterData = data as { visibleNodes: number; totalNodes: number };
        setStatus(prev => ({
          ...prev,
          visibleNodes: filterData.visibleNodes,
          totalNodes: filterData.totalNodes
        }));
      });
      return unsubscribe;
    }
  }, []);

  const isFullyConnected =
    status.websocket === 'connected' &&
    status.metadata === 'loaded' &&
    status.nodes > 0;

  const getStatusColor = (connected: boolean | string): string => {
    if (connected === true || connected === 'connected' || connected === 'loaded') {
      return '#22c55e'; // green
    }
    if (connected === 'connecting' || connected === 'loading') {
      return '#f59e0b'; // amber
    }
    return '#ef4444'; // red
  };

  const getStatusIcon = (connected: boolean | string) => {
    if (connected === true || connected === 'connected' || connected === 'loaded') {
      return <Check size={8} />;
    }
    if (connected === 'connecting' || connected === 'loading') {
      return <Loader size={8} className="animate-spin" />;
    }
    return <AlertCircle size={8} />;
  };

  return (
    <div style={{
      background: isFullyConnected
        ? 'linear-gradient(135deg, rgba(34,197,94,0.15), rgba(16,185,129,0.1))'
        : 'linear-gradient(135deg, rgba(239,68,68,0.15), rgba(245,158,11,0.1))',
      border: `1px solid ${isFullyConnected ? 'rgba(34,197,94,0.3)' : 'rgba(239,68,68,0.3)'}`,
      borderRadius: '4px',
      padding: '6px 8px',
      marginBottom: '6px'
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        marginBottom: '6px',
        color: isFullyConnected ? '#22c55e' : '#f59e0b',
        fontWeight: '600',
        fontSize: '10px'
      }}>
        <Activity size={12} />
        <span>System Status</span>
        <div style={{
          marginLeft: 'auto',
          width: '8px',
          height: '8px',
          borderRadius: '50%',
          background: isFullyConnected ? '#22c55e' : '#f59e0b',
          boxShadow: `0 0 6px ${isFullyConnected ? '#22c55e' : '#f59e0b'}`
        }} />
      </div>

      {/* Status Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(2, 1fr)',
        gap: '4px',
        fontSize: '9px'
      }}>
        {/* WebSocket */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
          padding: '3px 6px',
          background: 'rgba(255,255,255,0.05)',
          borderRadius: '3px'
        }}>
          <Wifi size={10} style={{ color: getStatusColor(status.websocket) }} />
          <span style={{ color: 'rgba(255,255,255,0.7)' }}>WS</span>
          <span style={{
            marginLeft: 'auto',
            color: getStatusColor(status.websocket),
            display: 'flex',
            alignItems: 'center',
            gap: '2px'
          }}>
            {getStatusIcon(status.websocket)}
          </span>
        </div>

        {/* Metadata */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
          padding: '3px 6px',
          background: 'rgba(255,255,255,0.05)',
          borderRadius: '3px'
        }}>
          <Database size={10} style={{ color: getStatusColor(status.metadata) }} />
          <span style={{ color: 'rgba(255,255,255,0.7)' }}>Meta</span>
          <span style={{
            marginLeft: 'auto',
            color: getStatusColor(status.metadata),
            display: 'flex',
            alignItems: 'center',
            gap: '2px'
          }}>
            {getStatusIcon(status.metadata)}
          </span>
        </div>

        {/* Nodes */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
          padding: '3px 6px',
          background: 'rgba(255,255,255,0.05)',
          borderRadius: '3px'
        }}>
          <div style={{
            width: '10px',
            height: '10px',
            borderRadius: '50%',
            background: status.nodes > 0 ? '#22c55e' : '#ef4444',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }} />
          <span style={{ color: 'rgba(255,255,255,0.7)' }}>Nodes</span>
          <span style={{
            marginLeft: 'auto',
            color: status.nodes > 0 ? '#22c55e' : '#ef4444',
            fontWeight: '600'
          }}>
            {status.nodes.toLocaleString()}
          </span>
        </div>

        {/* Filtered Nodes (if filter active) */}
        {status.visibleNodes !== undefined && status.totalNodes !== undefined && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
            padding: '3px 6px',
            background: 'rgba(255,255,255,0.05)',
            borderRadius: '3px'
          }}>
            <Filter size={10} style={{ color: '#a855f7' }} />
            <span style={{ color: 'rgba(255,255,255,0.7)' }}>Visible</span>
            <span style={{
              marginLeft: 'auto',
              color: '#a855f7',
              fontWeight: '600',
              fontSize: '8px'
            }}>
              {status.visibleNodes}/{status.totalNodes}
            </span>
          </div>
        )}

        {/* MCP Swarm */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
          padding: '3px 6px',
          background: 'rgba(255,255,255,0.05)',
          borderRadius: '3px'
        }}>
          <Server size={10} style={{ color: getStatusColor(status.mcpSwarm) }} />
          <span style={{ color: 'rgba(255,255,255,0.7)' }}>MCP</span>
          <span style={{
            marginLeft: 'auto',
            color: getStatusColor(status.mcpSwarm),
            display: 'flex',
            alignItems: 'center',
            gap: '2px'
          }}>
            {getStatusIcon(status.mcpSwarm)}
          </span>
        </div>
      </div>

      {/* Sync status text */}
      <div style={{
        marginTop: '6px',
        fontSize: '8px',
        color: 'rgba(255,255,255,0.5)',
        textAlign: 'center'
      }}>
        {isFullyConnected
          ? 'All systems synchronized'
          : 'Waiting for connections...'}
      </div>
    </div>
  );
};
