import React, { useMemo } from 'react';
import { Card } from '../../design-system/components/Card';
import { createLogger } from '../../../utils/logger';
import { MCPBridgeStatus } from '../types';

const logger = createLogger('MCPBridgeMonitor');

interface MCPBridgeMonitorProps {
  bridges: MCPBridgeStatus[];
  className?: string;
}

export const MCPBridgeMonitor: React.FC<MCPBridgeMonitorProps> = ({
  bridges,
  className = ''
}) => {
  const bridgeStats = useMemo(() => {
    return {
      total: bridges.length,
      active: bridges.filter(b => b.status === 'active').length,
      idle: bridges.filter(b => b.status === 'idle').length,
      error: bridges.filter(b => b.status === 'error').length,
      disconnected: bridges.filter(b => b.status === 'disconnected').length,
      avgLatency: bridges.length > 0
        ? bridges.reduce((sum, b) => sum + b.latency, 0) / bridges.length
        : 0,
      totalMessages: bridges.reduce((sum, b) => sum + b.messageCount, 0),
      totalErrors: bridges.reduce((sum, b) => sum + b.errorCount, 0)
    };
  }, [bridges]);

  const getStatusColor = (status: MCPBridgeStatus['status']) => {
    switch (status) {
      case 'active': return 'text-green-600 bg-green-50 border-green-200';
      case 'idle': return 'text-blue-600 bg-blue-50 border-blue-200';
      case 'error': return 'text-red-600 bg-red-50 border-red-200';
      case 'disconnected': return 'text-gray-600 bg-gray-50 border-gray-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getStatusIcon = (status: MCPBridgeStatus['status']) => {
    switch (status) {
      case 'active': return '🟢';
      case 'idle': return '🔵';
      case 'error': return '🔴';
      case 'disconnected': return '⚫';
      default: return '❓';
    }
  };

  const getLatencyColor = (latency: number) => {
    if (latency < 50) return 'text-green-600';
    if (latency < 100) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <Card className={className}>
      <div className="p-4">
        <h3 className="text-lg font-semibold mb-4">MCP Bridge Monitor</h3>

        {/* Overview Statistics */}
        <div className="grid grid-cols-2 gap-3 mb-6">
          <div className="text-center p-3 bg-blue-50 rounded-lg">
            <div className="text-xl font-bold text-blue-600">{bridgeStats.total}</div>
            <div className="text-sm text-blue-700">Total Bridges</div>
          </div>
          <div className="text-center p-3 bg-green-50 rounded-lg">
            <div className="text-xl font-bold text-green-600">{bridgeStats.active}</div>
            <div className="text-sm text-green-700">Active</div>
          </div>
          <div className="text-center p-3 bg-gray-50 rounded-lg">
            <div className={`text-xl font-bold ${getLatencyColor(bridgeStats.avgLatency)}`}>
              {bridgeStats.avgLatency.toFixed(0)}ms
            </div>
            <div className="text-sm text-gray-700">Avg Latency</div>
          </div>
          <div className="text-center p-3 bg-purple-50 rounded-lg">
            <div className="text-xl font-bold text-purple-600">{bridgeStats.totalMessages}</div>
            <div className="text-sm text-purple-700">Messages</div>
          </div>
        </div>

        {/* Bridge List */}
        <div>
          <h4 className="font-medium mb-3">Bridge Status</h4>
          <div className="space-y-3 max-h-64 overflow-y-auto">
            {bridges.length === 0 ? (
              <div className="text-center text-gray-500 py-6">
                No MCP bridges detected
              </div>
            ) : (
              bridges.map(bridge => {
                const timeSinceHeartbeat = Date.now() - bridge.lastHeartbeat.getTime();
                const isStale = timeSinceHeartbeat > 30000; // 30 seconds

                return (
                  <div
                    key={bridge.bridgeId}
                    className={`p-3 rounded-lg border transition-colors ${getStatusColor(bridge.status)} ${
                      isStale ? 'opacity-60' : ''
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-lg">{getStatusIcon(bridge.status)}</span>
                        <div>
                          <div className="font-medium text-sm">
                            {bridge.bridgeId}
                          </div>
                          <div className="text-xs opacity-75">
                            {bridge.status} • {bridge.isConnected ? 'Connected' : 'Disconnected'}
                          </div>
                        </div>
                      </div>

                      <div className="text-right text-xs">
                        <div className={getLatencyColor(bridge.latency)}>
                          {bridge.latency}ms
                        </div>
                        <div className="text-gray-500">
                          {bridge.messageCount} msgs
                        </div>
                      </div>
                    </div>

                    {/* Detailed stats */}
                    <div className="mt-2 grid grid-cols-3 gap-3 text-xs">
                      <div>
                        <span className="text-gray-600">Messages:</span>
                        <div className="font-mono">{bridge.messageCount.toLocaleString()}</div>
                      </div>
                      <div>
                        <span className="text-gray-600">Errors:</span>
                        <div className="font-mono text-red-600">{bridge.errorCount}</div>
                      </div>
                      <div>
                        <span className="text-gray-600">Last Beat:</span>
                        <div className="font-mono">
                          {Math.floor(timeSinceHeartbeat / 1000)}s ago
                        </div>
                      </div>
                    </div>

                    {/* Health indicator */}
                    <div className="mt-2">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-gray-600">Health:</span>
                        <span>{bridge.errorCount === 0 ? '100%' : `${Math.max(0, 100 - bridge.errorCount * 10)}%`}</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-1.5 mt-1">
                        <div
                          className={`h-1.5 rounded-full transition-all ${
                            bridge.errorCount === 0 ? 'bg-green-500' :
                            bridge.errorCount < 5 ? 'bg-yellow-500' : 'bg-red-500'
                          }`}
                          style={{
                            width: `${Math.max(0, 100 - bridge.errorCount * 10)}%`
                          }}
                        />
                      </div>
                    </div>

                    {/* Error indicator */}
                    {bridge.errorCount > 0 && (
                      <div className="mt-2 text-xs text-red-600 bg-red-50 p-1 rounded">
                        ⚠️ {bridge.errorCount} error{bridge.errorCount !== 1 ? 's' : ''} detected
                      </div>
                    )}

                    {/* Stale connection warning */}
                    {isStale && (
                      <div className="mt-2 text-xs text-orange-600 bg-orange-50 p-1 rounded">
                        ⏰ Connection may be stale (no heartbeat for {Math.floor(timeSinceHeartbeat / 1000)}s)
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Connection Health Summary */}
        {bridges.length > 0 && (
          <div className="mt-4 pt-4 border-t border-gray-200">
            <div className="text-sm">
              <div className="flex justify-between items-center mb-2">
                <span className="text-gray-600">Overall Health:</span>
                <span className={`font-medium ${
                  bridgeStats.error === 0 && bridgeStats.disconnected === 0 ? 'text-green-600' :
                  bridgeStats.error > bridgeStats.active ? 'text-red-600' :
                  'text-yellow-600'
                }`}>
                  {bridgeStats.error === 0 && bridgeStats.disconnected === 0 ? 'Healthy' :
                   bridgeStats.error > bridgeStats.active ? 'Critical' :
                   'Warning'}
                </span>
              </div>
              <div className="grid grid-cols-4 gap-2 text-xs">
                <div className="text-center">
                  <div className="text-green-600 font-medium">{bridgeStats.active}</div>
                  <div>Active</div>
                </div>
                <div className="text-center">
                  <div className="text-blue-600 font-medium">{bridgeStats.idle}</div>
                  <div>Idle</div>
                </div>
                <div className="text-center">
                  <div className="text-red-600 font-medium">{bridgeStats.error}</div>
                  <div>Error</div>
                </div>
                <div className="text-center">
                  <div className="text-gray-600 font-medium">{bridgeStats.disconnected}</div>
                  <div>Offline</div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};