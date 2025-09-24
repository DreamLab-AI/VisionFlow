import React, { useState } from 'react';
import { Html } from '@react-three/drei';
import useHybridSystemStatus from '../hooks/useHybridSystemStatus';

interface HybridSystemDashboardProps {
  position?: [number, number, number];
  onSystemChange?: (status: any) => void;
}

export const HybridSystemDashboard: React.FC<HybridSystemDashboardProps> = ({
  position = [40, 15, 0],
  onSystemChange
}) => {
  const {
    status,
    isLoading,
    error,
    isSystemHealthy,
    isSystemDegraded,
    isSystemCritical,
    isDockerAvailable,
    isMcpAvailable,
    isConnected,
    refresh,
    reconnect,
    spawnSwarm,
    stopSwarm,
    getPerformanceReport,
  } = useHybridSystemStatus({
    pollingInterval: 15000, // 15 seconds
    enableWebSocket: true,
    enablePerformanceMetrics: true,
    autoReconnect: true,
  });

  const [activeTab, setActiveTab] = useState<'overview' | 'sessions' | 'performance' | 'health'>('overview');
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [spawnTask, setSpawnTask] = useState('');
  const [showSpawnDialog, setShowSpawnDialog] = useState(false);

  React.useEffect(() => {
    onSystemChange?.(status);
  }, [status, onSystemChange]);

  const handleSpawnSwarm = async () => {
    if (!spawnTask.trim()) return;

    try {
      await spawnSwarm(spawnTask, {
        priority: 'medium',
        strategy: 'hive-mind',
        method: 'hybrid',
        autoScale: true,
      });

      setSpawnTask('');
      setShowSpawnDialog(false);
    } catch (err) {
      console.error('Failed to spawn swarm:', err);
    }
  };

  const handleStopSwarm = async (sessionId: string) => {
    try {
      await stopSwarm(sessionId);
    } catch (err) {
      console.error('Failed to stop swarm:', err);
    }
  };

  const getSystemStatusColor = () => {
    if (isSystemHealthy) return '#00ff00';
    if (isSystemDegraded) return '#ffaa00';
    if (isSystemCritical) return '#ff0000';
    return '#888888';
  };

  const getHealthIndicatorColor = (health: string) => {
    switch (health) {
      case 'healthy':
      case 'connected':
        return '#00ff00';
      case 'degraded':
      case 'reconnecting':
        return '#ffaa00';
      case 'unavailable':
      case 'disconnected':
        return '#ff0000';
      default:
        return '#888888';
    }
  };

  const formatUptime = (timestamp: string) => {
    const now = new Date();
    const then = new Date(timestamp);
    const diffMs = now.getTime() - then.getTime();
    const diffMinutes = Math.floor(diffMs / 60000);

    if (diffMinutes < 60) {
      return `${diffMinutes}m`;
    } else if (diffMinutes < 1440) {
      return `${Math.floor(diffMinutes / 60)}h ${diffMinutes % 60}m`;
    } else {
      const days = Math.floor(diffMinutes / 1440);
      const hours = Math.floor((diffMinutes % 1440) / 60);
      return `${days}d ${hours}h`;
    }
  };

  const panelStyle: React.CSSProperties = {
    background: 'rgba(0, 0, 0, 0.95)',
    border: `2px solid ${getSystemStatusColor()}`,
    borderRadius: '12px',
    padding: isCollapsed ? '10px' : '20px',
    color: '#FFFFFF',
    fontFamily: 'monospace',
    fontSize: '12px',
    width: isCollapsed ? 'auto' : '420px',
    maxHeight: '700px',
    overflowY: 'auto',
    backdropFilter: 'blur(10px)',
    boxShadow: `0 0 20px ${getSystemStatusColor()}30`,
  };

  const tabStyle = (isActive: boolean): React.CSSProperties => ({
    padding: '6px 12px',
    margin: '0 4px',
    background: isActive ? getSystemStatusColor() : 'transparent',
    color: isActive ? '#000' : getSystemStatusColor(),
    border: `1px solid ${getSystemStatusColor()}`,
    borderRadius: '6px',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    fontSize: '11px',
  });

  const inputStyle: React.CSSProperties = {
    background: `rgba(${getSystemStatusColor().slice(1, 3)}, ${getSystemStatusColor().slice(3, 5)}, ${getSystemStatusColor().slice(5, 7)}, 0.1)`,
    border: `1px solid ${getSystemStatusColor()}`,
    borderRadius: '4px',
    color: getSystemStatusColor(),
    padding: '4px 8px',
    width: '100%',
    marginTop: '4px',
    fontSize: '11px',
  };

  const buttonStyle: React.CSSProperties = {
    ...inputStyle,
    cursor: 'pointer',
    padding: '6px 12px',
    textAlign: 'center' as const,
    transition: 'all 0.3s ease',
  };

  const healthIndicatorStyle = (health: string): React.CSSProperties => ({
    display: 'inline-block',
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    backgroundColor: getHealthIndicatorColor(health),
    marginRight: '8px',
    boxShadow: `0 0 6px ${getHealthIndicatorColor(health)}`,
  });

  if (isLoading && !status.activeSessions.length) {
    return (
      <Html position={position} style={{ pointerEvents: 'auto' }}>
        <div style={panelStyle}>
          <div style={{ textAlign: 'center', padding: '20px' }}>
            <div>Loading hybrid system status...</div>
          </div>
        </div>
      </Html>
    );
  }

  return (
    <Html position={position} style={{ pointerEvents: 'auto' }}>
      <div style={panelStyle}>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: isCollapsed ? '0' : '15px'
        }}>
          <h3 style={{ margin: 0, color: getSystemStatusColor() }}>
            {isCollapsed ? '🔧' : '🔧 Hybrid System Dashboard'}
          </h3>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            {!isCollapsed && (
              <>
                <button
                  onClick={refresh}
                  style={{
                    ...buttonStyle,
                    width: 'auto',
                    padding: '4px 8px',
                    fontSize: '10px',
                  }}
                  title="Refresh status"
                >
                  ↻
                </button>
                {error && (
                  <button
                    onClick={reconnect}
                    style={{
                      ...buttonStyle,
                      width: 'auto',
                      padding: '4px 8px',
                      fontSize: '10px',
                      borderColor: '#ff0000',
                      color: '#ff0000',
                    }}
                    title="Reconnect"
                  >
                    🔌
                  </button>
                )}
              </>
            )}
            <button
              onClick={() => setIsCollapsed(!isCollapsed)}
              style={{
                background: 'transparent',
                border: `1px solid ${getSystemStatusColor()}`,
                color: getSystemStatusColor(),
                borderRadius: '4px',
                cursor: 'pointer',
                padding: '4px 8px',
                fontSize: '10px',
              }}
            >
              {isCollapsed ? '➕' : '➖'}
            </button>
          </div>
        </div>

        {!isCollapsed && (
          <>
            {/* Connection Status */}
            <div style={{
              marginBottom: '15px',
              padding: '10px',
              background: 'rgba(255, 255, 255, 0.05)',
              borderRadius: '6px'
            }}>
              <div style={{ fontSize: '11px', marginBottom: '8px', fontWeight: 'bold' }}>
                System Status
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <span style={healthIndicatorStyle(status.dockerHealth)}></span>
                  <span>Docker: {status.dockerHealth}</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <span style={healthIndicatorStyle(status.mcpHealth)}></span>
                  <span>MCP: {status.mcpHealth}</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <span style={healthIndicatorStyle(isConnected ? 'connected' : 'disconnected')}></span>
                  <span>WebSocket: {isConnected ? 'connected' : 'disconnected'}</span>
                </div>
                <div style={{ fontSize: '10px', color: '#888', marginTop: '4px' }}>
                  Last updated: {formatUptime(status.lastUpdated)} ago
                </div>
              </div>
            </div>

            {error && (
              <div style={{
                background: 'rgba(255, 0, 0, 0.1)',
                border: '1px solid #ff0000',
                borderRadius: '4px',
                padding: '8px',
                marginBottom: '15px',
                fontSize: '10px',
                color: '#ff0000',
              }}>
                Error: {error}
              </div>
            )}

            {/* Tabs */}
            <div style={{ display: 'flex', marginBottom: '15px', flexWrap: 'wrap', gap: '4px' }}>
              <button
                style={tabStyle(activeTab === 'overview')}
                onClick={() => setActiveTab('overview')}
              >
                Overview
              </button>
              <button
                style={tabStyle(activeTab === 'sessions')}
                onClick={() => setActiveTab('sessions')}
              >
                Sessions ({status.activeSessions.length})
              </button>
              <button
                style={tabStyle(activeTab === 'performance')}
                onClick={() => setActiveTab('performance')}
              >
                Performance
              </button>
              <button
                style={tabStyle(activeTab === 'health')}
                onClick={() => setActiveTab('health')}
              >
                Health
              </button>
            </div>

            {/* Overview Tab */}
            {activeTab === 'overview' && (
              <div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '15px' }}>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '20px', fontWeight: 'bold', color: getSystemStatusColor() }}>
                      {status.activeSessions.length}
                    </div>
                    <div style={{ fontSize: '10px' }}>Active Swarms</div>
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '20px', fontWeight: 'bold', color: getSystemStatusColor() }}>
                      {status.networkLatency}ms
                    </div>
                    <div style={{ fontSize: '10px' }}>Network Latency</div>
                  </div>
                </div>

                <div style={{ marginBottom: '15px' }}>
                  <div style={{ fontSize: '11px', marginBottom: '8px', fontWeight: 'bold' }}>
                    Quick Actions
                  </div>
                  <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                    <button
                      onClick={() => setShowSpawnDialog(true)}
                      style={buttonStyle}
                    >
                      Spawn Swarm
                    </button>
                    <button
                      onClick={async () => {
                        try {
                          const report = await getPerformanceReport();
                          console.log('Performance report:', report);
                        } catch (err) {
                          console.error('Failed to get performance report:', err);
                        }
                      }}
                      style={buttonStyle}
                    >
                      Performance Report
                    </button>
                  </div>
                </div>

                {status.failoverActive && (
                  <div style={{
                    background: 'rgba(255, 165, 0, 0.1)',
                    border: '1px solid #ffaa00',
                    borderRadius: '4px',
                    padding: '8px',
                    fontSize: '10px',
                    color: '#ffaa00',
                  }}>
                    ⚠️ Failover mode active - some services may be degraded
                  </div>
                )}
              </div>
            )}

            {/* Sessions Tab */}
            {activeTab === 'sessions' && (
              <div>
                <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                  {status.activeSessions.length === 0 ? (
                    <div style={{ textAlign: 'center', color: '#888', fontSize: '11px', padding: '20px' }}>
                      No active swarms
                    </div>
                  ) : (
                    status.activeSessions.map(session => (
                      <div
                        key={session.sessionId}
                        style={{
                          background: 'rgba(255, 255, 255, 0.05)',
                          border: '1px solid rgba(255, 255, 255, 0.1)',
                          borderRadius: '6px',
                          padding: '10px',
                          marginBottom: '8px',
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                          <div style={{ flex: 1 }}>
                            <div style={{ fontWeight: 'bold', fontSize: '11px', marginBottom: '4px' }}>
                              {session.taskDescription}
                            </div>
                            <div style={{ fontSize: '10px', color: '#888' }}>
                              ID: {session.sessionId.slice(-8)}
                            </div>
                            <div style={{ fontSize: '10px', color: '#888' }}>
                              Workers: {session.activeWorkers} | Method: {session.method}
                            </div>
                            <div style={{ fontSize: '10px', color: '#888' }}>
                              Status: <span style={{ color: getHealthIndicatorColor(session.status) }}>
                                {session.status}
                              </span>
                            </div>
                          </div>
                          <button
                            onClick={() => handleStopSwarm(session.sessionId)}
                            style={{
                              ...buttonStyle,
                              width: '60px',
                              padding: '4px',
                              fontSize: '10px',
                              marginLeft: '8px',
                            }}
                          >
                            Stop
                          </button>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}

            {/* Performance Tab */}
            {activeTab === 'performance' && (
              <div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '15px' }}>
                  <div>
                    <div style={{ fontSize: '10px', marginBottom: '4px' }}>Cache Hit Ratio</div>
                    <div style={{ fontSize: '14px', fontWeight: 'bold' }}>
                      {(status.performance.cacheHitRatio * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '10px', marginBottom: '4px' }}>Avg Response Time</div>
                    <div style={{ fontSize: '14px', fontWeight: 'bold' }}>
                      {status.performance.averageResponseTimeMs.toFixed(0)}ms
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '10px', marginBottom: '4px' }}>Memory Usage</div>
                    <div style={{ fontSize: '14px', fontWeight: 'bold' }}>
                      {status.performance.memoryUsageMb.toFixed(1)}MB
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '10px', marginBottom: '4px' }}>Pool Utilization</div>
                    <div style={{ fontSize: '14px', fontWeight: 'bold' }}>
                      {(status.performance.connectionPoolUtilization * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                <div style={{ fontSize: '10px', marginBottom: '8px' }}>
                  Requests: {status.performance.successfulRequests}/{status.performance.totalRequests}
                  ({((status.performance.successfulRequests / status.performance.totalRequests) * 100 || 0).toFixed(1)}% success)
                </div>

                {status.performance.activeOptimizations.length > 0 && (
                  <div>
                    <div style={{ fontSize: '10px', marginBottom: '4px' }}>Active Optimizations:</div>
                    <div style={{ fontSize: '9px', color: '#888' }}>
                      {status.performance.activeOptimizations.join(', ')}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Health Tab */}
            {activeTab === 'health' && (
              <div>
                {status.containerHealth ? (
                  <div>
                    <div style={{ marginBottom: '10px' }}>
                      <div style={{ fontSize: '11px', fontWeight: 'bold', marginBottom: '8px' }}>
                        Container Health
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '10px' }}>
                        <div>
                          <div>Running: {status.containerHealth.isRunning ? '✓' : '✗'}</div>
                          <div>CPU: {status.containerHealth.cpuUsage.toFixed(1)}%</div>
                          <div>Memory: {status.containerHealth.memoryUsage.toFixed(1)}%</div>
                        </div>
                        <div>
                          <div>Network: {status.containerHealth.networkHealthy ? '✓' : '✗'}</div>
                          <div>Disk: {status.containerHealth.diskSpaceGb.toFixed(1)}GB</div>
                          <div>Response: {status.containerHealth.lastResponseMs}ms</div>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div style={{ textAlign: 'center', color: '#888', fontSize: '11px' }}>
                    Container health data not available
                  </div>
                )}
              </div>
            )}

            {/* Spawn Dialog */}
            {showSpawnDialog && (
              <div style={{
                position: 'fixed',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                background: 'rgba(0, 0, 0, 0.8)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: 1000,
              }}>
                <div style={{
                  background: 'rgba(0, 0, 0, 0.95)',
                  border: `2px solid ${getSystemStatusColor()}`,
                  borderRadius: '8px',
                  padding: '20px',
                  width: '300px',
                }}>
                  <h4 style={{ margin: '0 0 15px 0', color: getSystemStatusColor() }}>
                    Spawn New Swarm
                  </h4>
                  <input
                    type="text"
                    placeholder="Enter task description..."
                    value={spawnTask}
                    onChange={(e) => setSpawnTask(e.target.value)}
                    style={{
                      ...inputStyle,
                      marginBottom: '15px',
                    }}
                    onKeyPress={(e) => {
                      if (e.key === 'Enter') {
                        handleSpawnSwarm();
                      }
                    }}
                  />
                  <div style={{ display: 'flex', gap: '10px' }}>
                    <button
                      onClick={handleSpawnSwarm}
                      style={buttonStyle}
                      disabled={!spawnTask.trim()}
                    >
                      Spawn
                    </button>
                    <button
                      onClick={() => {
                        setShowSpawnDialog(false);
                        setSpawnTask('');
                      }}
                      style={{
                        ...buttonStyle,
                        borderColor: '#888',
                        color: '#888',
                      }}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </Html>
  );
};

export default HybridSystemDashboard;