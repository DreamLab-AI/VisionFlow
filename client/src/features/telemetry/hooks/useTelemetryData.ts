import { useState, useEffect, useCallback, useRef } from 'react';
import { createLogger } from '../../../utils/logger';
import {
  AgentLifecycleEvent,
  AgentPosition,
  MCPBridgeStatus,
  GPUMetrics,
  PerformanceMetrics,
  LogEntry,
  AgentCommunication,
  TelemetryFilters
} from '../types';

const logger = createLogger('useTelemetryData');

interface TelemetryData {
  lifecycleEvents: AgentLifecycleEvent[];
  agentPositions: Record<string, AgentPosition>;
  mcpBridgeStatus: MCPBridgeStatus[];
  gpuMetrics: GPUMetrics[];
  performanceMetrics: PerformanceMetrics[];
  logEntries: LogEntry[];
  communications: AgentCommunication[];
}

interface TelemetryDataHook {
  data: TelemetryData;
  isConnected: boolean;
  error: Error | null;
  filters: TelemetryFilters;
  setFilters: (filters: Partial<TelemetryFilters>) => void;
  clearData: () => void;
  exportData: (format: 'json' | 'csv') => void;
}

export function useTelemetryData(): TelemetryDataHook {
  const [data, setData] = useState<TelemetryData>({
    lifecycleEvents: [],
    agentPositions: {},
    mcpBridgeStatus: [],
    gpuMetrics: [],
    performanceMetrics: [],
    logEntries: [],
    communications: []
  });

  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [filters, setFiltersState] = useState<TelemetryFilters>({});

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  const setFilters = useCallback((newFilters: Partial<TelemetryFilters>) => {
    setFiltersState(prev => ({ ...prev, ...newFilters }));
  }, []);

  const clearData = useCallback(() => {
    setData({
      lifecycleEvents: [],
      agentPositions: {},
      mcpBridgeStatus: [],
      gpuMetrics: [],
      performanceMetrics: [],
      logEntries: [],
      communications: []
    });
  }, []);

  const exportData = useCallback((format: 'json' | 'csv') => {
    try {
      const timestamp = new Date().toISOString();
      const filename = `telemetry-data-${timestamp}.${format}`;

      if (format === 'json') {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      } else if (format === 'csv') {
        // Convert to CSV format - implementation would depend on specific requirements
        const csvData = convertToCSV(data);
        const blob = new Blob([csvData], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }

      logger.info(`Telemetry data exported as ${format}`);
    } catch (err) {
      logger.error('Failed to export telemetry data:', err);
      setError(err instanceof Error ? err : new Error('Export failed'));
    }
  }, [data]);

  const connectWebSocket = useCallback(() => {
    try {
      // Determine WebSocket URL based on current location
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${wsProtocol}//${window.location.host}/ws/telemetry`;

      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        logger.info('Telemetry WebSocket connected');
        setIsConnected(true);
        setError(null);
        reconnectAttempts.current = 0;
      };

      wsRef.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);

          setData(prevData => {
            const newData = { ...prevData };

            switch (message.type) {
              case 'lifecycle_event':
                newData.lifecycleEvents = [
                  ...prevData.lifecycleEvents.slice(-999),
                  {
                    ...message.data,
                    timestamp: new Date(message.data.timestamp)
                  }
                ];
                break;

              case 'agent_position':
                newData.agentPositions = {
                  ...prevData.agentPositions,
                  [message.data.agentId]: {
                    ...message.data,
                    timestamp: new Date(message.data.timestamp)
                  }
                };
                break;

              case 'mcp_bridge_status':
                const bridgeIndex = prevData.mcpBridgeStatus.findIndex(
                  b => b.bridgeId === message.data.bridgeId
                );
                if (bridgeIndex >= 0) {
                  newData.mcpBridgeStatus = [...prevData.mcpBridgeStatus];
                  newData.mcpBridgeStatus[bridgeIndex] = {
                    ...message.data,
                    lastHeartbeat: new Date(message.data.lastHeartbeat)
                  };
                } else {
                  newData.mcpBridgeStatus = [
                    ...prevData.mcpBridgeStatus,
                    {
                      ...message.data,
                      lastHeartbeat: new Date(message.data.lastHeartbeat)
                    }
                  ];
                }
                break;

              case 'gpu_metrics':
                newData.gpuMetrics = [
                  ...prevData.gpuMetrics.slice(-99),
                  {
                    ...message.data,
                    timestamp: new Date(message.data.timestamp)
                  }
                ];
                break;

              case 'performance_metrics':
                newData.performanceMetrics = [
                  ...prevData.performanceMetrics.slice(-99),
                  {
                    ...message.data,
                    timestamp: new Date(message.data.timestamp)
                  }
                ];
                break;

              case 'log_entry':
                newData.logEntries = [
                  ...prevData.logEntries.slice(-4999),
                  {
                    ...message.data,
                    timestamp: new Date(message.data.timestamp)
                  }
                ];
                break;

              case 'agent_communication':
                newData.communications = [
                  ...prevData.communications.slice(-999),
                  {
                    ...message.data,
                    timestamp: new Date(message.data.timestamp)
                  }
                ];
                break;

              default:
                logger.debug('Unknown telemetry message type:', message.type);
            }

            return newData;
          });
        } catch (err) {
          logger.error('Failed to parse telemetry message:', err);
        }
      };

      wsRef.current.onclose = () => {
        logger.info('Telemetry WebSocket disconnected');
        setIsConnected(false);

        // Attempt to reconnect
        if (reconnectAttempts.current < maxReconnectAttempts) {
          reconnectAttempts.current++;
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);

          reconnectTimeoutRef.current = setTimeout(() => {
            logger.info(`Attempting to reconnect (${reconnectAttempts.current}/${maxReconnectAttempts})...`);
            connectWebSocket();
          }, delay);
        } else {
          setError(new Error('Failed to reconnect to telemetry service'));
        }
      };

      wsRef.current.onerror = (error) => {
        logger.error('Telemetry WebSocket error:', error);
        setError(new Error('WebSocket connection error'));
      };

    } catch (err) {
      logger.error('Failed to establish telemetry WebSocket connection:', err);
      setError(err instanceof Error ? err : new Error('Connection failed'));
    }
  }, []);

  useEffect(() => {
    connectWebSocket();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connectWebSocket]);

  return {
    data,
    isConnected,
    error,
    filters,
    setFilters,
    clearData,
    exportData
  };
}

// Helper function to convert data to CSV
function convertToCSV(data: TelemetryData): string {
  const rows: string[] = [];

  // Add headers
  rows.push('Type,Timestamp,AgentId,Data');

  // Add lifecycle events
  data.lifecycleEvents.forEach(event => {
    rows.push(`LifecycleEvent,${event.timestamp.toISOString()},${event.agentId},"${JSON.stringify(event).replace(/"/g, '""')}"`);
  });

  // Add log entries
  data.logEntries.forEach(entry => {
    rows.push(`LogEntry,${entry.timestamp.toISOString()},${entry.agentId || ''},"${JSON.stringify(entry).replace(/"/g, '""')}"`);
  });

  return rows.join('\n');
}