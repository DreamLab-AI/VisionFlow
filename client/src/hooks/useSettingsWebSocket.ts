import { useEffect, useRef, useState } from 'react';
import { useSettingsStore } from '../store/settingsStore';
import { createLogger } from '../utils/loggerConfig';

const logger = createLogger('useSettingsWebSocket');

// Mock toast function for notifications
const toast = (options: { title: string; description: string; duration?: number }) => {
  logger.info(`[Toast] ${options.title}: ${options.description}`);
};



interface SettingsBroadcastMessage {
  type: 'SettingChanged' | 'SettingsBatchChanged' | 'SettingsReloaded' | 'PresetApplied' | 'Ping' | 'Pong';
  key?: string;
  value?: unknown;
  changes?: Array<{ key: string; value: unknown }>;
  timestamp: number;
  reason?: string;
  preset_id?: string;
  settings_count?: number;
}

interface UseSettingsWebSocketOptions {
  
  enabled?: boolean;
  
  autoReconnect?: boolean;
  
  reconnectDelay?: number;
  
  showNotifications?: boolean;
}

interface UseSettingsWebSocketReturn {
  
  connected: boolean;
  
  lastUpdate: Date | null;
  
  messageCount: number;
  
  reconnect: () => void;
  
  disconnect: () => void;
}

export const useSettingsWebSocket = (
  options: UseSettingsWebSocketOptions = {}
): UseSettingsWebSocketReturn => {
  const {
    enabled = true,
    autoReconnect = true,
    reconnectDelay = 3000,
    showNotifications = true
  } = options;

  // Use the store's set and batchUpdate methods
  const setByPath = useSettingsStore(state => state.setByPath);
  const batchUpdate = useSettingsStore(state => state.batchUpdate);

  const [connected, setConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [messageCount, setMessageCount] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);

  const connect = () => {
    if (!enabled) return;

    
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws/settings`;

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        logger.info('[SettingsWS] Connected to settings WebSocket');
        setConnected(true);
        reconnectAttemptsRef.current = 0;

        if (showNotifications) {
          toast({
            title: 'Settings Sync',
            description: 'Real-time settings synchronization connected',
            duration: 2000,
          });
        }
      };

      ws.onmessage = (event) => {
        try {
          const message: SettingsBroadcastMessage = JSON.parse(event.data);
          handleMessage(message);
          setLastUpdate(new Date());
          setMessageCount(prev => prev + 1);
        } catch (error) {
          logger.error('[SettingsWS] Failed to parse message:', error);
        }
      };

      ws.onerror = (error) => {
        logger.error('[SettingsWS] WebSocket error:', error);
      };

      ws.onclose = () => {
        logger.info('[SettingsWS] Connection closed');
        setConnected(false);
        wsRef.current = null;

        
        if (autoReconnect && enabled) {
          reconnectAttemptsRef.current += 1;
          const delay = Math.min(reconnectDelay * reconnectAttemptsRef.current, 30000);

          logger.warn(`[SettingsWS] Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current})`);

          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, delay);
        }
      };
    } catch (error) {
      logger.error('[SettingsWS] Failed to connect:', error);
      setConnected(false);
    }
  };

  // Allowlist of setting paths that may be mutated via WebSocket
  const ALLOWED_SETTING_PATHS = new Set([
    'visualisation.nodes',
    'visualisation.edges',
    'visualisation.labels',
    'visualisation.physics',
    'visualisation.rendering',
    'visualisation.glow',
    'visualisation.animations',
    'visualisation.sceneEffects',
    'nodeFilter',
  ]);

  const handleMessage = (message: SettingsBroadcastMessage) => {
    switch (message.type) {
      case 'SettingChanged':
        if (message.key && message.value !== undefined) {
          const basePath = message.key.split('.').slice(0, 2).join('.');
          if (ALLOWED_SETTING_PATHS.has(basePath)) {
            setByPath(message.key as any, message.value);

            if (showNotifications) {
              toast({
                title: 'Setting Updated',
                description: `${message.key} changed`,
                duration: 1500,
              });
            }
          } else {
            logger.warn(`Blocked unauthorized setting change: ${message.key}`);
          }
        }
        break;

      case 'SettingsBatchChanged':
        if (message.changes && message.changes.length > 0) {
          const allowedChanges = message.changes.filter(change => {
            const basePath = change.key.split('.').slice(0, 2).join('.');
            if (!ALLOWED_SETTING_PATHS.has(basePath)) {
              logger.warn(`Blocked unauthorized setting change: ${change.key}`);
              return false;
            }
            return true;
          });

          if (allowedChanges.length > 0) {

            const updates = allowedChanges.map(change => ({
              path: change.key as any,
              value: change.value
            }));

            batchUpdate(updates);

            if (showNotifications) {
              toast({
                title: 'Settings Updated',
                description: `${allowedChanges.length} settings synchronized`,
                duration: 2000,
              });
            }
          }
        }
        break;

      case 'SettingsReloaded':
        logger.warn('Server requested page reload - ignoring for security');
        break;

      case 'PresetApplied':

        if (showNotifications) {
          toast({
            title: 'Preset Applied',
            description: `${message.preset_id} preset with ${message.settings_count} settings`,
            duration: 2000,
          });
        }
        break;

      case 'Ping':
        
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({
            type: 'Pong',
            timestamp: Date.now()
          }));
        }
        break;

      case 'Pong':
        
        break;

      default:
        logger.warn('[SettingsWS] Unknown message type:', message.type);
    }
  };

  const disconnect = () => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setConnected(false);
  };

  const reconnect = () => {
    disconnect();
    setTimeout(() => connect(), 100);
  };

  
  useEffect(() => {
    if (enabled) {
      connect();
    }

    
    return () => {
      disconnect();
    };
  }, [enabled]); 

  return {
    connected,
    lastUpdate,
    messageCount,
    reconnect,
    disconnect
  };
};
