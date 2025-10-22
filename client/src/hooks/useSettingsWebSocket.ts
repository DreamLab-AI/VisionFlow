import { useEffect, useRef, useState } from 'react';
import { useSettingsStore } from '@/stores/settingsStore';
import { toast } from '@/hooks/use-toast';

/**
 * Real-time Settings WebSocket Hook
 *
 * Features:
 * - Automatic connection and reconnection
 * - Heartbeat monitoring
 * - Real-time settings synchronization
 * - Batch update handling
 * - Preset application notifications
 * - Hot-reload notifications
 *
 * Usage:
 * ```typescript
 * const MyComponent = () => {
 *   const { connected, lastUpdate } = useSettingsWebSocket();
 *
 *   return (
 *     <div>
 *       Status: {connected ? 'Connected' : 'Disconnected'}
 *       Last update: {lastUpdate?.toLocaleTimeString()}
 *     </div>
 *   );
 * };
 * ```
 */

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
  /** Enable WebSocket connection */
  enabled?: boolean;
  /** Auto-reconnect on disconnect */
  autoReconnect?: boolean;
  /** Reconnection delay in ms */
  reconnectDelay?: number;
  /** Show toast notifications for changes */
  showNotifications?: boolean;
}

interface UseSettingsWebSocketReturn {
  /** Connection status */
  connected: boolean;
  /** Last message timestamp */
  lastUpdate: Date | null;
  /** Number of messages received */
  messageCount: number;
  /** Manually reconnect */
  reconnect: () => void;
  /** Manually disconnect */
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

  const { updateSetting, bulkUpdateSettings } = useSettingsStore();

  const [connected, setConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [messageCount, setMessageCount] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);

  const connect = () => {
    if (!enabled) return;

    // Determine WebSocket URL
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/api/settings/ws`;

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('[SettingsWS] Connected to settings WebSocket');
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
          console.error('[SettingsWS] Failed to parse message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('[SettingsWS] WebSocket error:', error);
      };

      ws.onclose = () => {
        console.log('[SettingsWS] Connection closed');
        setConnected(false);
        wsRef.current = null;

        // Auto-reconnect
        if (autoReconnect && enabled) {
          reconnectAttemptsRef.current += 1;
          const delay = Math.min(reconnectDelay * reconnectAttemptsRef.current, 30000);

          console.log(`[SettingsWS] Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current})`);

          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, delay);
        }
      };
    } catch (error) {
      console.error('[SettingsWS] Failed to connect:', error);
      setConnected(false);
    }
  };

  const handleMessage = (message: SettingsBroadcastMessage) => {
    switch (message.type) {
      case 'SettingChanged':
        if (message.key && message.value !== undefined) {
          console.log(`[SettingsWS] Setting changed: ${message.key}`);
          updateSetting(message.key, message.value);

          if (showNotifications) {
            toast({
              title: 'Setting Updated',
              description: `${message.key} changed`,
              duration: 1500,
            });
          }
        }
        break;

      case 'SettingsBatchChanged':
        if (message.changes && message.changes.length > 0) {
          console.log(`[SettingsWS] Batch update: ${message.changes.length} settings`);

          const updates: Record<string, unknown> = {};
          message.changes.forEach(change => {
            updates[change.key] = change.value;
          });

          bulkUpdateSettings(updates);

          if (showNotifications) {
            toast({
              title: 'Settings Updated',
              description: `${message.changes.length} settings synchronized`,
              duration: 2000,
            });
          }
        }
        break;

      case 'SettingsReloaded':
        console.log(`[SettingsWS] Settings reloaded: ${message.reason}`);

        if (showNotifications) {
          toast({
            title: 'Settings Reloaded',
            description: message.reason || 'Configuration updated',
            duration: 3000,
          });
        }

        // Trigger full settings refresh
        window.location.reload();
        break;

      case 'PresetApplied':
        console.log(`[SettingsWS] Preset applied: ${message.preset_id}`);

        if (showNotifications) {
          toast({
            title: 'Preset Applied',
            description: `${message.preset_id} preset with ${message.settings_count} settings`,
            duration: 2000,
          });
        }
        break;

      case 'Ping':
        // Respond to server ping with pong
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({
            type: 'Pong',
            timestamp: Date.now()
          }));
        }
        break;

      case 'Pong':
        // Server acknowledged our ping
        break;

      default:
        console.warn('[SettingsWS] Unknown message type:', message.type);
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

  // Initialize connection
  useEffect(() => {
    if (enabled) {
      connect();
    }

    // Cleanup on unmount
    return () => {
      disconnect();
    };
  }, [enabled]); // Only depend on enabled to avoid reconnections

  return {
    connected,
    lastUpdate,
    messageCount,
    reconnect,
    disconnect
  };
};

/**
 * Settings WebSocket Status Component
 *
 * Display connection status indicator
 */
export const SettingsWebSocketStatus: React.FC<{
  className?: string;
}> = ({ className }) => {
  const { connected, lastUpdate, messageCount } = useSettingsWebSocket();

  return (
    <div className={`flex items-center gap-2 text-xs ${className}`}>
      <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`} />
      <span className="text-muted-foreground">
        {connected ? 'Live' : 'Offline'}
      </span>
      {lastUpdate && (
        <span className="text-muted-foreground">
          • {messageCount} updates
        </span>
      )}
    </div>
  );
};
