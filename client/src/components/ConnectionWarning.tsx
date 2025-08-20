import React, { useState, useEffect } from 'react';
import { createLogger } from '../utils/logger';
import WebSocketService from '../services/WebSocketService';
import { useSettingsStore } from '../store/settingsStore';
import AlertCircle from 'lucide-react/dist/esm/icons/alert-circle';
import WifiOff from 'lucide-react/dist/esm/icons/wifi-off';
import RefreshCw from 'lucide-react/dist/esm/icons/refresh-cw';

const logger = createLogger('ConnectionWarning');

/**
 * Displays a prominent warning when the application fails to connect to the backend
 * Indicates when running on cached/local settings only
 */
export const ConnectionWarning: React.FC = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [settingsSource, setSettingsSource] = useState<'server' | 'local'>('server');
  const [isReconnecting, setIsReconnecting] = useState(false);
  const { settings } = useSettingsStore();

  useEffect(() => {
    // Check WebSocket connection status
    const wsService = WebSocketService.getInstance();
    
    const handleConnectionChange = (connected: boolean) => {
      setIsConnected(connected);
      if (!connected) {
        logger.warn('Lost connection to backend server');
      }
    };

    // Subscribe to connection status changes
    const unsubscribe = wsService.onConnectionStatusChange(handleConnectionChange);
    
    // Check initial connection state
    setIsConnected(wsService.isReady());

    // Check if settings came from local storage
    const checkSettingsSource = () => {
      const localStorageSettings = localStorage.getItem('settings');
      if (localStorageSettings && !isConnected) {
        setSettingsSource('local');
        logger.warn('Using cached settings from local storage - server settings unavailable');
      }
    };
    
    checkSettingsSource();

    return () => {
      if (typeof unsubscribe === 'function') {
        unsubscribe();
      }
    };
  }, []);

  const handleReconnect = async () => {
    setIsReconnecting(true);
    try {
      logger.info('Attempting manual reconnection...');
      const wsService = WebSocketService.getInstance();
      await wsService.connect();
      
      // Try to reload settings from server
      const { initialize } = useSettingsStore.getState();
      await initialize();
      
      logger.info('Reconnection successful');
    } catch (error) {
      logger.error('Manual reconnection failed:', error);
    } finally {
      setIsReconnecting(false);
    }
  };

  // Don't show warning if connected and using server settings
  if (isConnected && settingsSource === 'server') {
    return null;
  }

  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-gradient-to-r from-orange-600 to-red-600 text-white px-4 py-3 shadow-lg">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <div className="flex items-center space-x-3">
          {!isConnected ? (
            <WifiOff className="h-5 w-5 animate-pulse" />
          ) : (
            <AlertCircle className="h-5 w-5" />
          )}
          
          <div className="flex flex-col">
            <div className="font-semibold text-sm">
              {!isConnected 
                ? 'Connection to Backend Failed' 
                : 'Using Cached Settings'}
            </div>
            <div className="text-xs opacity-90">
              {!isConnected 
                ? 'Running in offline mode with cached settings. Real-time features disabled.'
                : 'Unable to fetch server configuration. Using local storage fallback.'}
            </div>
          </div>
        </div>

        <button
          onClick={handleReconnect}
          disabled={isReconnecting}
          className="flex items-center space-x-2 px-3 py-1.5 bg-white/20 hover:bg-white/30 rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          title="Attempt to reconnect to server"
        >
          <RefreshCw className={`h-4 w-4 ${isReconnecting ? 'animate-spin' : ''}`} />
          <span className="text-sm font-medium">
            {isReconnecting ? 'Reconnecting...' : 'Retry'}
          </span>
        </button>
      </div>

      {/* Additional details in debug mode */}
      {settings.system?.debug?.enabled && (
        <div className="max-w-7xl mx-auto mt-2 text-xs opacity-75 font-mono">
          <div>Settings Source: {settingsSource === 'local' ? 'localStorage' : 'server'}</div>
          <div>WebSocket: {isConnected ? 'connected' : 'disconnected'}</div>
          <div>API Endpoint: /api/settings - {settingsSource === 'server' ? 'OK' : 'FAILED'}</div>
        </div>
      )}
    </div>
  );
};

export default ConnectionWarning;