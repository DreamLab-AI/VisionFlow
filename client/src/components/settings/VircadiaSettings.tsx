

import React, { useState, useEffect } from 'react';
import { useVircadia } from '../../contexts/VircadiaContext';
import { useVircadiaBridges } from '../../contexts/VircadiaBridgesContext';
import { useSettingsStore } from '../../store/settingsStore';
import { createLogger } from '../../utils/loggerConfig';

const logger = createLogger('VircadiaSettings');

export const VircadiaSettings: React.FC = () => {
  const { client, connect, disconnect, connectionInfo, isConnected, isConnecting, error } = useVircadia();
  const { isInitialized: bridgesInitialized, activeUsers, error: bridgeError } = useVircadiaBridges();

  const [vircadiaEnabled, setVircadiaEnabled] = useState(false);
  const [serverUrl, setServerUrl] = useState('ws://vircadia-world-server:3020/world/ws');
  const [autoConnect, setAutoConnect] = useState(false);

  
  useEffect(() => {
    const settings = useSettingsStore.getState().settings;
    const vircadiaConfig = settings?.vircadia;

    if (vircadiaConfig) {
      setVircadiaEnabled(vircadiaConfig.enabled || false);
      setServerUrl(vircadiaConfig.serverUrl || serverUrl);
      setAutoConnect(vircadiaConfig.autoConnect || false);
    }
  }, []);

  
  const saveSettings = () => {
    const currentSettings = useSettingsStore.getState().settings;
    useSettingsStore.getState().updateSettings({
      ...currentSettings,
      vircadia: {
        enabled: vircadiaEnabled,
        serverUrl,
        autoConnect
      }
    });
  };

  
  const handleEnableToggle = async () => {
    const newEnabled = !vircadiaEnabled;
    setVircadiaEnabled(newEnabled);

    if (newEnabled) {
      try {
        await connect();
        logger.info('Vircadia enabled and connected');
      } catch (err) {
        logger.error('Failed to connect to Vircadia:', err);
      }
    } else {
      disconnect();
      logger.info('Vircadia disabled');
    }

    saveSettings();
  };

  
  const handleServerUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setServerUrl(e.target.value);
    saveSettings();
  };

  
  const handleReconnect = async () => {
    disconnect();
    setTimeout(async () => {
      try {
        await connect();
      } catch (err) {
        logger.error('Failed to reconnect:', err);
      }
    }, 1000);
  };

  
  const getStatusColor = () => {
    if (isConnected) return 'text-green-500';
    if (isConnecting) return 'text-yellow-500';
    if (error) return 'text-red-500';
    return 'text-gray-500';
  };

  const getStatusText = () => {
    if (isConnected) return 'Connected';
    if (isConnecting) return 'Connecting...';
    if (error) return 'Error';
    return 'Disconnected';
  };

  return (
    <div className="space-y-6 p-6 bg-gray-900 rounded-lg border border-gray-700">
      <div>
        <h2 className="text-2xl font-bold text-white mb-2">Multi-User XR (Vircadia)</h2>
        <p className="text-gray-400 text-sm">
          Enable collaborative visualization with multiple users in real-time
        </p>
      </div>

      {}
      <div className="flex items-center justify-between">
        <div>
          <label className="text-white font-medium">Enable Multi-User Mode</label>
          <p className="text-gray-400 text-sm">Connect to Vircadia World Server for collaboration</p>
        </div>
        <button
          onClick={handleEnableToggle}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
            vircadiaEnabled ? 'bg-blue-600' : 'bg-gray-600'
          }`}
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
              vircadiaEnabled ? 'translate-x-6' : 'translate-x-1'
            }`}
          />
        </button>
      </div>

      {}
      <div>
        <label className="block text-white font-medium mb-2">Vircadia Server URL</label>
        <input
          type="text"
          value={serverUrl}
          onChange={handleServerUrlChange}
          disabled={!vircadiaEnabled}
          placeholder="ws://vircadia-world-server:3020/world/ws"
          className="w-full px-4 py-2 bg-gray-800 text-white border border-gray-600 rounded focus:outline-none focus:border-blue-500 disabled:opacity-50"
        />
        <p className="text-gray-400 text-xs mt-1">
          Default: ws://vircadia-world-server:3020/world/ws (Docker network)
        </p>
      </div>

      {}
      <div className="p-4 bg-gray-800 rounded border border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-white font-medium">Connection Status</h3>
            <p className={`text-sm ${getStatusColor()}`}>{getStatusText()}</p>
          </div>
          {vircadiaEnabled && !isConnecting && (
            <button
              onClick={handleReconnect}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
            >
              Reconnect
            </button>
          )}
        </div>

        {connectionInfo && (
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">Agent ID:</span>
              <span className="text-white font-mono">{connectionInfo.agentId || 'N/A'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Session ID:</span>
              <span className="text-white font-mono">{connectionInfo.sessionId || 'N/A'}</span>
            </div>
            {connectionInfo.connectionDuration && (
              <div className="flex justify-between">
                <span className="text-gray-400">Connected For:</span>
                <span className="text-white">{Math.floor(connectionInfo.connectionDuration / 1000)}s</span>
              </div>
            )}
          </div>
        )}

        {error && (
          <div className="mt-4 p-3 bg-red-900/20 border border-red-500 rounded">
            <p className="text-red-400 text-sm">Error: {error.message}</p>
          </div>
        )}

        {bridgeError && (
          <div className="mt-4 p-3 bg-orange-900/20 border border-orange-500 rounded">
            <p className="text-orange-400 text-sm">Bridge Error: {bridgeError.message}</p>
          </div>
        )}
      </div>

      {}
      {bridgesInitialized && isConnected && (
        <div className="p-4 bg-gray-800 rounded border border-gray-700">
          <h3 className="text-white font-medium mb-3">Active Users ({activeUsers.length})</h3>
          {activeUsers.length === 0 ? (
            <p className="text-gray-400 text-sm">No other users connected</p>
          ) : (
            <div className="space-y-2">
              {activeUsers.map((user) => (
                <div
                  key={user.userId}
                  className="flex items-center justify-between p-2 bg-gray-700 rounded"
                >
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-white">{user.username}</span>
                  </div>
                  {user.selectedNodes.length > 0 && (
                    <span className="text-gray-400 text-xs">
                      {user.selectedNodes.length} nodes selected
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {}
      {vircadiaEnabled && isConnected && (
        <div className="p-4 bg-gray-800 rounded border border-gray-700">
          <h3 className="text-white font-medium mb-3">Synchronization Status</h3>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-gray-400">Bots Bridge:</span>
              <span className={bridgesInitialized ? 'text-green-500' : 'text-gray-500'}>
                {bridgesInitialized ? '✓ Active' : 'Initializing...'}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-400">Graph Bridge:</span>
              <span className={bridgesInitialized ? 'text-green-500' : 'text-gray-500'}>
                {bridgesInitialized ? '✓ Active' : 'Initializing...'}
              </span>
            </div>
          </div>
        </div>
      )}

      {}
      <div className="p-4 bg-blue-900/20 border border-blue-500 rounded">
        <h4 className="text-blue-400 font-medium mb-2">📘 How to Use Multi-User Mode</h4>
        <ul className="text-gray-300 text-sm space-y-1">
          <li>• Enable toggle to connect to Vircadia server</li>
          <li>• See other users' agent selections in real-time</li>
          <li>• Add annotations visible to all users</li>
          <li>• Hear spatial audio based on proximity</li>
          <li>• Best experienced in VR with Meta Quest 3</li>
        </ul>
      </div>

      {}
      <div className="p-4 bg-gray-800 rounded border border-gray-700">
        <h4 className="text-white font-medium mb-2">🐳 Docker Setup</h4>
        <p className="text-gray-400 text-sm mb-2">To enable Vircadia server, run:</p>
        <code className="block p-2 bg-gray-900 text-green-400 text-xs rounded">
          docker-compose -f docker-compose.yml -f docker-compose.vircadia.yml --profile dev up -d
        </code>
      </div>
    </div>
  );
};
