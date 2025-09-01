import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../../design-system/components/Card';
import { Label } from '../../design-system/components/Label';
import { Switch } from '../../design-system/components/Switch';
import { Input } from '../../design-system/components/Input';
import { Button } from '../../design-system/components/Button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../../design-system/components/Select';
import { Badge } from '../../design-system/components/Badge';
import { Wifi, WifiOff, Globe, Shield, Clock, AlertCircle, CheckCircle2 } from 'lucide-react';
import { useSelectiveSetting, useSettingSetter } from '../../../hooks/useSelectiveSettingsStore';

/**
 * NetworkSettings Settings Panel
 * Provides network and connection settings with selective access patterns
 */
export function NetworkSettings() {
  const { set } = useSettingSetter();
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'connecting' | 'disconnected'>('disconnected');
  const [latency, setLatency] = useState<number | null>(null);
  
  // Use selective settings access for network-related settings
  const backendUrl = useSelectiveSetting<string>('system.customBackendUrl') ?? '';
  const autoReconnect = useSelectiveSetting<boolean>('system.websocket.autoReconnect') ?? true;
  const reconnectAttempts = useSelectiveSetting<number>('system.websocket.reconnectAttempts') ?? 5;
  const reconnectInterval = useSelectiveSetting<number>('system.websocket.reconnectInterval') ?? 5000;
  const heartbeatInterval = useSelectiveSetting<number>('system.websocket.heartbeatInterval') ?? 30000;
  const timeout = useSelectiveSetting<number>('system.websocket.timeout') ?? 10000;
  const bufferSize = useSelectiveSetting<number>('system.websocket.bufferSize') ?? 1024;
  
  // Security settings
  const sslEnabled = useSelectiveSetting<boolean>('system.network.ssl.enabled') ?? true;
  const sslVerify = useSelectiveSetting<boolean>('system.network.ssl.verify') ?? true;
  const corsEnabled = useSelectiveSetting<boolean>('system.network.cors.enabled') ?? true;
  
  // API settings
  const apiVersion = useSelectiveSetting<string>('system.api.version') ?? 'v1';
  const rateLimitEnabled = useSelectiveSetting<boolean>('system.api.rateLimit.enabled') ?? true;
  const maxRequestsPerMinute = useSelectiveSetting<number>('system.api.rateLimit.maxRequests') ?? 60;
  
  // Proxy settings
  const proxyEnabled = useSelectiveSetting<boolean>('system.network.proxy.enabled') ?? false;
  const proxyHost = useSelectiveSetting<string>('system.network.proxy.host') ?? '';
  const proxyPort = useSelectiveSetting<number>('system.network.proxy.port') ?? 8080;
  const proxyAuth = useSelectiveSetting<boolean>('system.network.proxy.auth') ?? false;
  const proxyUsername = useSelectiveSetting<string>('system.network.proxy.username') ?? '';
  const proxyPassword = useSelectiveSetting<string>('system.network.proxy.password') ?? '';

  const handleSettingChange = async (path: string, value: any) => {
    await set(path, value);
  };

  const testConnection = async () => {
    setConnectionStatus('connecting');
    const startTime = Date.now();
    
    try {
      const url = backendUrl || window.location.origin;
      const response = await fetch(`${url}/api/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        signal: AbortSignal.timeout(timeout),
      });
      
      if (response.ok) {
        setLatency(Date.now() - startTime);
        setConnectionStatus('connected');
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      setConnectionStatus('disconnected');
      setLatency(null);
      console.error('Connection test failed:', error);
    }
  };

  const resetNetworkSettings = async () => {
    await Promise.all([
      set('system.customBackendUrl', ''),
      set('system.websocket.autoReconnect', true),
      set('system.websocket.reconnectAttempts', 5),
      set('system.websocket.reconnectInterval', 5000),
      set('system.websocket.heartbeatInterval', 30000),
      set('system.websocket.timeout', 10000),
      set('system.websocket.bufferSize', 1024),
      set('system.network.ssl.enabled', true),
      set('system.network.ssl.verify', true),
      set('system.network.cors.enabled', true),
      set('system.api.version', 'v1'),
      set('system.api.rateLimit.enabled', true),
      set('system.api.rateLimit.maxRequests', 60),
      set('system.network.proxy.enabled', false),
      set('system.network.proxy.host', ''),
      set('system.network.proxy.port', 8080),
      set('system.network.proxy.auth', false),
      set('system.network.proxy.username', ''),
      set('system.network.proxy.password', '')
    ]);
  };

  // Test connection on mount
  useEffect(() => {
    testConnection();
  }, [backendUrl]);

  const getConnectionBadge = () => {
    switch (connectionStatus) {
      case 'connected':
        return (
          <Badge variant="outline" className="text-green-700 border-green-200">
            <CheckCircle2 className="w-3 h-3 mr-1" />
            Connected
            {latency && <span className="ml-1">({latency}ms)</span>}
          </Badge>
        );
      case 'connecting':
        return (
          <Badge variant="outline" className="text-yellow-700 border-yellow-200">
            <Clock className="w-3 h-3 mr-1 animate-spin" />
            Connecting...
          </Badge>
        );
      case 'disconnected':
        return (
          <Badge variant="outline" className="text-red-700 border-red-200">
            <AlertCircle className="w-3 h-3 mr-1" />
            Disconnected
          </Badge>
        );
    }
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Globe className="w-5 h-5" />
              <CardTitle>Network Settings</CardTitle>
            </div>
            {getConnectionBadge()}
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Backend Connection */}
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <Wifi className="w-4 h-4" />
              <h3 className="text-sm font-medium">Backend Connection</h3>
            </div>
            
            <div className="space-y-3">
              <div className="space-y-2">
                <Label className="text-sm">Backend URL</Label>
                <div className="flex gap-2">
                  <Input
                    type="url"
                    value={backendUrl}
                    onChange={(e) => handleSettingChange('system.customBackendUrl', e.target.value)}
                    placeholder="Leave empty for default (current domain)"
                    className="flex-1"
                  />
                  <Button
                    variant="outline"
                    onClick={testConnection}
                    disabled={connectionStatus === 'connecting'}
                  >
                    Test
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground">
                  Specify a custom backend URL (e.g., http://localhost:8000) or leave empty to use the current domain
                </p>
              </div>
            </div>
          </div>

          {/* WebSocket Settings */}
          <div className="space-y-4">
            <div className="border-t pt-4">
              <div className="flex items-center gap-2 mb-3">
                <WifiOff className="w-4 h-4" />
                <h3 className="text-sm font-medium">WebSocket Settings</h3>
              </div>
              
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label className="text-sm">Auto Reconnect</Label>
                    <p className="text-xs text-muted-foreground">
                      Automatically reconnect when connection is lost
                    </p>
                  </div>
                  <Switch
                    checked={autoReconnect}
                    onCheckedChange={(checked) => handleSettingChange('system.websocket.autoReconnect', checked)}
                  />
                </div>

                {autoReconnect && (
                  <div className="grid grid-cols-2 gap-4 pl-4 border-l-2 border-muted">
                    <div className="space-y-2">
                      <Label className="text-sm">Reconnect Attempts</Label>
                      <Input
                        type="number"
                        min="1"
                        max="20"
                        value={reconnectAttempts}
                        onChange={(e) => handleSettingChange('system.websocket.reconnectAttempts', parseInt(e.target.value))}
                        className="w-20"
                      />
                    </div>

                    <div className="space-y-2">
                      <Label className="text-sm">Reconnect Interval</Label>
                      <div className="flex items-center gap-2">
                        <Input
                          type="number"
                          min="1000"
                          max="60000"
                          step="1000"
                          value={reconnectInterval}
                          onChange={(e) => handleSettingChange('system.websocket.reconnectInterval', parseInt(e.target.value))}
                          className="w-24"
                        />
                        <span className="text-xs text-muted-foreground">ms</span>
                      </div>
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label className="text-sm">Heartbeat Interval</Label>
                    <div className="flex items-center gap-2">
                      <Input
                        type="number"
                        min="5000"
                        max="120000"
                        step="5000"
                        value={heartbeatInterval}
                        onChange={(e) => handleSettingChange('system.websocket.heartbeatInterval', parseInt(e.target.value))}
                        className="w-24"
                      />
                      <span className="text-xs text-muted-foreground">ms</span>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label className="text-sm">Connection Timeout</Label>
                    <div className="flex items-center gap-2">
                      <Input
                        type="number"
                        min="5000"
                        max="60000"
                        step="1000"
                        value={timeout}
                        onChange={(e) => handleSettingChange('system.websocket.timeout', parseInt(e.target.value))}
                        className="w-24"
                      />
                      <span className="text-xs text-muted-foreground">ms</span>
                    </div>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label className="text-sm">Buffer Size</Label>
                  <div className="flex items-center gap-2">
                    <Input
                      type="number"
                      min="512"
                      max="8192"
                      step="512"
                      value={bufferSize}
                      onChange={(e) => handleSettingChange('system.websocket.bufferSize', parseInt(e.target.value))}
                      className="w-24"
                    />
                    <span className="text-xs text-muted-foreground">bytes</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Security Settings */}
          <div className="space-y-4">
            <div className="border-t pt-4">
              <div className="flex items-center gap-2 mb-3">
                <Shield className="w-4 h-4" />
                <h3 className="text-sm font-medium">Security Settings</h3>
              </div>
              
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label className="text-sm">SSL/TLS Enabled</Label>
                      <p className="text-xs text-muted-foreground">Use secure connections</p>
                    </div>
                    <Switch
                      checked={sslEnabled}
                      onCheckedChange={(checked) => handleSettingChange('system.network.ssl.enabled', checked)}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label className="text-sm">SSL Verification</Label>
                      <p className="text-xs text-muted-foreground">Verify SSL certificates</p>
                    </div>
                    <Switch
                      checked={sslVerify}
                      onCheckedChange={(checked) => handleSettingChange('system.network.ssl.verify', checked)}
                      disabled={!sslEnabled}
                    />
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label className="text-sm">CORS Enabled</Label>
                    <p className="text-xs text-muted-foreground">
                      Enable Cross-Origin Resource Sharing
                    </p>
                  </div>
                  <Switch
                    checked={corsEnabled}
                    onCheckedChange={(checked) => handleSettingChange('system.network.cors.enabled', checked)}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* API Settings */}
          <div className="space-y-4">
            <div className="border-t pt-4">
              <h3 className="text-sm font-medium mb-3">API Settings</h3>
              
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label className="text-sm">API Version</Label>
                    <Select
                      value={apiVersion}
                      onValueChange={(value) => handleSettingChange('system.api.version', value)}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select version" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="v1">v1</SelectItem>
                        <SelectItem value="v2">v2 (Beta)</SelectItem>
                        <SelectItem value="v3">v3 (Alpha)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label className="text-sm">Rate Limiting</Label>
                      <Switch
                        checked={rateLimitEnabled}
                        onCheckedChange={(checked) => handleSettingChange('system.api.rateLimit.enabled', checked)}
                      />
                    </div>
                    {rateLimitEnabled && (
                      <div className="flex items-center gap-2">
                        <Input
                          type="number"
                          min="10"
                          max="1000"
                          value={maxRequestsPerMinute}
                          onChange={(e) => handleSettingChange('system.api.rateLimit.maxRequests', parseInt(e.target.value))}
                          className="w-20"
                        />
                        <span className="text-xs text-muted-foreground">req/min</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Proxy Settings */}
          <div className="space-y-4">
            <div className="border-t pt-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium">Proxy Settings</h3>
                <Switch
                  checked={proxyEnabled}
                  onCheckedChange={(checked) => handleSettingChange('system.network.proxy.enabled', checked)}
                />
              </div>
              
              {proxyEnabled && (
                <div className="space-y-4 pl-4 border-l-2 border-muted">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label className="text-sm">Host</Label>
                      <Input
                        type="text"
                        value={proxyHost}
                        onChange={(e) => handleSettingChange('system.network.proxy.host', e.target.value)}
                        placeholder="proxy.example.com"
                      />
                    </div>

                    <div className="space-y-2">
                      <Label className="text-sm">Port</Label>
                      <Input
                        type="number"
                        min="1"
                        max="65535"
                        value={proxyPort}
                        onChange={(e) => handleSettingChange('system.network.proxy.port', parseInt(e.target.value))}
                      />
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <Label className="text-sm">Authentication Required</Label>
                      <Switch
                        checked={proxyAuth}
                        onCheckedChange={(checked) => handleSettingChange('system.network.proxy.auth', checked)}
                      />
                    </div>

                    {proxyAuth && (
                      <div className="grid grid-cols-2 gap-4 pl-4 border-l-2 border-muted">
                        <div className="space-y-2">
                          <Label className="text-sm">Username</Label>
                          <Input
                            type="text"
                            value={proxyUsername}
                            onChange={(e) => handleSettingChange('system.network.proxy.username', e.target.value)}
                          />
                        </div>

                        <div className="space-y-2">
                          <Label className="text-sm">Password</Label>
                          <Input
                            type="password"
                            value={proxyPassword}
                            onChange={(e) => handleSettingChange('system.network.proxy.password', e.target.value)}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Reset Button */}
          <div className="border-t pt-4">
            <Button
              variant="outline"
              onClick={resetNetworkSettings}
              className="w-full"
            >
              Reset Network Settings
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default NetworkSettings;