import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../../design-system/components/Card';
import { Label } from '../../design-system/components/Label';
import { Switch } from '../../design-system/components/Switch';
import { Input } from '../../design-system/components/Input';
import { Button } from '../../design-system/components/Button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../../design-system/components/Select';
import { Badge } from '../../design-system/components/Badge';
import { Textarea } from '../../design-system/components/Textarea';
import { Bug, Terminal, FileText, Download, Trash2, AlertTriangle, Info, CheckCircle2 } from 'lucide-react';
import { useSelectiveSetting, useSettingSetter } from '../../../hooks/useSelectiveSettingsStore';

/**
 * DebugSettings Settings Panel
 * Provides debugging and development settings with selective access patterns
 */
export function DebugSettings() {
  const { set, batchSet } = useSettingSetter();
  const [logs, setLogs] = useState<string[]>([]);
  const [logLevel, setLogLevel] = useState<'debug' | 'info' | 'warn' | 'error'>('info');
  
  // Use selective settings access for debug-related settings
  const debugMode = useSelectiveSetting<boolean>('system.debug.enabled') ?? false;
  const consoleLogging = useSelectiveSetting<boolean>('system.debug.console') ?? true;
  const fileLogging = useSelectiveSetting<boolean>('system.debug.fileLogging') ?? false;
  const networkLogging = useSelectiveSetting<boolean>('system.debug.network') ?? false;
  const performanceMetrics = useSelectiveSetting<boolean>('system.debug.performance') ?? false;
  const verboseMode = useSelectiveSetting<boolean>('system.debug.verbose') ?? false;
  const stackTraces = useSelectiveSetting<boolean>('system.debug.stackTraces') ?? true;
  const devTools = useSelectiveSetting<boolean>('system.debug.devTools') ?? false;
  
  // Logging settings
  const maxLogSize = useSelectiveSetting<number>('system.debug.maxLogSize') ?? 1000;
  const logRetention = useSelectiveSetting<number>('system.debug.logRetention') ?? 7; // days
  const autoExport = useSelectiveSetting<boolean>('system.debug.autoExport') ?? false;
  
  // Error reporting
  const errorReporting = useSelectiveSetting<boolean>('system.debug.errorReporting') ?? false;
  const crashReports = useSelectiveSetting<boolean>('system.debug.crashReports') ?? false;
  const anonymizeErrors = useSelectiveSetting<boolean>('system.debug.anonymizeErrors') ?? true;
  
  // Development features
  const hotReload = useSelectiveSetting<boolean>('system.debug.hotReload') ?? false;
  const sourceMapEnabled = useSelectiveSetting<boolean>('system.debug.sourceMaps') ?? false;
  const profileMode = useSelectiveSetting<boolean>('system.debug.profiling') ?? false;

  const handleSettingChange = async (path: string, value: any) => {
    await set(path, value);
  };

  const handleBatchChange = async (updates: Record<string, any>) => {
    const pathValuePairs = Object.entries(updates).map(([path, value]) => ({
      path,
      value
    }));
    await batchSet(pathValuePairs);
  };

  const clearLogs = () => {
    setLogs([]);
    console.clear();
  };

  const exportLogs = () => {
    const logData = logs.join('\n');
    const blob = new Blob([logData], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `debug-logs-${new Date().toISOString().slice(0, 10)}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const toggleDebugMode = async (enabled: boolean) => {
    await handleBatchChange({
      'system.debug.enabled': enabled,
      'system.debug.console': enabled,
      'system.debug.performance': enabled && performanceMetrics
    });
  };

  const resetDebugSettings = async () => {
    await handleBatchChange({
      'system.debug.enabled': false,
      'system.debug.console': true,
      'system.debug.fileLogging': false,
      'system.debug.network': false,
      'system.debug.performance': false,
      'system.debug.verbose': false,
      'system.debug.stackTraces': true,
      'system.debug.devTools': false,
      'system.debug.maxLogSize': 1000,
      'system.debug.logRetention': 7,
      'system.debug.autoExport': false,
      'system.debug.errorReporting': false,
      'system.debug.crashReports': false,
      'system.debug.anonymizeErrors': true,
      'system.debug.hotReload': false,
      'system.debug.sourceMaps': false,
      'system.debug.profiling': false
    });
  };

  // Simulate log collection
  useEffect(() => {
    if (!debugMode) return;

    const interval = setInterval(() => {
      const logTypes = ['debug', 'info', 'warn', 'error'];
      const messages = [
        'WebSocket connection established',
        'Settings updated successfully',
        'Performance metrics collected',
        'Network request completed',
        'Component rendered',
        'User interaction detected',
        'Cache updated',
        'Debug mode active'
      ];

      const randomType = logTypes[Math.floor(Math.random() * logTypes.length)];
      const randomMessage = messages[Math.floor(Math.random() * messages.length)];
      const timestamp = new Date().toLocaleTimeString();
      const newLog = `[${timestamp}] ${randomType.toUpperCase()}: ${randomMessage}`;

      setLogs(prev => {
        const updated = [...prev, newLog].slice(-maxLogSize);
        return updated;
      });
    }, 2000);

    return () => clearInterval(interval);
  }, [debugMode, maxLogSize]);

  const getLogIcon = (level: string) => {
    switch (level.toLowerCase()) {
      case 'error': return <AlertTriangle className="w-3 h-3 text-red-500" />;
      case 'warn': return <AlertTriangle className="w-3 h-3 text-yellow-500" />;
      case 'info': return <Info className="w-3 h-3 text-blue-500" />;
      case 'debug': return <Bug className="w-3 h-3 text-gray-500" />;
      default: return <CheckCircle2 className="w-3 h-3 text-green-500" />;
    }
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Bug className="w-5 h-5" />
              <CardTitle>Debug Settings</CardTitle>
            </div>
            <Badge variant={debugMode ? 'default' : 'secondary'}>
              {debugMode ? 'Debug Active' : 'Debug Inactive'}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Master Debug Toggle */}
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label className="text-sm font-medium">Enable Debug Mode</Label>
              <p className="text-xs text-muted-foreground">
                Master toggle for all debugging features
              </p>
            </div>
            <Switch
              checked={debugMode}
              onCheckedChange={toggleDebugMode}
            />
          </div>

          {debugMode && (
            <>
              {/* Logging Settings */}
              <div className="space-y-4">
                <div className="border-t pt-4">
                  <div className="flex items-center gap-2 mb-3">
                    <FileText className="w-4 h-4" />
                    <h3 className="text-sm font-medium">Logging</h3>
                  </div>
                  
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="flex items-center justify-between">
                        <div className="space-y-0.5">
                          <Label className="text-sm">Console Logging</Label>
                          <p className="text-xs text-muted-foreground">Log to browser console</p>
                        </div>
                        <Switch
                          checked={consoleLogging}
                          onCheckedChange={(checked) => handleSettingChange('system.debug.console', checked)}
                        />
                      </div>

                      <div className="flex items-center justify-between">
                        <div className="space-y-0.5">
                          <Label className="text-sm">File Logging</Label>
                          <p className="text-xs text-muted-foreground">Save logs to file</p>
                        </div>
                        <Switch
                          checked={fileLogging}
                          onCheckedChange={(checked) => handleSettingChange('system.debug.fileLogging', checked)}
                        />
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div className="flex items-center justify-between">
                        <div className="space-y-0.5">
                          <Label className="text-sm">Network Logging</Label>
                          <p className="text-xs text-muted-foreground">Log API requests</p>
                        </div>
                        <Switch
                          checked={networkLogging}
                          onCheckedChange={(checked) => handleSettingChange('system.debug.network', checked)}
                        />
                      </div>

                      <div className="flex items-center justify-between">
                        <div className="space-y-0.5">
                          <Label className="text-sm">Performance Metrics</Label>
                          <p className="text-xs text-muted-foreground">Track performance data</p>
                        </div>
                        <Switch
                          checked={performanceMetrics}
                          onCheckedChange={(checked) => handleSettingChange('system.debug.performance', checked)}
                        />
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div className="flex items-center justify-between">
                        <div className="space-y-0.5">
                          <Label className="text-sm">Verbose Mode</Label>
                          <p className="text-xs text-muted-foreground">Detailed logging output</p>
                        </div>
                        <Switch
                          checked={verboseMode}
                          onCheckedChange={(checked) => handleSettingChange('system.debug.verbose', checked)}
                        />
                      </div>

                      <div className="flex items-center justify-between">
                        <div className="space-y-0.5">
                          <Label className="text-sm">Stack Traces</Label>
                          <p className="text-xs text-muted-foreground">Include stack traces in errors</p>
                        </div>
                        <Switch
                          checked={stackTraces}
                          onCheckedChange={(checked) => handleSettingChange('system.debug.stackTraces', checked)}
                        />
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label className="text-sm">Max Log Size</Label>
                        <div className="flex items-center gap-2">
                          <Input
                            type="number"
                            min="100"
                            max="10000"
                            value={maxLogSize}
                            onChange={(e) => handleSettingChange('system.debug.maxLogSize', parseInt(e.target.value))}
                            className="w-24"
                          />
                          <span className="text-xs text-muted-foreground">entries</span>
                        </div>
                      </div>

                      <div className="space-y-2">
                        <Label className="text-sm">Log Retention</Label>
                        <div className="flex items-center gap-2">
                          <Input
                            type="number"
                            min="1"
                            max="30"
                            value={logRetention}
                            onChange={(e) => handleSettingChange('system.debug.logRetention', parseInt(e.target.value))}
                            className="w-20"
                          />
                          <span className="text-xs text-muted-foreground">days</span>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label className="text-sm">Auto Export Logs</Label>
                        <p className="text-xs text-muted-foreground">
                          Automatically export logs daily
                        </p>
                      </div>
                      <Switch
                        checked={autoExport}
                        onCheckedChange={(checked) => handleSettingChange('system.debug.autoExport', checked)}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Error Reporting */}
              <div className="space-y-4">
                <div className="border-t pt-4">
                  <div className="flex items-center gap-2 mb-3">
                    <AlertTriangle className="w-4 h-4" />
                    <h3 className="text-sm font-medium">Error Reporting</h3>
                  </div>
                  
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="flex items-center justify-between">
                        <div className="space-y-0.5">
                          <Label className="text-sm">Error Reporting</Label>
                          <p className="text-xs text-muted-foreground">Send error reports</p>
                        </div>
                        <Switch
                          checked={errorReporting}
                          onCheckedChange={(checked) => handleSettingChange('system.debug.errorReporting', checked)}
                        />
                      </div>

                      <div className="flex items-center justify-between">
                        <div className="space-y-0.5">
                          <Label className="text-sm">Crash Reports</Label>
                          <p className="text-xs text-muted-foreground">Send crash data</p>
                        </div>
                        <Switch
                          checked={crashReports}
                          onCheckedChange={(checked) => handleSettingChange('system.debug.crashReports', checked)}
                        />
                      </div>
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label className="text-sm">Anonymize Error Data</Label>
                        <p className="text-xs text-muted-foreground">
                          Remove personal information from error reports
                        </p>
                      </div>
                      <Switch
                        checked={anonymizeErrors}
                        onCheckedChange={(checked) => handleSettingChange('system.debug.anonymizeErrors', checked)}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Development Features */}
              <div className="space-y-4">
                <div className="border-t pt-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Terminal className="w-4 h-4" />
                    <h3 className="text-sm font-medium">Development Features</h3>
                  </div>
                  
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="flex items-center justify-between">
                        <div className="space-y-0.5">
                          <Label className="text-sm">Hot Reload</Label>
                          <p className="text-xs text-muted-foreground">Auto-reload on changes</p>
                        </div>
                        <Switch
                          checked={hotReload}
                          onCheckedChange={(checked) => handleSettingChange('system.debug.hotReload', checked)}
                        />
                      </div>

                      <div className="flex items-center justify-between">
                        <div className="space-y-0.5">
                          <Label className="text-sm">Source Maps</Label>
                          <p className="text-xs text-muted-foreground">Enable source maps</p>
                        </div>
                        <Switch
                          checked={sourceMapEnabled}
                          onCheckedChange={(checked) => handleSettingChange('system.debug.sourceMaps', checked)}
                        />
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div className="flex items-center justify-between">
                        <div className="space-y-0.5">
                          <Label className="text-sm">Dev Tools</Label>
                          <p className="text-xs text-muted-foreground">Enable developer tools</p>
                        </div>
                        <Switch
                          checked={devTools}
                          onCheckedChange={(checked) => handleSettingChange('system.debug.devTools', checked)}
                        />
                      </div>

                      <div className="flex items-center justify-between">
                        <div className="space-y-0.5">
                          <Label className="text-sm">Profiling Mode</Label>
                          <p className="text-xs text-muted-foreground">Enable performance profiling</p>
                        </div>
                        <Switch
                          checked={profileMode}
                          onCheckedChange={(checked) => handleSettingChange('system.debug.profiling', checked)}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Live Log Viewer */}
              <div className="space-y-4">
                <div className="border-t pt-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Terminal className="w-4 h-4" />
                      <h3 className="text-sm font-medium">Live Logs</h3>
                    </div>
                    <div className="flex gap-2">
                      <Select value={logLevel} onValueChange={(value: any) => setLogLevel(value)}>
                        <SelectTrigger className="w-24 h-8">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="debug">Debug</SelectItem>
                          <SelectItem value="info">Info</SelectItem>
                          <SelectItem value="warn">Warn</SelectItem>
                          <SelectItem value="error">Error</SelectItem>
                        </SelectContent>
                      </Select>
                      <Button variant="outline" size="sm" onClick={exportLogs} disabled={logs.length === 0}>
                        <Download className="w-3 h-3 mr-1" />
                        Export
                      </Button>
                      <Button variant="outline" size="sm" onClick={clearLogs} disabled={logs.length === 0}>
                        <Trash2 className="w-3 h-3 mr-1" />
                        Clear
                      </Button>
                    </div>
                  </div>
                  
                  <div className="bg-muted/20 rounded-md p-3 h-48 overflow-y-auto font-mono text-xs">
                    {logs.length === 0 ? (
                      <div className="text-muted-foreground text-center pt-16">
                        No logs available. Debug mode will generate sample logs.
                      </div>
                    ) : (
                      <div className="space-y-1">
                        {logs
                          .filter(log => {
                            const level = log.match(/\[(DEBUG|INFO|WARN|ERROR)\]/)?.[1]?.toLowerCase();
                            return !level || level >= logLevel;
                          })
                          .map((log, index) => {
                            const level = log.match(/\[(DEBUG|INFO|WARN|ERROR)\]/)?.[1]?.toLowerCase() || 'info';
                            return (
                              <div key={index} className="flex items-start gap-2">
                                {getLogIcon(level)}
                                <span className="flex-1 break-all">{log}</span>
                              </div>
                            );
                          })}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </>
          )}

          {/* Actions */}
          <div className="border-t pt-4 space-y-2">
            <div className="flex gap-2">
              <Button
                variant="outline"
                onClick={resetDebugSettings}
                className="flex-1"
              >
                Reset Debug Settings
              </Button>
              {debugMode && (
                <Button
                  variant="outline"
                  onClick={exportLogs}
                  disabled={logs.length === 0}
                >
                  <Download className="w-4 h-4 mr-2" />
                  Export All Logs
                </Button>
              )}
            </div>
            
            {debugMode && (
              <div className="flex items-center gap-2 p-3 bg-yellow-50 dark:bg-yellow-950/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
                <AlertTriangle className="w-4 h-4 text-yellow-600" />
                <span className="text-sm text-yellow-700 dark:text-yellow-300">
                  Debug mode is active. This may impact performance and should be disabled in production.
                </span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default DebugSettings;