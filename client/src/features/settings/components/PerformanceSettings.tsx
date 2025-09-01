import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../../design-system/components/Card';
import { Label } from '../../design-system/components/Label';
import { Switch } from '../../design-system/components/Switch';
import { Input } from '../../design-system/components/Input';
import { Slider } from '../../design-system/components/Slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../../design-system/components/Select';
import { Badge } from '../../design-system/components/Badge';
import { Button } from '../../design-system/components/Button';
import { Zap, Cpu, HardDrive, Clock, BarChart3, AlertTriangle } from 'lucide-react';
import { useSelectiveSetting, useSettingSetter } from '../../../hooks/useSelectiveSettingsStore';
import { usePerformanceSettings } from '../../../hooks/usePerformanceSettings';

/**
 * PerformanceSettings Settings Panel
 * Provides performance optimization settings with selective access patterns
 */
export function PerformanceSettings() {
  const { set, batchSet } = useSettingSetter();
  const performanceSettings = usePerformanceSettings();
  
  // Use selective settings access for performance-related settings
  const performanceMode = useSelectiveSetting<string>('system.performance.mode') ?? 'balanced';
  const enableMonitoring = useSelectiveSetting<boolean>('system.performance.monitoring') ?? true;
  const memoryOptimization = useSelectiveSetting<boolean>('system.performance.memoryOptimization') ?? true;
  const cacheEnabled = useSelectiveSetting<boolean>('system.performance.cache') ?? true;
  const cacheTTL = useSelectiveSetting<number>('system.performance.cacheTTL') ?? 300000; // 5 minutes
  
  // Rendering performance
  const frameRateLimit = useSelectiveSetting<number>('system.performance.rendering.frameRate') ?? 60;
  const batchSize = useSelectiveSetting<number>('system.performance.rendering.batchSize') ?? 1000;
  const cullingEnabled = useSelectiveSetting<boolean>('system.performance.rendering.culling') ?? true;
  const lodEnabled = useSelectiveSetting<boolean>('system.performance.rendering.lod') ?? true;
  
  // Memory settings
  const maxMemoryUsage = useSelectiveSetting<number>('system.performance.memory.maxUsage') ?? 512; // MB
  const gcInterval = useSelectiveSetting<number>('system.performance.memory.gcInterval') ?? 30000; // 30s
  
  // Background processing
  const backgroundProcessing = useSelectiveSetting<boolean>('system.performance.background.enabled') ?? true;
  const maxWorkers = useSelectiveSetting<number>('system.performance.background.maxWorkers') ?? 4;
  const taskQueueSize = useSelectiveSetting<number>('system.performance.background.queueSize') ?? 100;

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

  const performanceModes = [
    { value: 'power-saver', label: 'Power Saver', icon: '🔋', description: 'Minimize CPU usage and extend battery life' },
    { value: 'balanced', label: 'Balanced', icon: '⚖️', description: 'Balance performance and efficiency' },
    { value: 'performance', label: 'Performance', icon: '🚀', description: 'Maximize performance' },
    { value: 'custom', label: 'Custom', icon: '⚙️', description: 'Custom performance settings' }
  ];

  const applyPerformanceMode = async (mode: string) => {
    const modeSettings: Record<string, Record<string, any>> = {
      'power-saver': {
        'system.performance.rendering.frameRate': 30,
        'system.performance.rendering.batchSize': 500,
        'system.performance.rendering.culling': true,
        'system.performance.rendering.lod': true,
        'system.performance.memory.maxUsage': 256,
        'system.performance.background.maxWorkers': 2,
        'system.performance.cache': true,
        'system.performance.cacheTTL': 600000 // 10 minutes
      },
      'balanced': {
        'system.performance.rendering.frameRate': 60,
        'system.performance.rendering.batchSize': 1000,
        'system.performance.rendering.culling': true,
        'system.performance.rendering.lod': true,
        'system.performance.memory.maxUsage': 512,
        'system.performance.background.maxWorkers': 4,
        'system.performance.cache': true,
        'system.performance.cacheTTL': 300000 // 5 minutes
      },
      'performance': {
        'system.performance.rendering.frameRate': 120,
        'system.performance.rendering.batchSize': 2000,
        'system.performance.rendering.culling': false,
        'system.performance.rendering.lod': false,
        'system.performance.memory.maxUsage': 1024,
        'system.performance.background.maxWorkers': 8,
        'system.performance.cache': true,
        'system.performance.cacheTTL': 60000 // 1 minute
      }
    };

    await handleSettingChange('system.performance.mode', mode);
    
    if (mode !== 'custom' && modeSettings[mode]) {
      await handleBatchChange(modeSettings[mode]);
    }
  };

  const getPerformanceStatus = () => {
    const score = Math.floor(
      (frameRateLimit / 120) * 30 +
      (batchSize / 2000) * 20 +
      (maxMemoryUsage / 1024) * 20 +
      (backgroundProcessing ? 15 : 0) +
      (cacheEnabled ? 10 : 0) +
      (cullingEnabled ? 5 : 0)
    );
    
    if (score >= 80) return { level: 'High', color: 'bg-green-500', textColor: 'text-green-700' };
    if (score >= 60) return { level: 'Medium', color: 'bg-yellow-500', textColor: 'text-yellow-700' };
    return { level: 'Low', color: 'bg-red-500', textColor: 'text-red-700' };
  };

  const status = getPerformanceStatus();

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Zap className="w-5 h-5" />
              <CardTitle>Performance Settings</CardTitle>
            </div>
            <Badge variant="outline" className={`${status.textColor} border-current`}>
              <div className={`w-2 h-2 rounded-full ${status.color} mr-2`} />
              {status.level} Performance
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Performance Mode */}
          <div className="space-y-3">
            <Label className="text-sm font-medium">Performance Mode</Label>
            <div className="grid grid-cols-2 gap-2">
              {performanceModes.map((mode) => (
                <Button
                  key={mode.value}
                  variant={performanceMode === mode.value ? 'default' : 'outline'}
                  className="flex flex-col items-start gap-1 h-auto p-3 text-left"
                  onClick={() => applyPerformanceMode(mode.value)}
                >
                  <div className="flex items-center gap-2">
                    <span className="text-lg">{mode.icon}</span>
                    <span className="font-medium">{mode.label}</span>
                  </div>
                  <span className="text-xs text-muted-foreground">{mode.description}</span>
                </Button>
              ))}
            </div>
          </div>

          {/* Monitoring */}
          <div className="space-y-3">
            <div className="border-t pt-4">
              <div className="flex items-center gap-2 mb-3">
                <BarChart3 className="w-4 h-4" />
                <h3 className="text-sm font-medium">Monitoring</h3>
              </div>
              
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label className="text-sm">Performance Monitoring</Label>
                  <p className="text-xs text-muted-foreground">
                    Track performance metrics and resource usage
                  </p>
                </div>
                <Switch
                  checked={enableMonitoring}
                  onCheckedChange={(checked) => handleSettingChange('system.performance.monitoring', checked)}
                />
              </div>
            </div>
          </div>

          {/* Custom Settings (only visible in custom mode) */}
          {performanceMode === 'custom' && (
            <div className="space-y-4">
              <div className="border-t pt-4">
                <div className="flex items-center gap-2 mb-4">
                  <Cpu className="w-4 h-4" />
                  <h3 className="text-sm font-medium">Rendering Performance</h3>
                </div>
                
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label className="text-sm">Frame Rate Limit</Label>
                      <div className="space-y-2">
                        <Slider
                          value={[frameRateLimit]}
                          onValueChange={([value]) => handleSettingChange('system.performance.rendering.frameRate', value)}
                          min={30}
                          max={144}
                          step={1}
                          className="w-full"
                        />
                        <div className="flex justify-between text-xs text-muted-foreground">
                          <span>30 FPS</span>
                          <span className="font-mono">{frameRateLimit} FPS</span>
                          <span>144 FPS</span>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <Label className="text-sm">Batch Size</Label>
                      <div className="flex items-center gap-2">
                        <Input
                          type="number"
                          min="100"
                          max="5000"
                          value={batchSize}
                          onChange={(e) => handleSettingChange('system.performance.rendering.batchSize', parseInt(e.target.value))}
                          className="w-24"
                        />
                        <span className="text-sm text-muted-foreground">items</span>
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label className="text-sm">Frustum Culling</Label>
                        <p className="text-xs text-muted-foreground">Skip rendering off-screen objects</p>
                      </div>
                      <Switch
                        checked={cullingEnabled}
                        onCheckedChange={(checked) => handleSettingChange('system.performance.rendering.culling', checked)}
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label className="text-sm">Level of Detail</Label>
                        <p className="text-xs text-muted-foreground">Reduce detail for distant objects</p>
                      </div>
                      <Switch
                        checked={lodEnabled}
                        onCheckedChange={(checked) => handleSettingChange('system.performance.rendering.lod', checked)}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Memory Management */}
              <div className="border-t pt-4">
                <div className="flex items-center gap-2 mb-4">
                  <HardDrive className="w-4 h-4" />
                  <h3 className="text-sm font-medium">Memory Management</h3>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label className="text-sm">Memory Limit</Label>
                    <div className="flex items-center gap-2">
                      <Input
                        type="number"
                        min="128"
                        max="2048"
                        value={maxMemoryUsage}
                        onChange={(e) => handleSettingChange('system.performance.memory.maxUsage', parseInt(e.target.value))}
                        className="w-24"
                      />
                      <span className="text-sm text-muted-foreground">MB</span>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label className="text-sm">Garbage Collection Interval</Label>
                    <div className="flex items-center gap-2">
                      <Input
                        type="number"
                        min="5000"
                        max="120000"
                        step="5000"
                        value={gcInterval}
                        onChange={(e) => handleSettingChange('system.performance.memory.gcInterval', parseInt(e.target.value))}
                        className="w-24"
                      />
                      <span className="text-sm text-muted-foreground">ms</span>
                    </div>
                  </div>
                </div>

                <div className="mt-4">
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label className="text-sm">Memory Optimization</Label>
                      <p className="text-xs text-muted-foreground">
                        Automatically free unused memory
                      </p>
                    </div>
                    <Switch
                      checked={memoryOptimization}
                      onCheckedChange={(checked) => handleSettingChange('system.performance.memoryOptimization', checked)}
                    />
                  </div>
                </div>
              </div>

              {/* Background Processing */}
              <div className="border-t pt-4">
                <div className="flex items-center gap-2 mb-4">
                  <Clock className="w-4 h-4" />
                  <h3 className="text-sm font-medium">Background Processing</h3>
                </div>
                
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label className="text-sm">Enable Background Tasks</Label>
                      <p className="text-xs text-muted-foreground">
                        Process tasks in background workers
                      </p>
                    </div>
                    <Switch
                      checked={backgroundProcessing}
                      onCheckedChange={(checked) => handleSettingChange('system.performance.background.enabled', checked)}
                    />
                  </div>

                  {backgroundProcessing && (
                    <div className="grid grid-cols-2 gap-4 pl-4 border-l-2 border-muted">
                      <div className="space-y-2">
                        <Label className="text-sm">Max Workers</Label>
                        <Input
                          type="number"
                          min="1"
                          max="16"
                          value={maxWorkers}
                          onChange={(e) => handleSettingChange('system.performance.background.maxWorkers', parseInt(e.target.value))}
                          className="w-20"
                        />
                      </div>

                      <div className="space-y-2">
                        <Label className="text-sm">Queue Size</Label>
                        <Input
                          type="number"
                          min="10"
                          max="1000"
                          value={taskQueueSize}
                          onChange={(e) => handleSettingChange('system.performance.background.queueSize', parseInt(e.target.value))}
                          className="w-24"
                        />
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Cache Settings */}
              <div className="border-t pt-4">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label className="text-sm">Enable Caching</Label>
                      <p className="text-xs text-muted-foreground">
                        Cache frequently accessed data
                      </p>
                    </div>
                    <Switch
                      checked={cacheEnabled}
                      onCheckedChange={(checked) => handleSettingChange('system.performance.cache', checked)}
                    />
                  </div>

                  {cacheEnabled && (
                    <div className="space-y-2 pl-4 border-l-2 border-muted">
                      <Label className="text-sm">Cache TTL</Label>
                      <div className="flex items-center gap-2">
                        <Input
                          type="number"
                          min="30000"
                          max="3600000"
                          step="30000"
                          value={cacheTTL}
                          onChange={(e) => handleSettingChange('system.performance.cacheTTL', parseInt(e.target.value))}
                          className="w-32"
                        />
                        <span className="text-sm text-muted-foreground">ms ({Math.floor(cacheTTL / 60000)}min)</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Performance Warning */}
          {(frameRateLimit > 60 || maxMemoryUsage > 512) && (
            <div className="flex items-center gap-2 p-3 bg-yellow-50 dark:bg-yellow-950/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
              <AlertTriangle className="w-4 h-4 text-yellow-600" />
              <span className="text-sm text-yellow-700 dark:text-yellow-300">
                High performance settings may impact battery life and system stability
              </span>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export default PerformanceSettings;