import React, { useState, useEffect } from 'react';
import { Activity, RefreshCw, Gauge, Layers, TrendingUp, Settings as SettingsIcon } from 'lucide-react';
import { useSettingsStore } from '../../../../store/settingsStore';
import { Button } from '@/features/design-system/components/Button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { Switch } from '@/features/design-system/components/Switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/features/design-system/components/Select';
import { Slider } from '@/features/design-system/components/Slider';
import { Badge } from '@/features/design-system/components/Badge';
import { Separator } from '@/features/design-system/components/Separator';
import { Label } from '@/features/design-system/components/Label';



interface SystemStatus {
  graphStatus: 'idle' | 'computing' | 'converged' | 'error';
  currentIteration: number;
  convergenceValue: number;
  activeConstraintsCount: number;
  clusteringActive: boolean;
  lastUpdate: Date;
}

const COMPUTE_MODES = [
  { value: 'basic-force-directed', label: 'Basic Force-Directed', description: 'Classic spring-based layout' },
  { value: 'dual-graph', label: 'Dual Graph', description: 'Separate page/block graphs' },
  { value: 'hierarchical', label: 'Hierarchical', description: 'Tree-based organization' },
  { value: 'clustered', label: 'Clustered', description: 'Community detection' },
  { value: 'hybrid', label: 'Hybrid (Advanced)', description: 'GPU-accelerated multi-pass' }
];

export const DashboardControlPanel: React.FC = () => {
  const settings = useSettingsStore(state => state.settings);
  const updateSettings = useSettingsStore(state => state.updateSettings);

  const updateSetting = (path: string, value: any) => {
    updateSettings((draft: any) => {
      const parts = path.split('.');
      let current = draft;
      for (let i = 0; i < parts.length - 1; i++) {
        if (!current[parts[i]]) current[parts[i]] = {};
        current = current[parts[i]];
      }
      current[parts[parts.length - 1]] = value;
    });
  };
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    graphStatus: 'idle',
    currentIteration: 0,
    convergenceValue: 0,
    activeConstraintsCount: 0,
    clusteringActive: false,
    lastUpdate: new Date()
  });

  
  useEffect(() => {
    const pollStatus = async () => {
      try {
        
        
        const response = await fetch('/api/physics/status');
        if (response.ok) {
          const data = await response.json();
          setSystemStatus({
            graphStatus: data.status || 'idle',
            currentIteration: data.iteration || 0,
            convergenceValue: data.convergence || 0,
            activeConstraintsCount: data.constraints?.length || 0,
            clusteringActive: data.clustering_active || false,
            lastUpdate: new Date()
          });
        }
      } catch (error) {
        console.warn('Failed to fetch system status:', error);
      }
    };

    if (settings?.dashboard?.autoRefresh) {
      const interval = (settings?.dashboard?.refreshInterval || 2) * 1000;
      const timer = setInterval(pollStatus, interval);
      pollStatus(); 
      return () => clearInterval(timer);
    }
  }, [settings?.dashboard?.autoRefresh, settings?.dashboard?.refreshInterval]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'idle': return 'bg-gray-500';
      case 'computing': return 'bg-blue-500 animate-pulse';
      case 'converged': return 'bg-green-500';
      case 'error': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'computing': return <Activity className="w-4 h-4 animate-spin" />;
      case 'converged': return <TrendingUp className="w-4 h-4" />;
      default: return <Gauge className="w-4 h-4" />;
    }
  };

  return (
    <div className="space-y-6">
      {}
      <div>
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <SettingsIcon className="w-6 h-6" />
          Dashboard Settings
        </h2>
        <p className="text-sm text-muted-foreground mt-1">
          Real-time system monitoring and computation controls
        </p>
      </div>

      {}
      {(settings?.dashboard as any)?.showStatus && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {getStatusIcon(systemStatus.graphStatus)}
              Graph Computation Status
            </CardTitle>
            <CardDescription>
              Live monitoring of physics and graph computation
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {}
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Current Status:</span>
              <Badge className={`${getStatusColor(systemStatus.graphStatus)} text-white`}>
                {systemStatus.graphStatus.toUpperCase()}
              </Badge>
            </div>

            {}
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Iterations:</span>
              <span className="text-lg font-mono">
                {systemStatus.currentIteration.toLocaleString()} / {(settings?.dashboard?.iterationCount || 1000).toLocaleString()}
              </span>
            </div>

            {}
            {settings?.dashboard?.showConvergence && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Convergence:</span>
                  <span className="text-lg font-mono">
                    {(systemStatus.convergenceValue * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-green-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${systemStatus.convergenceValue * 100}%` }}
                  />
                </div>
              </div>
            )}

            {}
            {settings?.dashboard?.activeConstraints && (
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Active Constraints:</span>
                <Badge variant="outline">
                  {systemStatus.activeConstraintsCount}
                </Badge>
              </div>
            )}

            {}
            {settings?.dashboard?.clusteringActive && (
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Clustering:</span>
                <Badge variant={systemStatus.clusteringActive ? "default" : "secondary"}>
                  {systemStatus.clusteringActive ? 'Active' : 'Inactive'}
                </Badge>
              </div>
            )}

            {}
            <div className="text-xs text-muted-foreground pt-2 border-t">
              Last updated: {systemStatus.lastUpdate.toLocaleTimeString()}
            </div>
          </CardContent>
        </Card>
      )}

      <Separator />

      {}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers className="w-5 h-5" />
            Computation Mode
          </CardTitle>
          <CardDescription>
            Select the graph layout algorithm
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="compute-mode">Layout Algorithm</Label>
            <Select
              value={settings?.dashboard?.computeMode || 'basic-force-directed'}
              onValueChange={(value) => updateSetting('dashboard.computeMode', value)}
            >
              <SelectTrigger id="compute-mode">
                <SelectValue placeholder="Select compute mode" />
              </SelectTrigger>
              <SelectContent>
                {COMPUTE_MODES.map(mode => (
                  <SelectItem key={mode.value} value={mode.value}>
                    <div className="flex flex-col">
                      <span className="font-medium">{mode.label}</span>
                      <span className="text-xs text-muted-foreground">{mode.description}</span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="iteration-count">Max Iterations</Label>
              <span className="text-sm font-mono">{settings?.dashboard?.iterationCount || 1000}</span>
            </div>
            <Slider
              id="iteration-count"
              min={100}
              max={5000}
              step={100}
              value={[settings?.dashboard?.iterationCount || 1000]}
              onValueChange={([value]) => updateSetting('dashboard.iterationCount', value)}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground">
              Higher values = more accurate layout but slower computation
            </p>
          </div>
        </CardContent>
      </Card>

      <Separator />

      {}
      <Card>
        <CardHeader>
          <CardTitle>Display Settings</CardTitle>
          <CardDescription>
            Configure what information is shown on the dashboard
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {}
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label htmlFor="show-status">Show Graph Status</Label>
              <p className="text-xs text-muted-foreground">
                Display real-time computation status card
              </p>
            </div>
            <Switch
              id="show-status"
              checked={(settings?.dashboard as any)?.showStatus ?? true}
              onCheckedChange={(checked) => updateSetting('dashboard.showStatus', checked)}
            />
          </div>

          {}
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label htmlFor="show-convergence">Show Convergence</Label>
              <p className="text-xs text-muted-foreground">
                Display convergence progress bar
              </p>
            </div>
            <Switch
              id="show-convergence"
              checked={settings?.dashboard?.showConvergence ?? true}
              onCheckedChange={(checked) => updateSetting('dashboard.showConvergence', checked)}
            />
          </div>

          {}
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label htmlFor="show-constraints">Show Active Constraints</Label>
              <p className="text-xs text-muted-foreground">
                Display count of active physics constraints
              </p>
            </div>
            <Switch
              id="show-constraints"
              checked={Boolean(settings?.dashboard?.activeConstraints) ?? true}
              onCheckedChange={(checked) => updateSetting('dashboard.activeConstraints', checked)}
            />
          </div>

          {}
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label htmlFor="show-clustering">Show Clustering Status</Label>
              <p className="text-xs text-muted-foreground">
                Display whether clustering is active
              </p>
            </div>
            <Switch
              id="show-clustering"
              checked={settings?.dashboard?.clusteringActive ?? true}
              onCheckedChange={(checked) => updateSetting('dashboard.clusteringActive', checked)}
            />
          </div>
        </CardContent>
      </Card>

      <Separator />

      {}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <RefreshCw className="w-5 h-5" />
            Auto-Refresh
          </CardTitle>
          <CardDescription>
            Automatic status polling configuration
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {}
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label htmlFor="auto-refresh">Enable Auto-Refresh</Label>
              <p className="text-xs text-muted-foreground">
                Automatically update dashboard status
              </p>
            </div>
            <Switch
              id="auto-refresh"
              checked={settings?.dashboard?.autoRefresh ?? true}
              onCheckedChange={(checked) => updateSetting('dashboard.autoRefresh', checked)}
            />
          </div>

          {}
          {settings?.dashboard?.autoRefresh && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="refresh-interval">Refresh Interval</Label>
                <span className="text-sm font-mono">{settings?.dashboard?.refreshInterval || 2}s</span>
              </div>
              <Slider
                id="refresh-interval"
                min={1}
                max={10}
                step={1}
                value={[settings?.dashboard?.refreshInterval || 2]}
                onValueChange={([value]) => updateSetting('dashboard.refreshInterval', value)}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground">
                How often to poll for status updates (1-10 seconds)
              </p>
            </div>
          )}

          {}
          <Button
            variant="outline"
            className="w-full"
            onClick={() => setSystemStatus(prev => ({ ...prev, lastUpdate: new Date() }))}
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh Now
          </Button>
        </CardContent>
      </Card>
    </div>
  );
};
