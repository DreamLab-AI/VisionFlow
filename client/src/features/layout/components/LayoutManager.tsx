import React, { useState, useMemo } from 'react';
import { useSelectiveSetting, useSelectiveSettings, useSettingSetter } from '@/hooks/useSelectiveSettingsStore';
import { Card, CardHeader, CardTitle, CardContent } from '@/features/design-system/components/Card';
import { Button } from '@/features/design-system/components/Button';
import { Badge } from '@/features/design-system/components/Badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/features/design-system/components/Select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/features/design-system/components/Tabs';
import { ScrollArea } from '@/features/design-system/components/ScrollArea';
import { Layout, Grid, Maximize, Minimize, RotateCw, Save } from 'lucide-react';
import { createLogger } from '@/utils/logger';

const logger = createLogger('LayoutManager');

interface LayoutManagerProps {
  className?: string;
}

interface LayoutPreset {
  id: string;
  name: string;
  description: string;
  type: 'builtin' | 'custom';
  layout: {
    panels: Array<{
      id: string;
      component: string;
      position: { x: number; y: number; width: number; height: number };
      visible: boolean;
    }>;
    grid: { columns: number; rows: number };
  };
  createdAt: Date;
}

export const LayoutManager: React.FC<LayoutManagerProps> = ({ className }) => {
  const { set } = useSettingSetter();
  const [selectedPreset, setSelectedPreset] = useState('default');
  const [activeTab, setActiveTab] = useState('presets');
  
  // Subscribe only to layout settings
  const layoutSettings = useSelectiveSettings({
    enabled: 'layout.enabled',
    currentLayout: 'layout.currentLayout',
    autoSave: 'layout.autoSave.enabled',
    responsive: 'layout.responsive.enabled',
    snapToGrid: 'layout.snapToGrid',
    showGrid: 'layout.grid.visible',
    gridSize: 'layout.grid.size',
    maxPanels: 'layout.limits.maxPanels',
    allowOverlap: 'layout.allowOverlap'
  });
  
  // Mock layout presets
  const layoutPresets: LayoutPreset[] = useMemo(() => [
    {
      id: 'default',
      name: 'Default Dashboard',
      description: 'Standard layout with graph view and control panels',
      type: 'builtin',
      layout: {
        panels: [
          { id: 'graph', component: 'GraphCanvas', position: { x: 0, y: 0, width: 8, height: 6 }, visible: true },
          { id: 'controls', component: 'ControlPanel', position: { x: 8, y: 0, width: 4, height: 6 }, visible: true },
          { id: 'data', component: 'DataPanel', position: { x: 0, y: 6, width: 6, height: 4 }, visible: true },
          { id: 'analytics', component: 'AnalyticsPanel', position: { x: 6, y: 6, width: 6, height: 4 }, visible: true }
        ],
        grid: { columns: 12, rows: 10 }
      },
      createdAt: new Date()
    },
    {
      id: 'analytics-focused',
      name: 'Analytics Focused',
      description: 'Layout optimized for data analysis and reporting',
      type: 'builtin',
      layout: {
        panels: [
          { id: 'analytics', component: 'AnalyticsPanel', position: { x: 0, y: 0, width: 8, height: 8 }, visible: true },
          { id: 'metrics', component: 'MetricsDisplay', position: { x: 8, y: 0, width: 4, height: 4 }, visible: true },
          { id: 'export', component: 'ExportPanel', position: { x: 8, y: 4, width: 4, height: 4 }, visible: true },
          { id: 'filters', component: 'FilterPanel', position: { x: 0, y: 8, width: 12, height: 2 }, visible: true }
        ],
        grid: { columns: 12, rows: 10 }
      },
      createdAt: new Date()
    },
    {
      id: 'development',
      name: 'Development Mode',
      description: 'Layout for development and debugging',
      type: 'builtin',
      layout: {
        panels: [
          { id: 'graph', component: 'GraphCanvas', position: { x: 0, y: 0, width: 6, height: 6 }, visible: true },
          { id: 'bots', component: 'BotManager', position: { x: 6, y: 0, width: 6, height: 6 }, visible: true },
          { id: 'history', component: 'HistoryPanel', position: { x: 0, y: 6, width: 4, height: 4 }, visible: true },
          { id: 'timeline', component: 'TimelinePanel', position: { x: 4, y: 6, width: 4, height: 4 }, visible: true },
          { id: 'notifications', component: 'NotificationCenter', position: { x: 8, y: 6, width: 4, height: 4 }, visible: true }
        ],
        grid: { columns: 12, rows: 10 }
      },
      createdAt: new Date()
    }
  ], []);
  
  const availableComponents = useMemo(() => [
    { id: 'graph', name: 'Graph Canvas', category: 'Visualization' },
    { id: 'controls', name: 'Control Panel', category: 'Interface' },
    { id: 'data', name: 'Data Panel', category: 'Data' },
    { id: 'analytics', name: 'Analytics Panel', category: 'Analytics' },
    { id: 'metrics', name: 'Metrics Display', category: 'Analytics' },
    { id: 'bots', name: 'Bot Manager', category: 'Automation' },
    { id: 'filters', name: 'Filter Panel', category: 'Data' },
    { id: 'search', name: 'Search Panel', category: 'Data' },
    { id: 'export', name: 'Export Panel', category: 'Data' },
    { id: 'history', name: 'History Panel', category: 'System' },
    { id: 'timeline', name: 'Timeline Panel', category: 'System' },
    { id: 'notifications', name: 'Notification Center', category: 'Interface' },
    { id: 'workspace', name: 'Workspace Manager', category: 'Interface' },
    { id: 'collaboration', name: 'Collaboration Panel', category: 'Interface' },
    { id: 'ai', name: 'AI Assistant', category: 'AI' }
  ], []);
  
  const currentPreset = layoutPresets.find(p => p.id === selectedPreset) || layoutPresets[0];
  
  const applyLayout = (presetId: string) => {
    const preset = layoutPresets.find(p => p.id === presetId);
    if (preset) {
      setSelectedPreset(presetId);
      set('layout.currentLayout', presetId);
      logger.info('Applied layout preset', { presetId, preset: preset.name });
    }
  };
  
  const saveCurrentLayout = () => {
    logger.info('Saving current layout as custom preset');
    // In real app, save current panel positions as new preset
  };
  
  const resetLayout = () => {
    applyLayout('default');
    logger.info('Reset to default layout');
  };
  
  const LayoutPreview: React.FC<{ preset: LayoutPreset }> = ({ preset }) => (
    <div className="border rounded-lg p-4 aspect-video bg-gray-50 relative overflow-hidden">
      <div className="absolute inset-2 grid gap-1" 
           style={{ 
             gridTemplateColumns: `repeat(${preset.layout.grid.columns}, 1fr)`,
             gridTemplateRows: `repeat(${preset.layout.grid.rows}, 1fr)`
           }}>
        {preset.layout.panels.filter(p => p.visible).map((panel) => (
          <div 
            key={panel.id}
            className="bg-blue-200 border border-blue-300 rounded text-xs flex items-center justify-center font-medium text-blue-800"
            style={{
              gridColumn: `${panel.position.x + 1} / span ${panel.position.width}`,
              gridRow: `${panel.position.y + 1} / span ${panel.position.height}`
            }}
          >
            {availableComponents.find(c => c.id === panel.id)?.name || panel.component}
          </div>
        ))}
      </div>
    </div>
  );
  
  if (!layoutSettings.enabled) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layout size={20} />
            Layout Manager
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <Layout size={48} className="mx-auto mb-4 text-gray-400" />
            <p className="text-muted-foreground">Layout management is disabled</p>
            <p className="text-sm text-muted-foreground mt-2">
              Enable layout management to customize panel arrangements
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Layout size={20} />
            Layout Manager
            <Badge variant="outline">
              {layoutSettings.currentLayout}
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <Button size="sm" variant="outline" onClick={saveCurrentLayout}>
              <Save size={16} className="mr-1" />
              Save
            </Button>
            <Button size="sm" variant="outline" onClick={resetLayout}>
              <RotateCw size={16} className="mr-1" />
              Reset
            </Button>
          </div>
        </CardTitle>
      </CardHeader>
      
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="presets">Layout Presets</TabsTrigger>
            <TabsTrigger value="components">Components</TabsTrigger>
            <TabsTrigger value="settings">Settings</TabsTrigger>
          </TabsList>
          
          <TabsContent value="presets" className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Select Layout</label>
              <Select value={selectedPreset} onValueChange={applyLayout}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {layoutPresets.map(preset => (
                    <SelectItem key={preset.id} value={preset.id}>
                      {preset.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            
            <ScrollArea className="h-[400px]">
              <div className="space-y-4">
                {layoutPresets.map((preset) => (
                  <div key={preset.id} className="border rounded-lg p-4">
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <h3 className="font-medium flex items-center gap-2">
                          {preset.name}
                          <Badge className={preset.type === 'builtin' ? 'bg-blue-100 text-blue-800' : 'bg-green-100 text-green-800'}>
                            {preset.type}
                          </Badge>
                          {preset.id === selectedPreset && (
                            <Badge className="bg-orange-100 text-orange-800">
                              Active
                            </Badge>
                          )}
                        </h3>
                        <p className="text-sm text-muted-foreground">
                          {preset.description}
                        </p>
                      </div>
                      <Button
                        size="sm"
                        onClick={() => applyLayout(preset.id)}
                        disabled={preset.id === selectedPreset}
                      >
                        Apply
                      </Button>
                    </div>
                    
                    <LayoutPreview preset={preset} />
                    
                    <div className="mt-3 flex items-center justify-between text-xs text-muted-foreground">
                      <span>{preset.layout.panels.length} panels</span>
                      <span>{preset.layout.grid.columns}×{preset.layout.grid.rows} grid</span>
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </TabsContent>
          
          <TabsContent value="components" className="space-y-4">
            <div className="space-y-4">
              <h3 className="font-medium">Available Components</h3>
              <ScrollArea className="h-[400px]">
                <div className="grid grid-cols-2 gap-2">
                  {availableComponents.map((component) => {
                    const isVisible = currentPreset.layout.panels.find(
                      p => p.id === component.id
                    )?.visible;
                    
                    return (
                      <div key={component.id} className="border rounded p-3">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm font-medium">{component.name}</span>
                          <Badge variant="secondary" className="text-xs">
                            {component.category}
                          </Badge>
                        </div>
                        <div className="flex items-center gap-2">
                          <Button
                            size="sm"
                            variant={isVisible ? 'default' : 'outline'}
                            onClick={() => {
                              // In real app, toggle component visibility
                              logger.info('Toggle component', { componentId: component.id, visible: !isVisible });
                            }}
                            className="h-6 text-xs"
                          >
                            {isVisible ? 'Visible' : 'Hidden'}
                          </Button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </ScrollArea>
            </div>
          </TabsContent>
          
          <TabsContent value="settings" className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Auto Save</span>
                <Button
                  variant={layoutSettings.autoSave ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => set('layout.autoSave.enabled', !layoutSettings.autoSave)}
                >
                  {layoutSettings.autoSave ? 'ON' : 'OFF'}
                </Button>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Responsive</span>
                <Button
                  variant={layoutSettings.responsive ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => set('layout.responsive.enabled', !layoutSettings.responsive)}
                >
                  {layoutSettings.responsive ? 'ON' : 'OFF'}
                </Button>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Snap to Grid</span>
                <Button
                  variant={layoutSettings.snapToGrid ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => set('layout.snapToGrid', !layoutSettings.snapToGrid)}
                >
                  {layoutSettings.snapToGrid ? 'ON' : 'OFF'}
                </Button>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Show Grid</span>
                <Button
                  variant={layoutSettings.showGrid ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => set('layout.grid.visible', !layoutSettings.showGrid)}
                >
                  {layoutSettings.showGrid ? 'ON' : 'OFF'}
                </Button>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Allow Overlap</span>
                <Button
                  variant={layoutSettings.allowOverlap ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => set('layout.allowOverlap', !layoutSettings.allowOverlap)}
                >
                  {layoutSettings.allowOverlap ? 'ON' : 'OFF'}
                </Button>
              </div>
            </div>
            
            <div className="border-t pt-4">
              <h3 className="font-medium mb-2">Layout Limits</h3>
              <div className="text-sm text-muted-foreground space-y-1">
                <p>Maximum panels: {layoutSettings.maxPanels}</p>
                <p>Grid size: {layoutSettings.gridSize}px</p>
                <p>Current panels: {currentPreset.layout.panels.filter(p => p.visible).length}</p>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default LayoutManager;