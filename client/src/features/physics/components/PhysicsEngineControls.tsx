import React, { useState, useCallback, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { useSettingsStore } from '@/store/settingsStore';
import { PhysicsSettings } from '@/features/settings/config/settings';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/features/design-system/components/Select';
import { Slider } from '@/features/design-system/components/Slider';
import { Switch } from '@/features/design-system/components/Switch';
import { Label } from '@/features/design-system/components/Label';
import { Button } from '@/features/design-system/components/Button';
import { Badge } from '@/features/design-system/components/Badge';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/features/design-system/components/Tabs';
import { Info, Cpu, Zap, Layers, GitBranch, Activity, AlertCircle, Plus } from 'lucide-react';
import { useToast } from '@/features/design-system/components/Toast';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/features/design-system/components/Tooltip';
import { SemanticClusteringControls } from '@/features/analytics/components/SemanticClusteringControls';
import { ConstraintBuilderDialog } from './ConstraintBuilderDialog';
import { PhysicsPresets } from './PhysicsPresets';

type KernelMode = 'legacy' | 'advanced' | 'visual_analytics';

interface ForceParameters {
  repulsion: number;
  attraction: number;
  spring: number;
  damping: number;
  gravity: number;
  timeStep: number;
  maxVelocity: number;
  temperature: number;
}

interface ConstraintType {
  id: string;
  name: string;
  enabled: boolean;
  description: string;
  icon: string;
}

interface IsolationLayer {
  id: string;
  name: string;
  color: string;
  opacity: number;
  nodeCount: number;
  active: boolean;
}

export function PhysicsEngineControls() {
  const { toast } = useToast();
  // Use the settings store
  const { settings, initialized, updateSettings } = useSettingsStore();
  const [currentGraph] = useState<'logseq' | 'visionflow'>('logseq');
  
  // Helper function to update physics settings
  const updatePhysics = async (physicsUpdate: Partial<PhysicsSettings>) => {
    updateSettings((draft) => {
      if (!draft.visualisation.graphs[currentGraph]) {
        draft.visualisation.graphs[currentGraph] = {
          nodes: draft.visualisation.graphs.logseq.nodes,
          edges: draft.visualisation.graphs.logseq.edges,
          labels: draft.visualisation.graphs.logseq.labels,
          physics: draft.visualisation.graphs.logseq.physics,
        };
      }
      Object.assign(draft.visualisation.graphs[currentGraph].physics, physicsUpdate);
    });
  };
  
  const loadSettings = async () => {
    // Settings are automatically loaded by the store
  };
  
  // State management
  const [kernelMode, setKernelMode] = useState<KernelMode>('visual_analytics');
  const [showConstraintBuilder, setShowConstraintBuilder] = useState(false);
  
  // Initialize force params from settings
  const physicsSettings = settings?.visualisation?.graphs?.[currentGraph]?.physics;
  const [forceParams, setForceParams] = useState<ForceParameters>({
    repulsion: physicsSettings?.repulsion_strength || 2.0,    // Stable default
    attraction: physicsSettings?.attraction_strength || 0.0001, // Stable default
    spring: physicsSettings?.spring_strength || 0.005,        // Stable default
    damping: physicsSettings?.damping || 0.95,                // Stable default
    gravity: physicsSettings?.gravity || 0.0001,              // Stable default
    timeStep: physicsSettings?.time_step || 0.016,            // Stable default (60fps)
    maxVelocity: physicsSettings?.max_velocity || 2.0,        // Stable default
    temperature: physicsSettings?.temperature || 0.01,        // Stable default
  });
  
  const [constraints, setConstraints] = useState<ConstraintType[]>([
    { id: 'fixed', name: 'Fixed Position', enabled: false, description: 'Lock nodes in place', icon: '📌' },
    { id: 'separation', name: 'Separation', enabled: true, description: 'Minimum distance between nodes', icon: '↔️' },
    { id: 'alignment_h', name: 'Horizontal Alignment', enabled: false, description: 'Align nodes horizontally', icon: '═' },
    { id: 'alignment_v', name: 'Vertical Alignment', enabled: false, description: 'Align nodes vertically', icon: '║' },
    { id: 'boundary', name: 'Boundary', enabled: true, description: 'Keep nodes within bounds', icon: '⬚' },
    { id: 'cluster', name: 'Cluster', enabled: false, description: 'Group related nodes', icon: '🔶' },
    { id: 'tree', name: 'Tree Layout', enabled: false, description: 'Hierarchical tree structure', icon: '🌳' },
    { id: 'radial', name: 'Radial', enabled: false, description: 'Radial distance constraints', icon: '⭕' },
    { id: 'layer', name: 'Layer', enabled: false, description: 'Layer-based positioning', icon: '📚' },
    { id: 'collision', name: 'Collision', enabled: true, description: 'Prevent node overlap', icon: '💥' },
  ]);
  
  const [isolationLayers, setIsolationLayers] = useState<IsolationLayer[]>([
    { id: 'focus', name: 'Focus Layer', color: '#3B82F6', opacity: 1.0, nodeCount: 0, active: false },
    { id: 'context', name: 'Context Layer', color: '#8B5CF6', opacity: 0.7, nodeCount: 0, active: false },
    { id: 'background', name: 'Background Layer', color: '#6B7280', opacity: 0.3, nodeCount: 0, active: false },
  ]);
  
  const [trajectorySettings, setTrajectorySettings] = useState({
    enabled: false,
    length: 8,
    fadeRate: 0.9,
    colorByVelocity: true,
  });
  
  const [gpuMetrics, setGpuMetrics] = useState({
    utilization: 0,
    memory: 0,
    temperature: 0,
    power: 0,
  });

  // Initialize the settings store on mount
  useEffect(() => {
    if (!initialized) {
      // The settings store automatically initializes on first use
    }
  }, [initialized]);
  
  // Update local state when settings change
  useEffect(() => {
    if (physicsSettings && initialized) {
      setForceParams({
        repulsion: physicsSettings.repulsion_strength || 2.0,      // Stable default
        attraction: physicsSettings.attraction_strength || 0.0001, // Stable default
        spring: physicsSettings.spring_strength || 0.005,          // Stable default
        damping: physicsSettings.damping || 0.95,                  // Stable default
        gravity: physicsSettings.gravity || 0.0001,                // Stable default
        timeStep: physicsSettings.time_step || 0.016,              // Stable default (60fps)
        maxVelocity: physicsSettings.max_velocity || 2.0,          // Stable default
        temperature: physicsSettings.temperature || 0.01,          // Stable default
      });
    }
  }, [physicsSettings, initialized]);
  
  // Fetch GPU metrics periodically
  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await fetch('/api/analytics/gpu-metrics');
        if (response.ok) {
          const data = await response.json();
          setGpuMetrics(data);
        }
      } catch (error) {
        console.error('Failed to fetch GPU metrics:', error);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 2000);
    return () => clearInterval(interval);
  }, []);

  // Handlers
  const handleKernelModeChange = useCallback(async (mode: KernelMode) => {
    try {
      const response = await fetch('/api/analytics/kernel-mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode }),
      });
      
      if (response.ok) {
        setKernelMode(mode);
        toast({
          title: 'Kernel Mode Changed',
          description: `Switched to ${mode} kernel`,
        });
      }
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to change kernel mode',
        variant: 'destructive',
      });
    }
  }, [toast]);

  const handleForceParamChange = useCallback(async (param: keyof ForceParameters, value: number) => {
    const newParams = { ...forceParams, [param]: value };
    setForceParams(newParams);
    
    // Map to settings structure
    const physicsUpdate: any = {};
    switch (param) {
      case 'repulsion':
        physicsUpdate.repulsion_strength = value;
        break;
      case 'attraction':
        physicsUpdate.attraction_strength = value;
        break;
      case 'spring':
        physicsUpdate.spring_strength = value;
        break;
      case 'damping':
        physicsUpdate.damping = value;
        break;
      case 'gravity':
        physicsUpdate.gravity = value;
        break;
      case 'timeStep':
        physicsUpdate.time_step = value;
        break;
      case 'maxVelocity':
        physicsUpdate.max_velocity = value;
        break;
      case 'temperature':
        physicsUpdate.temperature = value;
        break;
    }
    
    try {
      // Update through settings store
      await updatePhysics(physicsUpdate);
      
      // Also send to analytics endpoint for immediate GPU update
      await fetch('/api/analytics/params', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newParams),
      });
    } catch (error) {
      console.error('Failed to update force parameters:', error);
      toast({
        title: 'Error',
        description: 'Failed to update physics parameters',
        variant: 'destructive',
      });
    }
  }, [forceParams, updatePhysics, toast]);

  const handleConstraintToggle = useCallback(async (constraintId: string) => {
    const newConstraints = constraints.map(c => 
      c.id === constraintId ? { ...c, enabled: !c.enabled } : c
    );
    setConstraints(newConstraints);
    
    try {
      await fetch('/api/analytics/constraints', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          constraints: newConstraints.filter(c => c.enabled).map(c => c.id),
        }),
      });
    } catch (error) {
      console.error('Failed to update constraints:', error);
    }
  }, [constraints]);

  const handleLayerToggle = useCallback(async (layerId: string) => {
    const newLayers = isolationLayers.map(l => 
      l.id === layerId ? { ...l, active: !l.active } : l
    );
    setIsolationLayers(newLayers);
    
    try {
      await fetch('/api/analytics/layers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          layers: newLayers.filter(l => l.active),
        }),
      });
    } catch (error) {
      console.error('Failed to update isolation layers:', error);
    }
  }, [isolationLayers]);

  const handleSaveConstraint = useCallback((constraint: any) => {
    console.log('Saving constraint:', constraint);
    // TODO: Implement constraint saving
  }, []);

  return (
    <div className="h-full overflow-auto">
      <Tabs defaultValue="engine" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="engine">Engine</TabsTrigger>
          <TabsTrigger value="forces">Forces</TabsTrigger>
          <TabsTrigger value="constraints">Constraints</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>
        
        {/* Engine Tab */}
        <TabsContent value="engine" className="space-y-4">
          {/* Presets */}
          <PhysicsPresets />
          
          {/* GPU Status Card */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Cpu className="h-5 w-5" />
                GPU Engine Status
              </CardTitle>
              <CardDescription>
                Real-time GPU performance metrics
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Utilization</span>
                    <Badge variant={gpuMetrics.utilization > 80 ? 'destructive' : 'default'}>
                      {gpuMetrics.utilization}%
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Memory</span>
                    <Badge variant={gpuMetrics.memory > 80 ? 'destructive' : 'default'}>
                      {gpuMetrics.memory}%
                    </Badge>
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Temperature</span>
                    <Badge variant={gpuMetrics.temperature > 75 ? 'destructive' : 'default'}>
                      {gpuMetrics.temperature}°C
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Power</span>
                    <Badge>{gpuMetrics.power}W</Badge>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Kernel Mode Selector */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="h-5 w-5" />
                GPU Kernel Mode
              </CardTitle>
              <CardDescription>
                Select the GPU computation kernel
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Select value={kernelMode} onValueChange={handleKernelModeChange}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="legacy">
                    <div className="flex flex-col">
                      <span>Legacy</span>
                      <span className="text-xs text-muted-foreground">Basic force-directed layout</span>
                    </div>
                  </SelectItem>
                  <SelectItem value="advanced">
                    <div className="flex flex-col">
                      <span>Advanced</span>
                      <span className="text-xs text-muted-foreground">UMAP, spectral clustering, GNN</span>
                    </div>
                  </SelectItem>
                  <SelectItem value="visual_analytics">
                    <div className="flex flex-col">
                      <span>Visual Analytics</span>
                      <span className="text-xs text-muted-foreground">Temporal-spatial 4D visualization</span>
                    </div>
                  </SelectItem>
                </SelectContent>
              </Select>
            </CardContent>
          </Card>

          {/* Isolation Layers */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Layers className="h-5 w-5" />
                Isolation Layers
              </CardTitle>
              <CardDescription>
                Visual focus and context layers
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {isolationLayers.map(layer => (
                  <div key={layer.id} className="flex items-center justify-between p-3 rounded-lg border">
                    <div className="flex items-center gap-3">
                      <div 
                        className="w-4 h-4 rounded"
                        style={{ 
                          backgroundColor: layer.color,
                          opacity: layer.opacity,
                        }}
                      />
                      <div>
                        <Label htmlFor={layer.id}>{layer.name}</Label>
                        <p className="text-xs text-muted-foreground">
                          {layer.nodeCount} nodes • {(layer.opacity * 100).toFixed(0)}% opacity
                        </p>
                      </div>
                    </div>
                    <Switch
                      id={layer.id}
                      checked={layer.active}
                      onCheckedChange={() => handleLayerToggle(layer.id)}
                    />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Forces Tab */}
        <TabsContent value="forces" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Force Parameters
              </CardTitle>
              <CardDescription>
                Fine-tune physics simulation parameters
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label htmlFor="repulsion">Repulsion</Label>
                  <span className="text-sm text-muted-foreground">{forceParams.repulsion.toFixed(1)}</span>
                </div>
                <Slider
                  id="repulsion"
                  min={0.1}
                  max={20}  // Reasonable range for stable physics
                  step={0.1}
                  value={[forceParams.repulsion]}
                  onValueChange={([v]) => handleForceParamChange('repulsion', v)}
                />
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label htmlFor="attraction">Attraction</Label>
                  <span className="text-sm text-muted-foreground">{forceParams.attraction.toFixed(3)}</span>
                </div>
                <Slider
                  id="attraction"
                  min={0}
                  max={0.01}  // Subtle attraction range
                  step={0.0001}
                  value={[forceParams.attraction]}
                  onValueChange={([v]) => handleForceParamChange('attraction', v)}
                />
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label htmlFor="damping">Damping</Label>
                  <span className="text-sm text-muted-foreground">{forceParams.damping.toFixed(2)}</span>
                </div>
                <Slider
                  id="damping"
                  min={0.5}  // Allow more range for experimentation
                  max={0.99}
                  step={0.01}
                  value={[forceParams.damping]}
                  onValueChange={([v]) => handleForceParamChange('damping', v)}
                />
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label htmlFor="temperature">Temperature</Label>
                  <span className="text-sm text-muted-foreground">{forceParams.temperature.toFixed(2)}</span>
                </div>
                <Slider
                  id="temperature"
                  min={0}
                  max={0.5}  // Low temperature for stability
                  step={0.01}
                  value={[forceParams.temperature]}
                  onValueChange={([v]) => handleForceParamChange('temperature', v)}
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label htmlFor="gravity">Gravity</Label>
                  <span className="text-sm text-muted-foreground">{forceParams.gravity.toFixed(2)}</span>
                </div>
                <Slider
                  id="gravity"
                  min={0}
                  max={0.01}  // Very subtle gravity
                  step={0.0001}
                  value={[forceParams.gravity]}
                  onValueChange={([v]) => handleForceParamChange('gravity', v)}
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label htmlFor="maxVelocity">Max Velocity</Label>
                  <span className="text-sm text-muted-foreground">{forceParams.maxVelocity.toFixed(1)}</span>
                </div>
                <Slider
                  id="maxVelocity"
                  min={0.5}
                  max={10}   // Allow reasonable movement speed
                  step={0.1}
                  value={[forceParams.maxVelocity]}
                  onValueChange={([v]) => handleForceParamChange('maxVelocity', v)}
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label htmlFor="timeStep">Time Step</Label>
                  <span className="text-sm text-muted-foreground">{forceParams.timeStep?.toFixed(3) || '0.150'}</span>
                </div>
                <Slider
                  id="timeStep"
                  min={0.005}
                  max={0.05}   // Standard range for simulation timestep
                  step={0.001}
                  value={[forceParams.timeStep || 0.016]}
                  onValueChange={([v]) => handleForceParamChange('timeStep', v)}
                />
              </div>
            </CardContent>
          </Card>

          {/* Trajectory Visualization */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Trajectory Visualization
              </CardTitle>
              <CardDescription>
                Show node movement trails
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <Label htmlFor="trajectory-enabled">Enable Trajectories</Label>
                  <Switch
                    id="trajectory-enabled"
                    checked={trajectorySettings.enabled}
                    onCheckedChange={(enabled) => 
                      setTrajectorySettings({ ...trajectorySettings, enabled })
                    }
                  />
                </div>
                
                {trajectorySettings.enabled && (
                  <>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <Label htmlFor="trajectory-length">Trail Length</Label>
                        <span className="text-sm text-muted-foreground">{trajectorySettings.length}</span>
                      </div>
                      <Slider
                        id="trajectory-length"
                        min={2}
                        max={16}
                        step={1}
                        value={[trajectorySettings.length]}
                        onValueChange={([v]) => 
                          setTrajectorySettings({ ...trajectorySettings, length: v })
                        }
                      />
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <Label htmlFor="color-velocity">Color by Velocity</Label>
                      <Switch
                        id="color-velocity"
                        checked={trajectorySettings.colorByVelocity}
                        onCheckedChange={(colorByVelocity) => 
                          setTrajectorySettings({ ...trajectorySettings, colorByVelocity })
                        }
                      />
                    </div>
                  </>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Constraints Tab */}
        <TabsContent value="constraints" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <GitBranch className="h-5 w-5" />
                Layout Constraints
              </CardTitle>
              <CardDescription>
                Enable and configure layout constraints
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {constraints.map(constraint => (
                  <div key={constraint.id} className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="text-lg">{constraint.icon}</span>
                      <div>
                        <Label htmlFor={constraint.id}>{constraint.name}</Label>
                        <p className="text-xs text-muted-foreground">{constraint.description}</p>
                      </div>
                    </div>
                    <Switch
                      id={constraint.id}
                      checked={constraint.enabled}
                      onCheckedChange={() => handleConstraintToggle(constraint.id)}
                    />
                  </div>
                ))}
              </div>
              
              <div className="mt-4 pt-4 border-t">
                <Button 
                  onClick={() => setShowConstraintBuilder(true)}
                  className="w-full"
                  variant="outline"
                >
                  <Plus className="mr-2 h-4 w-4" />
                  Create Custom Constraint
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Analytics Tab */}
        <TabsContent value="analytics" className="space-y-4">
          <SemanticClusteringControls />
        </TabsContent>
      </Tabs>

      {/* Constraint Builder Dialog */}
      <ConstraintBuilderDialog
        isOpen={showConstraintBuilder}
        onClose={() => setShowConstraintBuilder(false)}
        onSave={handleSaveConstraint}
      />
    </div>
  );
}