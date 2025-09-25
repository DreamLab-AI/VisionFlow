import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { Button } from '@/features/design-system/components/Button';
import { Badge } from '@/features/design-system/components/Badge';
import { 
  Sparkles, 
  Zap, 
  Target, 
  Trees, 
  Network, 
  Globe,
  Layers,
  Save,
  Upload,
  Download,
  Info
} from 'lucide-react';
import { useToast } from '@/features/design-system/components/Toast';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/features/design-system/components/Tooltip';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/features/design-system/components/Dialog';
import { Input } from '@/features/design-system/components/Input';
import { Label } from '@/features/design-system/components/Label';
import { unifiedApiClient } from '../../../services/api';

interface Preset {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  config: {
    kernelMode: string;
    forceParams: Record<string, number>;
    constraints: string[];
    layers: string[];
    trajectories: boolean;
  };
  tags: string[];
}

export function PhysicsPresets() {
  const { toast } = useToast();
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [customPresetName, setCustomPresetName] = useState('');
  const [customPresetDescription, setCustomPresetDescription] = useState('');

  const presets: Preset[] = [
    {
      id: 'default',
      name: 'Balanced',
      description: 'Default balanced layout for general use',
      icon: <Sparkles className="h-4 w-4" />,
      config: {
        kernelMode: 'visual_analytics',
        forceParams: {
          repulsion: 100,
          attraction: 0.01,
          damping: 0.95,
          temperature: 1.0,
        },
        constraints: ['separation', 'boundary', 'collision'],
        layers: [],
        trajectories: false,
      },
      tags: ['general', 'stable'],
    },
    {
      id: 'performance',
      name: 'High Performance',
      description: 'Optimized for large graphs with minimal effects',
      icon: <Zap className="h-4 w-4" />,
      config: {
        kernelMode: 'legacy',
        forceParams: {
          repulsion: 150,
          attraction: 0.005,
          damping: 0.98,
          temperature: 0.5,
        },
        constraints: ['boundary'],
        layers: [],
        trajectories: false,
      },
      tags: ['fast', 'large-graphs'],
    },
    {
      id: 'analytical',
      name: 'Deep Analysis',
      description: 'Advanced analytics with clustering and anomaly detection',
      icon: <Target className="h-4 w-4" />,
      config: {
        kernelMode: 'advanced',
        forceParams: {
          repulsion: 80,
          attraction: 0.02,
          damping: 0.92,
          temperature: 2.0,
        },
        constraints: ['separation', 'cluster', 'layer'],
        layers: ['focus', 'context'],
        trajectories: true,
      },
      tags: ['analytics', 'clustering'],
    },
    {
      id: 'hierarchical',
      name: 'Tree Layout',
      description: 'Hierarchical tree structure for organizational data',
      icon: <Trees className="h-4 w-4" />,
      config: {
        kernelMode: 'visual_analytics',
        forceParams: {
          repulsion: 200,
          attraction: 0.001,
          damping: 0.99,
          temperature: 0.1,
        },
        constraints: ['tree', 'alignment_v', 'layer'],
        layers: [],
        trajectories: false,
      },
      tags: ['hierarchy', 'tree'],
    },
    {
      id: 'network',
      name: 'Network Flow',
      description: 'Optimized for network topology visualization',
      icon: <Network className="h-4 w-4" />,
      config: {
        kernelMode: 'visual_analytics',
        forceParams: {
          repulsion: 120,
          attraction: 0.015,
          damping: 0.93,
          temperature: 1.5,
        },
        constraints: ['separation', 'radial', 'collision'],
        layers: ['focus'],
        trajectories: true,
      },
      tags: ['network', 'flow'],
    },
    {
      id: 'geographic',
      name: 'Geographic',
      description: 'Spatial layout with fixed positions and regions',
      icon: <Globe className="h-4 w-4" />,
      config: {
        kernelMode: 'visual_analytics',
        forceParams: {
          repulsion: 50,
          attraction: 0.03,
          damping: 0.85,
          temperature: 0.5,
        },
        constraints: ['fixed', 'boundary', 'cluster'],
        layers: ['background'],
        trajectories: false,
      },
      tags: ['spatial', 'geographic'],
    },
  ];

  const applyPreset = async (preset: Preset) => {
    try {
      await unifiedApiClient.post('/api/analytics/preset', preset.config);

      toast({
        title: 'Preset Applied',
        description: `${preset.name} configuration loaded`,
      });
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to apply preset',
        variant: 'destructive',
      });
    }
  };

  const saveCustomPreset = async () => {
    if (!customPresetName) {
      toast({
        title: 'Error',
        description: 'Please enter a preset name',
        variant: 'destructive',
      });
      return;
    }

    try {
      await unifiedApiClient.post('/api/analytics/preset/save', {
        name: customPresetName,
        description: customPresetDescription,
      });

      toast({
        title: 'Preset Saved',
        description: `"${customPresetName}" has been saved`,
      });
      setShowSaveDialog(false);
      setCustomPresetName('');
      setCustomPresetDescription('');
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to save preset',
        variant: 'destructive',
      });
    }
  };

  const exportPresets = async () => {
    try {
      const response = await unifiedApiClient.get('/api/analytics/preset/export');

      // Convert response data to blob for download
      const blob = new Blob([JSON.stringify(response.data)], { type: 'application/json' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'physics-presets.json';
      a.click();
      window.URL.revokeObjectURL(url);

      toast({
        title: 'Presets Exported',
        description: 'Download started',
      });
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to export presets',
        variant: 'destructive',
      });
    }
  };

  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers className="h-5 w-5" />
            Presets
          </CardTitle>
          <CardDescription>
            Quick configurations for common scenarios
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {presets.map(preset => (
              <div
                key={preset.id}
                className="p-3 rounded-lg border hover:bg-muted/50 transition-colors cursor-pointer"
                onClick={() => applyPreset(preset)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-3">
                    <div className="mt-1">{preset.icon}</div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{preset.name}</span>
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Info className="h-3 w-3 text-muted-foreground" />
                            </TooltipTrigger>
                            <TooltipContent>
                              <div className="max-w-xs">
                                <p className="font-medium mb-1">Configuration:</p>
                                <ul className="text-xs space-y-1">
                                  <li>• Kernel: {preset.config.kernelMode}</li>
                                  <li>• Constraints: {preset.config.constraints.length}</li>
                                  <li>• Layers: {preset.config.layers.length || 'None'}</li>
                                  <li>• Trajectories: {preset.config.trajectories ? 'Yes' : 'No'}</li>
                                </ul>
                              </div>
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      </div>
                      <p className="text-sm text-muted-foreground mt-1">
                        {preset.description}
                      </p>
                      <div className="flex gap-1 mt-2">
                        {preset.tags.map(tag => (
                          <Badge key={tag} variant="secondary" className="text-xs">
                            {tag}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          <div className="flex gap-2 mt-4 pt-4 border-t">
            <Button
              variant="outline"
              size="sm"
              className="flex-1"
              onClick={() => setShowSaveDialog(true)}
            >
              <Save className="mr-2 h-4 w-4" />
              Save Current
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="flex-1"
              onClick={exportPresets}
            >
              <Download className="mr-2 h-4 w-4" />
              Export
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="flex-1"
            >
              <Upload className="mr-2 h-4 w-4" />
              Import
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Save Preset Dialog */}
      <Dialog open={showSaveDialog} onOpenChange={setShowSaveDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Save Current Configuration</DialogTitle>
            <DialogDescription>
              Save your current physics engine configuration as a preset
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="preset-name">Preset Name</Label>
              <Input
                id="preset-name"
                value={customPresetName}
                onChange={(e) => setCustomPresetName(e.target.value)}
                placeholder="My Custom Preset"
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="preset-description">Description</Label>
              <Input
                id="preset-description"
                value={customPresetDescription}
                onChange={(e) => setCustomPresetDescription(e.target.value)}
                placeholder="Describe when to use this preset"
              />
            </div>
          </div>
          
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowSaveDialog(false)}>
              Cancel
            </Button>
            <Button onClick={saveCustomPreset}>
              Save Preset
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}