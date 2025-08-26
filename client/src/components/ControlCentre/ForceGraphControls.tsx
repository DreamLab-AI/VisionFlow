import React, { useCallback, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../features/design-system/components/Card';
import { Label } from '../../features/design-system/components/Label';
import { Slider } from '../../features/design-system/components/Slider';
import { Switch } from '../../features/design-system/components/Switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../../features/design-system/components/Select';
import { Button } from '../../features/design-system/components/Button';
import { useSettingsStore } from '../../store/settingsStore';
import { useToast } from '../../features/design-system/components/Toast';
import { Activity, GitBranch, Zap, Type, Palette } from 'lucide-react';

export const ForceGraphControls: React.FC = () => {
  const { toast } = useToast();
  const { settings, updateSettings, updateGPUPhysics } = useSettingsStore();
  const [activeGraph, setActiveGraph] = useState<'logseq' | 'visionflow'>('logseq');

  // Get current graph settings
  const graphSettings = settings?.visualisation?.graphs?.[activeGraph];
  const nodeSettings = graphSettings?.nodes || {};
  const edgeSettings = graphSettings?.edges || {};
  const labelSettings = graphSettings?.labels || {};
  const physicsSettings = graphSettings?.physics || {};

  const handleGraphChange = useCallback((graph: 'logseq' | 'visionflow') => {
    setActiveGraph(graph);
  }, []);

  // Node control handlers
  const handleNodeSizeChange = useCallback((size: number[]) => {
    const value = size[0];
    updateSettings((draft) => {
      if (!draft.visualisation?.graphs?.[activeGraph]?.nodes) return;
      draft.visualisation.graphs[activeGraph].nodes.nodeSize = value;
    });
  }, [updateSettings, activeGraph]);

  const handleNodeColorChange = useCallback((color: string) => {
    updateSettings((draft) => {
      if (!draft.visualisation?.graphs?.[activeGraph]?.nodes) return;
      draft.visualisation.graphs[activeGraph].nodes.baseColor = color;
    });
  }, [updateSettings, activeGraph]);

  const handleNodeOpacityChange = useCallback((opacity: number[]) => {
    const value = opacity[0];
    updateSettings((draft) => {
      if (!draft.visualisation?.graphs?.[activeGraph]?.nodes) return;
      draft.visualisation.graphs[activeGraph].nodes.opacity = value;
    });
  }, [updateSettings, activeGraph]);

  // Edge control handlers
  const handleEdgeWidthChange = useCallback((width: number[]) => {
    const value = width[0];
    updateSettings((draft) => {
      if (!draft.visualisation?.graphs?.[activeGraph]?.edges) return;
      draft.visualisation.graphs[activeGraph].edges.baseWidth = value;
    });
  }, [updateSettings, activeGraph]);

  const handleEdgeColorChange = useCallback((color: string) => {
    updateSettings((draft) => {
      if (!draft.visualisation?.graphs?.[activeGraph]?.edges) return;
      draft.visualisation.graphs[activeGraph].edges.color = color;
    });
  }, [updateSettings, activeGraph]);

  const handleEdgeOpacityChange = useCallback((opacity: number[]) => {
    const value = opacity[0];
    updateSettings((draft) => {
      if (!draft.visualisation?.graphs?.[activeGraph]?.edges) return;
      draft.visualisation.graphs[activeGraph].edges.opacity = value;
    });
  }, [updateSettings, activeGraph]);

  // Physics control handlers
  const handlePhysicsEnabledChange = useCallback((enabled: boolean) => {
    updateSettings((draft) => {
      if (!draft.visualisation?.graphs?.[activeGraph]?.physics) return;
      draft.visualisation.graphs[activeGraph].physics.enabled = enabled;
    });
  }, [updateSettings, activeGraph]);

  const handleSpringStrengthChange = useCallback((strength: number[]) => {
    const value = strength[0];
    updateGPUPhysics(activeGraph, { springK: value });
  }, [updateGPUPhysics, activeGraph]);

  const handleRepulsionStrengthChange = useCallback((strength: number[]) => {
    const value = strength[0];
    updateGPUPhysics(activeGraph, { repelK: value });
  }, [updateGPUPhysics, activeGraph]);

  const handleDampingChange = useCallback((damping: number[]) => {
    const value = damping[0];
    updateGPUPhysics(activeGraph, { damping: value });
  }, [updateGPUPhysics, activeGraph]);

  // Label control handlers
  const handleLabelsEnabledChange = useCallback((enabled: boolean) => {
    updateSettings((draft) => {
      if (!draft.visualisation?.graphs?.[activeGraph]?.labels) return;
      draft.visualisation.graphs[activeGraph].labels.enableLabels = enabled;
    });
  }, [updateSettings, activeGraph]);

  const handleLabelSizeChange = useCallback((size: number[]) => {
    const value = size[0];
    updateSettings((draft) => {
      if (!draft.visualisation?.graphs?.[activeGraph]?.labels) return;
      draft.visualisation.graphs[activeGraph].labels.desktopFontSize = value;
    });
  }, [updateSettings, activeGraph]);

  const handleResetPhysics = useCallback(() => {
    // Reset physics to sensible defaults for the current graph
    const defaultPhysics = {
      springK: activeGraph === 'logseq' ? 0.05 : 0.08,
      repelK: activeGraph === 'logseq' ? 2.0 : 1.5,
      damping: 0.9,
      attractionK: 0.1,
      maxVelocity: 5.0,
      dt: 0.02,
      temperature: 1.0,
      iterations: 100
    };

    updateGPUPhysics(activeGraph, defaultPhysics);
    
    toast({
      title: 'Physics Reset',
      description: `${activeGraph} graph physics reset to defaults`,
    });
  }, [activeGraph, updateGPUPhysics, toast]);

  return (
    <div className="space-y-4">
      {/* Graph Selection */}
      <Card className="bg-white/5 border-white/10">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <GitBranch className="h-4 w-4" />
            Active Graph
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Select value={activeGraph} onValueChange={handleGraphChange}>
            <SelectTrigger className="w-full bg-white/10 border-white/20 text-white">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-gray-900 border-white/20">
              <SelectItem value="logseq" className="text-white hover:bg-white/10">
                Logseq Graph
              </SelectItem>
              <SelectItem value="visionflow" className="text-white hover:bg-white/10">
                VisionFlow Graph
              </SelectItem>
            </SelectContent>
          </Select>
        </CardContent>
      </Card>

      {/* Node Controls */}
      <Card className="bg-white/5 border-white/10">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Palette className="h-4 w-4" />
            Node Appearance
          </CardTitle>
          <CardDescription className="text-xs text-white/60">
            Visual properties of graph nodes
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Node Size */}
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <Label className="text-xs font-medium">Node Size</Label>
              <span className="text-xs text-white/60">
                {(nodeSettings.nodeSize || 1).toFixed(2)}
              </span>
            </div>
            <Slider
              value={[nodeSettings.nodeSize || 1]}
              onValueChange={handleNodeSizeChange}
              min={0.1}
              max={5}
              step={0.05}
              className="w-full"
            />
          </div>

          {/* Node Color */}
          <div className="space-y-2">
            <Label className="text-xs font-medium">Node Color</Label>
            <div className="flex gap-2">
              <input
                type="color"
                value={nodeSettings.baseColor || '#4A90E2'}
                onChange={(e) => handleNodeColorChange(e.target.value)}
                className="w-12 h-8 rounded border border-white/20 bg-transparent cursor-pointer"
              />
              <input
                type="text"
                value={nodeSettings.baseColor || '#4A90E2'}
                onChange={(e) => handleNodeColorChange(e.target.value)}
                className="flex-1 px-2 py-1 text-xs bg-white/10 border border-white/20 rounded text-white"
              />
            </div>
          </div>

          {/* Node Opacity */}
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <Label className="text-xs font-medium">Node Opacity</Label>
              <span className="text-xs text-white/60">
                {((nodeSettings.opacity || 1) * 100).toFixed(0)}%
              </span>
            </div>
            <Slider
              value={[nodeSettings.opacity || 1]}
              onValueChange={handleNodeOpacityChange}
              min={0.1}
              max={1}
              step={0.01}
              className="w-full"
            />
          </div>
        </CardContent>
      </Card>

      {/* Edge Controls */}
      <Card className="bg-white/5 border-white/10">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <GitBranch className="h-4 w-4" />
            Edge Properties
          </CardTitle>
          <CardDescription className="text-xs text-white/60">
            Connection lines between nodes
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Edge Width */}
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <Label className="text-xs font-medium">Edge Width</Label>
              <span className="text-xs text-white/60">
                {(edgeSettings.baseWidth || 0.5).toFixed(2)}
              </span>
            </div>
            <Slider
              value={[edgeSettings.baseWidth || 0.5]}
              onValueChange={handleEdgeWidthChange}
              min={0.01}
              max={5}
              step={0.01}
              className="w-full"
            />
          </div>

          {/* Edge Color */}
          <div className="space-y-2">
            <Label className="text-xs font-medium">Edge Color</Label>
            <div className="flex gap-2">
              <input
                type="color"
                value={edgeSettings.color || '#FFFFFF'}
                onChange={(e) => handleEdgeColorChange(e.target.value)}
                className="w-12 h-8 rounded border border-white/20 bg-transparent cursor-pointer"
              />
              <input
                type="text"
                value={edgeSettings.color || '#FFFFFF'}
                onChange={(e) => handleEdgeColorChange(e.target.value)}
                className="flex-1 px-2 py-1 text-xs bg-white/10 border border-white/20 rounded text-white"
              />
            </div>
          </div>

          {/* Edge Opacity */}
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <Label className="text-xs font-medium">Edge Opacity</Label>
              <span className="text-xs text-white/60">
                {((edgeSettings.opacity || 0.6) * 100).toFixed(0)}%
              </span>
            </div>
            <Slider
              value={[edgeSettings.opacity || 0.6]}
              onValueChange={handleEdgeOpacityChange}
              min={0.1}
              max={1}
              step={0.01}
              className="w-full"
            />
          </div>
        </CardContent>
      </Card>

      {/* Physics Controls */}
      <Card className="bg-white/5 border-white/10">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Activity className="h-4 w-4" />
            Force Physics
          </CardTitle>
          <CardDescription className="text-xs text-white/60">
            Physics simulation parameters
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Physics Enabled */}
          <div className="flex items-center justify-between">
            <Label className="text-xs font-medium">Enable Physics</Label>
            <Switch
              checked={physicsSettings.enabled || false}
              onCheckedChange={handlePhysicsEnabledChange}
            />
          </div>

          {physicsSettings.enabled && (
            <>
              {/* Spring Strength */}
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label className="text-xs font-medium">Spring Strength</Label>
                  <span className="text-xs text-white/60">
                    {(physicsSettings.springK || 0.05).toFixed(3)}
                  </span>
                </div>
                <Slider
                  value={[physicsSettings.springK || 0.05]}
                  onValueChange={handleSpringStrengthChange}
                  min={0.001}
                  max={2}
                  step={0.001}
                  className="w-full"
                />
              </div>

              {/* Repulsion Strength */}
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label className="text-xs font-medium">Repulsion Strength</Label>
                  <span className="text-xs text-white/60">
                    {(physicsSettings.repelK || 1.0).toFixed(2)}
                  </span>
                </div>
                <Slider
                  value={[physicsSettings.repelK || 1.0]}
                  onValueChange={handleRepulsionStrengthChange}
                  min={0.01}
                  max={10}
                  step={0.01}
                  className="w-full"
                />
              </div>

              {/* Damping */}
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label className="text-xs font-medium">Damping</Label>
                  <span className="text-xs text-white/60">
                    {(physicsSettings.damping || 0.9).toFixed(2)}
                  </span>
                </div>
                <Slider
                  value={[physicsSettings.damping || 0.9]}
                  onValueChange={handleDampingChange}
                  min={0.5}
                  max={0.99}
                  step={0.01}
                  className="w-full"
                />
              </div>

              {/* Reset Physics Button */}
              <Button
                variant="outline"
                size="sm"
                onClick={handleResetPhysics}
                className="w-full bg-transparent border-white/20 text-white hover:bg-white/10"
              >
                <Zap className="h-3 w-3 mr-1" />
                Reset Physics
              </Button>
            </>
          )}
        </CardContent>
      </Card>

      {/* Label Controls */}
      <Card className="bg-white/5 border-white/10">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Type className="h-4 w-4" />
            Node Labels
          </CardTitle>
          <CardDescription className="text-xs text-white/60">
            Text labels for graph nodes
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Labels Enabled */}
          <div className="flex items-center justify-between">
            <Label className="text-xs font-medium">Enable Labels</Label>
            <Switch
              checked={labelSettings.enableLabels || false}
              onCheckedChange={handleLabelsEnabledChange}
            />
          </div>

          {labelSettings.enableLabels && (
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <Label className="text-xs font-medium">Label Size</Label>
                <span className="text-xs text-white/60">
                  {(labelSettings.desktopFontSize || 1).toFixed(2)}
                </span>
              </div>
              <Slider
                value={[labelSettings.desktopFontSize || 1]}
                onValueChange={handleLabelSizeChange}
                min={0.1}
                max={3}
                step={0.1}
                className="w-full"
              />
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};