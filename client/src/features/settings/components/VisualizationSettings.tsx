import React, { useCallback, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { Slider } from '@/features/design-system/components/Slider';
import { Switch } from '@/features/design-system/components/Switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/features/design-system/components/Select';
import { Label } from '@/features/design-system/components/Label';
import { ColorPicker } from '@/features/design-system/components/ColorPicker';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/features/design-system/components/Tabs';
import Eye from 'lucide-react/dist/esm/icons/eye';
import Palette from 'lucide-react/dist/esm/icons/palette';
import Settings2 from 'lucide-react/dist/esm/icons/settings2';
import Sparkles from 'lucide-react/dist/esm/icons/sparkles';
import Type from 'lucide-react/dist/esm/icons/type';
import { useToast } from '@/features/design-system/components/Toast';
import { useSettingsStore } from '../../../store/settingsStore';

// Validation bounds for settings
const BOUNDS = {
  nodes: {
    nodeSize: { min: 0.1, max: 10, step: 0.1 },
    metalness: { min: 0, max: 1, step: 0.05 },
    roughness: { min: 0, max: 1, step: 0.05 },
    opacity: { min: 0, max: 1, step: 0.05 },
  },
  edges: {
    baseWidth: { min: 0.1, max: 5, step: 0.1 },
    arrowSize: { min: 0, max: 0.5, step: 0.01 },
    opacity: { min: 0, max: 1, step: 0.05 },
  },
  labels: {
    desktopFontSize: { min: 0.1, max: 5, step: 0.1 },
    textOutlineWidth: { min: 0, max: 0.1, step: 0.001 },
    textPadding: { min: 0, max: 2, step: 0.1 },
    textResolution: { min: 16, max: 256, step: 16 },
  },
  rendering: {
    ambientLightIntensity: { min: 0, max: 5, step: 0.1 },
    directionalLightIntensity: { min: 0, max: 5, step: 0.1 },
    environmentIntensity: { min: 0, max: 2, step: 0.1 },
  },
  glow: {
    intensity: { min: 0, max: 5, step: 0.1 },
    radius: { min: 0, max: 2, step: 0.05 },
    threshold: { min: 0, max: 1, step: 0.05 },
    diffuseStrength: { min: 0, max: 5, step: 0.1 },
    atmosphericDensity: { min: 0, max: 2, step: 0.1 },
    volumetricIntensity: { min: 0, max: 3, step: 0.1 },
    opacity: { min: 0, max: 1, step: 0.05 },
    pulseSpeed: { min: 0, max: 5, step: 0.1 },
    flowSpeed: { min: 0, max: 5, step: 0.1 },
    nodeGlowStrength: { min: 0, max: 10, step: 0.1 },
    edgeGlowStrength: { min: 0, max: 10, step: 0.1 },
    environmentGlowStrength: { min: 0, max: 10, step: 0.1 },
  },
  hologram: {
    ringCount: { min: 0, max: 20, step: 1 },
  }
};

export function VisualizationSettings() {
  const { toast } = useToast();
  const {
    settings,
    loading,
    updateSettings,
    loadSettings
  } = useSettingsStore();
  
  const currentGraph = 'logseq' as const;

  // Load settings on mount
  useEffect(() => {
    if (!settings) {
      loadSettings();
    }
  }, [settings, loadSettings]);

  if (loading || !settings) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-muted-foreground">Loading settings...</div>
      </div>
    );
  }

  const graphSettings = settings.visualisation.graphs[currentGraph];
  const globalSettings = settings.visualisation;

  const handleNodeUpdate = useCallback(async (key: string, value: any) => {
    try {
      await updateSettings({
        visualisation: {
          graphs: {
            [currentGraph]: {
              nodes: { [key]: value }
            }
          }
        }
      });
      toast({
        title: 'Node Settings Updated',
        description: `Updated ${key} to ${value}`,
      });
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to update node settings',
        variant: 'destructive',
      });
    }
  }, [updateSettings, currentGraph, toast]);

  const handleEdgeUpdate = useCallback(async (key: string, value: any) => {
    try {
      await updateSettings({
        visualisation: {
          graphs: {
            [currentGraph]: {
              edges: { [key]: value }
            }
          }
        }
      });
      toast({
        title: 'Edge Settings Updated',
        description: `Updated ${key} to ${value}`,
      });
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to update edge settings',
        variant: 'destructive',
      });
    }
  }, [updateSettings, currentGraph, toast]);

  const handleLabelUpdate = useCallback(async (key: string, value: any) => {
    try {
      await updateSettings({
        visualisation: {
          graphs: {
            [currentGraph]: {
              labels: { [key]: value }
            }
          }
        }
      });
      toast({
        title: 'Label Settings Updated',
        description: `Updated ${key} to ${value}`,
      });
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to update label settings',
        variant: 'destructive',
      });
    }
  }, [updateSettings, currentGraph, toast]);

  const handleGlobalUpdate = useCallback(async (section: string, key: string, value: any) => {
    try {
      await updateSettings({
        visualisation: {
          [section]: {
            ...globalSettings[section],
            [key]: value,
          }
        }
      });
      toast({
        title: 'Settings Updated',
        description: `Updated ${section}.${key} to ${value}`,
      });
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to update settings',
        variant: 'destructive',
      });
    }
  }, [updateSettings, globalSettings, toast]);

  return (
    <div className="h-full overflow-auto">
      <Tabs defaultValue="nodes" className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="nodes">Nodes</TabsTrigger>
          <TabsTrigger value="edges">Edges</TabsTrigger>
          <TabsTrigger value="labels">Labels</TabsTrigger>
          <TabsTrigger value="rendering">Rendering</TabsTrigger>
          <TabsTrigger value="effects">Effects</TabsTrigger>
        </TabsList>

        {/* Nodes Tab */}
        <TabsContent value="nodes" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings2 className="h-5 w-5" />
                Node Appearance
              </CardTitle>
              <CardDescription>
                Configure how nodes appear in the graph
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label htmlFor="node-size">Node Size</Label>
                  <span className="text-sm text-muted-foreground">{graphSettings.nodes.nodeSize}</span>
                </div>
                <Slider
                  id="node-size"
                  min={BOUNDS.nodes.nodeSize.min}
                  max={BOUNDS.nodes.nodeSize.max}
                  step={BOUNDS.nodes.nodeSize.step}
                  value={[graphSettings.nodes.nodeSize]}
                  onValueChange={([v]) => handleNodeUpdate('nodeSize', v)}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="base-color">Base Color</Label>
                <div className="flex gap-2">
                  <input
                    type="color"
                    id="base-color"
                    value={graphSettings.nodes.baseColor}
                    onChange={(e) => handleNodeUpdate('baseColor', e.target.value)}
                    className="h-10 w-20"
                  />
                  <input
                    type="text"
                    value={graphSettings.nodes.baseColor}
                    onChange={(e) => handleNodeUpdate('baseColor', e.target.value)}
                    className="flex-1 px-3 py-2 border rounded-md"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label htmlFor="metalness">Metalness</Label>
                  <span className="text-sm text-muted-foreground">{graphSettings.nodes.metalness}</span>
                </div>
                <Slider
                  id="metalness"
                  min={BOUNDS.nodes.metalness.min}
                  max={BOUNDS.nodes.metalness.max}
                  step={BOUNDS.nodes.metalness.step}
                  value={[graphSettings.nodes.metalness]}
                  onValueChange={([v]) => handleNodeUpdate('metalness', v)}
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label htmlFor="roughness">Roughness</Label>
                  <span className="text-sm text-muted-foreground">{graphSettings.nodes.roughness}</span>
                </div>
                <Slider
                  id="roughness"
                  min={BOUNDS.nodes.roughness.min}
                  max={BOUNDS.nodes.roughness.max}
                  step={BOUNDS.nodes.roughness.step}
                  value={[graphSettings.nodes.roughness]}
                  onValueChange={([v]) => handleNodeUpdate('roughness', v)}
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label htmlFor="opacity">Opacity</Label>
                  <span className="text-sm text-muted-foreground">{graphSettings.nodes.opacity}</span>
                </div>
                <Slider
                  id="opacity"
                  min={BOUNDS.nodes.opacity.min}
                  max={BOUNDS.nodes.opacity.max}
                  step={BOUNDS.nodes.opacity.step}
                  value={[graphSettings.nodes.opacity]}
                  onValueChange={([v]) => handleNodeUpdate('opacity', v)}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="quality">Quality</Label>
                <Select 
                  value={graphSettings.nodes.quality} 
                  onValueChange={(v) => handleNodeUpdate('quality', v)}
                >
                  <SelectTrigger id="quality">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="low">Low</SelectItem>
                    <SelectItem value="medium">Medium</SelectItem>
                    <SelectItem value="high">High</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-center justify-between">
                <Label htmlFor="enable-instancing">Enable Instancing</Label>
                <Switch
                  id="enable-instancing"
                  checked={graphSettings.nodes.enableInstancing}
                  onCheckedChange={(v) => handleNodeUpdate('enableInstancing', v)}
                />
              </div>

              <div className="flex items-center justify-between">
                <Label htmlFor="enable-hologram">Enable Hologram</Label>
                <Switch
                  id="enable-hologram"
                  checked={graphSettings.nodes.enableHologram}
                  onCheckedChange={(v) => handleNodeUpdate('enableHologram', v)}
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Edges Tab */}
        <TabsContent value="edges" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings2 className="h-5 w-5" />
                Edge Configuration
              </CardTitle>
              <CardDescription>
                Configure how edges appear between nodes
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label htmlFor="base-width">Base Width</Label>
                  <span className="text-sm text-muted-foreground">{graphSettings.edges.baseWidth}</span>
                </div>
                <Slider
                  id="base-width"
                  min={BOUNDS.edges.baseWidth.min}
                  max={BOUNDS.edges.baseWidth.max}
                  step={BOUNDS.edges.baseWidth.step}
                  value={[graphSettings.edges.baseWidth]}
                  onValueChange={([v]) => handleEdgeUpdate('baseWidth', v)}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="edge-color">Edge Color</Label>
                <div className="flex gap-2">
                  <input
                    type="color"
                    id="edge-color"
                    value={graphSettings.edges.color}
                    onChange={(e) => handleEdgeUpdate('color', e.target.value)}
                    className="h-10 w-20"
                  />
                  <input
                    type="text"
                    value={graphSettings.edges.color}
                    onChange={(e) => handleEdgeUpdate('color', e.target.value)}
                    className="flex-1 px-3 py-2 border rounded-md"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label htmlFor="edge-opacity">Opacity</Label>
                  <span className="text-sm text-muted-foreground">{graphSettings.edges.opacity}</span>
                </div>
                <Slider
                  id="edge-opacity"
                  min={BOUNDS.edges.opacity.min}
                  max={BOUNDS.edges.opacity.max}
                  step={BOUNDS.edges.opacity.step}
                  value={[graphSettings.edges.opacity]}
                  onValueChange={([v]) => handleEdgeUpdate('opacity', v)}
                />
              </div>

              <div className="flex items-center justify-between">
                <Label htmlFor="enable-arrows">Enable Arrows</Label>
                <Switch
                  id="enable-arrows"
                  checked={graphSettings.edges.enableArrows}
                  onCheckedChange={(v) => handleEdgeUpdate('enableArrows', v)}
                />
              </div>

              {graphSettings.edges.enableArrows && (
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <Label htmlFor="arrow-size">Arrow Size</Label>
                    <span className="text-sm text-muted-foreground">{graphSettings.edges.arrowSize}</span>
                  </div>
                  <Slider
                    id="arrow-size"
                    min={BOUNDS.edges.arrowSize.min}
                    max={BOUNDS.edges.arrowSize.max}
                    step={BOUNDS.edges.arrowSize.step}
                    value={[graphSettings.edges.arrowSize]}
                    onValueChange={([v]) => handleEdgeUpdate('arrowSize', v)}
                  />
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Labels Tab */}
        <TabsContent value="labels" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Type className="h-5 w-5" />
                Label Settings
              </CardTitle>
              <CardDescription>
                Configure text labels for nodes
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <Label htmlFor="enable-labels">Enable Labels</Label>
                <Switch
                  id="enable-labels"
                  checked={graphSettings.labels.enableLabels}
                  onCheckedChange={(v) => handleLabelUpdate('enableLabels', v)}
                />
              </div>

              {graphSettings.labels.enableLabels && (
                <>
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <Label htmlFor="font-size">Font Size</Label>
                      <span className="text-sm text-muted-foreground">{graphSettings.labels.desktopFontSize}</span>
                    </div>
                    <Slider
                      id="font-size"
                      min={BOUNDS.labels.desktopFontSize.min}
                      max={BOUNDS.labels.desktopFontSize.max}
                      step={BOUNDS.labels.desktopFontSize.step}
                      value={[graphSettings.labels.desktopFontSize]}
                      onValueChange={([v]) => handleLabelUpdate('desktopFontSize', v)}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="text-color">Text Color</Label>
                    <div className="flex gap-2">
                      <input
                        type="color"
                        id="text-color"
                        value={graphSettings.labels.textColor}
                        onChange={(e) => handleLabelUpdate('textColor', e.target.value)}
                        className="h-10 w-20"
                      />
                      <input
                        type="text"
                        value={graphSettings.labels.textColor}
                        onChange={(e) => handleLabelUpdate('textColor', e.target.value)}
                        className="flex-1 px-3 py-2 border rounded-md"
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <Label htmlFor="text-padding">Text Padding</Label>
                      <span className="text-sm text-muted-foreground">{graphSettings.labels.textPadding}</span>
                    </div>
                    <Slider
                      id="text-padding"
                      min={BOUNDS.labels.textPadding.min}
                      max={BOUNDS.labels.textPadding.max}
                      step={BOUNDS.labels.textPadding.step}
                      value={[graphSettings.labels.textPadding]}
                      onValueChange={([v]) => handleLabelUpdate('textPadding', v)}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="billboard-mode">Billboard Mode</Label>
                    <Select 
                      value={graphSettings.labels.billboardMode} 
                      onValueChange={(v) => handleLabelUpdate('billboardMode', v)}
                    >
                      <SelectTrigger id="billboard-mode">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="camera">Camera</SelectItem>
                        <SelectItem value="vertical">Vertical</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Rendering Tab */}
        <TabsContent value="rendering" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Eye className="h-5 w-5" />
                Rendering Settings
              </CardTitle>
              <CardDescription>
                Global rendering configuration
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="bg-color">Background Color</Label>
                <div className="flex gap-2">
                  <input
                    type="color"
                    id="bg-color"
                    value={globalSettings.rendering.backgroundColor}
                    onChange={(e) => handleGlobalUpdate('rendering', 'backgroundColor', e.target.value)}
                    className="h-10 w-20"
                  />
                  <input
                    type="text"
                    value={globalSettings.rendering.backgroundColor}
                    onChange={(e) => handleGlobalUpdate('rendering', 'backgroundColor', e.target.value)}
                    className="flex-1 px-3 py-2 border rounded-md"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label htmlFor="ambient-light">Ambient Light Intensity</Label>
                  <span className="text-sm text-muted-foreground">{globalSettings.rendering.ambientLightIntensity}</span>
                </div>
                <Slider
                  id="ambient-light"
                  min={BOUNDS.rendering.ambientLightIntensity.min}
                  max={BOUNDS.rendering.ambientLightIntensity.max}
                  step={BOUNDS.rendering.ambientLightIntensity.step}
                  value={[globalSettings.rendering.ambientLightIntensity]}
                  onValueChange={([v]) => handleGlobalUpdate('rendering', 'ambientLightIntensity', v)}
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label htmlFor="directional-light">Directional Light Intensity</Label>
                  <span className="text-sm text-muted-foreground">{globalSettings.rendering.directionalLightIntensity}</span>
                </div>
                <Slider
                  id="directional-light"
                  min={BOUNDS.rendering.directionalLightIntensity.min}
                  max={BOUNDS.rendering.directionalLightIntensity.max}
                  step={BOUNDS.rendering.directionalLightIntensity.step}
                  value={[globalSettings.rendering.directionalLightIntensity]}
                  onValueChange={([v]) => handleGlobalUpdate('rendering', 'directionalLightIntensity', v)}
                />
              </div>

              <div className="flex items-center justify-between">
                <Label htmlFor="antialiasing">Enable Antialiasing</Label>
                <Switch
                  id="antialiasing"
                  checked={globalSettings.rendering.enableAntialiasing}
                  onCheckedChange={(v) => handleGlobalUpdate('rendering', 'enableAntialiasing', v)}
                />
              </div>

              <div className="flex items-center justify-between">
                <Label htmlFor="shadows">Enable Shadows</Label>
                <Switch
                  id="shadows"
                  checked={globalSettings.rendering.enableShadows}
                  onCheckedChange={(v) => handleGlobalUpdate('rendering', 'enableShadows', v)}
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Effects Tab */}
        <TabsContent value="effects" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="h-5 w-5" />
                Atmospheric Glow Effects
              </CardTitle>
              <CardDescription>
                Configure diffuse atmospheric glow and volumetric effects
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <Label htmlFor="glow-enabled">Enable Glow</Label>
                <Switch
                  id="glow-enabled"
                  checked={globalSettings.glow.enabled}
                  onCheckedChange={(v) => handleGlobalUpdate('glow', 'enabled', v)}
                />
              </div>

              {globalSettings.glow.enabled && (
                <>
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <Label htmlFor="glow-intensity">Intensity</Label>
                      <span className="text-sm text-muted-foreground">{globalSettings.glow.intensity}</span>
                    </div>
                    <Slider
                      id="glow-intensity"
                      min={BOUNDS.glow.intensity.min}
                      max={BOUNDS.glow.intensity.max}
                      step={BOUNDS.glow.intensity.step}
                      value={[globalSettings.glow.intensity]}
                      onValueChange={([v]) => handleGlobalUpdate('glow', 'intensity', v)}
                    />
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <Label htmlFor="glow-radius">Radius</Label>
                      <span className="text-sm text-muted-foreground">{globalSettings.glow.radius}</span>
                    </div>
                    <Slider
                      id="glow-radius"
                      min={BOUNDS.glow.radius.min}
                      max={BOUNDS.glow.radius.max}
                      step={BOUNDS.glow.radius.step}
                      value={[globalSettings.glow.radius]}
                      onValueChange={([v]) => handleGlobalUpdate('glow', 'radius', v)}
                    />
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <Label htmlFor="diffuse-strength">Diffuse Strength</Label>
                      <span className="text-sm text-muted-foreground">{globalSettings.glow.diffuseStrength}</span>
                    </div>
                    <Slider
                      id="diffuse-strength"
                      min={BOUNDS.glow.diffuseStrength.min}
                      max={BOUNDS.glow.diffuseStrength.max}
                      step={BOUNDS.glow.diffuseStrength.step}
                      value={[globalSettings.glow.diffuseStrength]}
                      onValueChange={([v]) => handleGlobalUpdate('glow', 'diffuseStrength', v)}
                    />
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <Label htmlFor="atmospheric-density">Atmospheric Density</Label>
                      <span className="text-sm text-muted-foreground">{globalSettings.glow.atmosphericDensity}</span>
                    </div>
                    <Slider
                      id="atmospheric-density"
                      min={BOUNDS.glow.atmosphericDensity.min}
                      max={BOUNDS.glow.atmosphericDensity.max}
                      step={BOUNDS.glow.atmosphericDensity.step}
                      value={[globalSettings.glow.atmosphericDensity]}
                      onValueChange={([v]) => handleGlobalUpdate('glow', 'atmosphericDensity', v)}
                    />
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <Label htmlFor="volumetric-intensity">Volumetric Intensity</Label>
                      <span className="text-sm text-muted-foreground">{globalSettings.glow.volumetricIntensity}</span>
                    </div>
                    <Slider
                      id="volumetric-intensity"
                      min={BOUNDS.glow.volumetricIntensity.min}
                      max={BOUNDS.glow.volumetricIntensity.max}
                      step={BOUNDS.glow.volumetricIntensity.step}
                      value={[globalSettings.glow.volumetricIntensity]}
                      onValueChange={([v]) => handleGlobalUpdate('glow', 'volumetricIntensity', v)}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="base-color">Base Color</Label>
                    <div className="flex gap-2">
                      <input
                        type="color"
                        id="base-color"
                        value={globalSettings.glow.baseColor}
                        onChange={(e) => handleGlobalUpdate('glow', 'baseColor', e.target.value)}
                        className="h-10 w-20"
                      />
                      <input
                        type="text"
                        value={globalSettings.glow.baseColor}
                        onChange={(e) => handleGlobalUpdate('glow', 'baseColor', e.target.value)}
                        className="flex-1 px-3 py-2 border rounded-md"
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="emission-color">Emission Color</Label>
                    <div className="flex gap-2">
                      <input
                        type="color"
                        id="emission-color"
                        value={globalSettings.glow.emissionColor}
                        onChange={(e) => handleGlobalUpdate('glow', 'emissionColor', e.target.value)}
                        className="h-10 w-20"
                      />
                      <input
                        type="text"
                        value={globalSettings.glow.emissionColor}
                        onChange={(e) => handleGlobalUpdate('glow', 'emissionColor', e.target.value)}
                        className="flex-1 px-3 py-2 border rounded-md"
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <Label htmlFor="node-glow">Node Glow Strength</Label>
                      <span className="text-sm text-muted-foreground">{globalSettings.glow.nodeGlowStrength}</span>
                    </div>
                    <Slider
                      id="node-glow"
                      min={BOUNDS.glow.nodeGlowStrength.min}
                      max={BOUNDS.glow.nodeGlowStrength.max}
                      step={BOUNDS.glow.nodeGlowStrength.step}
                      value={[globalSettings.glow.nodeGlowStrength]}
                      onValueChange={([v]) => handleGlobalUpdate('glow', 'nodeGlowStrength', v)}
                    />
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <Label htmlFor="edge-glow">Edge Glow Strength</Label>
                      <span className="text-sm text-muted-foreground">{globalSettings.glow.edgeGlowStrength}</span>
                    </div>
                    <Slider
                      id="edge-glow"
                      min={BOUNDS.glow.edgeGlowStrength.min}
                      max={BOUNDS.glow.edgeGlowStrength.max}
                      step={BOUNDS.glow.edgeGlowStrength.step}
                      value={[globalSettings.glow.edgeGlowStrength]}
                      onValueChange={([v]) => handleGlobalUpdate('glow', 'edgeGlowStrength', v)}
                    />
                  </div>
                </>
              )}
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="h-5 w-5" />
                Hologram Effects
              </CardTitle>
              <CardDescription>
                Configure hologram properties
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label htmlFor="ring-count">Ring Count</Label>
                  <span className="text-sm text-muted-foreground">{globalSettings.hologram.ringCount}</span>
                </div>
                <Slider
                  id="ring-count"
                  min={BOUNDS.hologram.ringCount.min}
                  max={BOUNDS.hologram.ringCount.max}
                  step={BOUNDS.hologram.ringCount.step}
                  value={[globalSettings.hologram.ringCount]}
                  onValueChange={([v]) => handleGlobalUpdate('hologram', 'ringCount', Math.round(v))}
                />
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="h-5 w-5" />
                Hologram Effects
              </CardTitle>
              <CardDescription>
                Configure hologram properties
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label htmlFor="ring-count">Ring Count</Label>
                  <span className="text-sm text-muted-foreground">{globalSettings.hologram.ringCount}</span>
                </div>
                <Slider
                  id="ring-count"
                  min={BOUNDS.hologram.ringCount.min}
                  max={BOUNDS.hologram.ringCount.max}
                  step={BOUNDS.hologram.ringCount.step}
                  value={[globalSettings.hologram.ringCount]}
                  onValueChange={([v]) => handleGlobalUpdate('hologram', 'ringCount', Math.round(v))}
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}