import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../../design-system/components/Card';
import { Label } from '../../design-system/components/Label';
import { Switch } from '../../design-system/components/Switch';
import { Input } from '../../design-system/components/Input';
import { Slider } from '../../design-system/components/Slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../../design-system/components/Select';
import { Button } from '../../design-system/components/Button';
import { Badge } from '../../design-system/components/Badge';
import { Eye, Camera, Sun, Lightbulb, Palette, Layers, Zap, RotateCcw } from 'lucide-react';
import { useSelectiveSetting, useSettingSetter } from '../../../hooks/useSelectiveSettingsStore';
import { useVisualizationSettings } from '../../../hooks/useVisualizationSettings';

/**
 * VisualizationSettings Settings Panel
 * Provides 3D visualization and rendering settings with selective access patterns
 */
export function VisualizationSettings() {
  const { set, batchSet } = useSettingSetter();
  const visualizationHook = useVisualizationSettings();
  
  // Use selective settings access for visualization-related settings
  const enabled = useSelectiveSetting<boolean>('visualisation.enabled') ?? true;
  const theme = useSelectiveSetting<string>('visualisation.theme') ?? 'dark';
  const quality = useSelectiveSetting<'low' | 'medium' | 'high' | 'ultra'>('visualisation.quality') ?? 'medium';
  const antialiasing = useSelectiveSetting<boolean>('visualisation.antialiasing') ?? true;
  const shadows = useSelectiveSetting<boolean>('visualisation.shadows') ?? true;
  const postProcessing = useSelectiveSetting<boolean>('visualisation.postProcessing') ?? true;
  const bloom = useSelectiveSetting<boolean>('visualisation.effects.bloom') ?? true;
  const ssao = useSelectiveSetting<boolean>('visualisation.effects.ssao') ?? false;
  
  // Camera settings
  const cameraFov = useSelectiveSetting<number>('visualisation.camera.fov') ?? 75;
  const cameraNear = useSelectiveSetting<number>('visualisation.camera.near') ?? 0.1;
  const cameraFar = useSelectiveSetting<number>('visualisation.camera.far') ?? 1000;
  const cameraPosition = useSelectiveSetting<[number, number, number]>('visualisation.camera.position') ?? [0, 0, 5];
  
  // Lighting settings
  const ambientIntensity = useSelectiveSetting<number>('visualisation.lighting.ambient.intensity') ?? 0.4;
  const directionalIntensity = useSelectiveSetting<number>('visualisation.lighting.directional.intensity') ?? 0.8;
  const directionalPosition = useSelectiveSetting<[number, number, number]>('visualisation.lighting.directional.position') ?? [10, 10, 5];
  
  // Rendering settings
  const pixelRatio = useSelectiveSetting<number>('visualisation.rendering.pixelRatio') ?? 1.0;
  const targetFPS = useSelectiveSetting<number>('visualisation.rendering.targetFPS') ?? 60;
  const adaptiveQuality = useSelectiveSetting<boolean>('visualisation.rendering.adaptiveQuality') ?? true;
  
  // Material settings
  const materialQuality = useSelectiveSetting<'low' | 'medium' | 'high'>('visualisation.materials.quality') ?? 'medium';
  const textureFiltering = useSelectiveSetting<'nearest' | 'linear' | 'trilinear'>('visualisation.materials.filtering') ?? 'linear';
  const anisotropicFiltering = useSelectiveSetting<number>('visualisation.materials.anisotropy') ?? 4;

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

  const applyQualityPreset = async (preset: string) => {
    const presets: Record<string, Record<string, any>> = {
      'low': {
        'visualisation.quality': 'low',
        'visualisation.antialiasing': false,
        'visualisation.shadows': false,
        'visualisation.postProcessing': false,
        'visualisation.effects.bloom': false,
        'visualisation.effects.ssao': false,
        'visualisation.rendering.pixelRatio': 0.75,
        'visualisation.rendering.targetFPS': 30,
        'visualisation.materials.quality': 'low',
        'visualisation.materials.filtering': 'nearest',
        'visualisation.materials.anisotropy': 1
      },
      'medium': {
        'visualisation.quality': 'medium',
        'visualisation.antialiasing': true,
        'visualisation.shadows': true,
        'visualisation.postProcessing': true,
        'visualisation.effects.bloom': true,
        'visualisation.effects.ssao': false,
        'visualisation.rendering.pixelRatio': 1.0,
        'visualisation.rendering.targetFPS': 60,
        'visualisation.materials.quality': 'medium',
        'visualisation.materials.filtering': 'linear',
        'visualisation.materials.anisotropy': 4
      },
      'high': {
        'visualisation.quality': 'high',
        'visualisation.antialiasing': true,
        'visualisation.shadows': true,
        'visualisation.postProcessing': true,
        'visualisation.effects.bloom': true,
        'visualisation.effects.ssao': true,
        'visualisation.rendering.pixelRatio': 1.0,
        'visualisation.rendering.targetFPS': 60,
        'visualisation.materials.quality': 'high',
        'visualisation.materials.filtering': 'trilinear',
        'visualisation.materials.anisotropy': 8
      },
      'ultra': {
        'visualisation.quality': 'ultra',
        'visualisation.antialiasing': true,
        'visualisation.shadows': true,
        'visualisation.postProcessing': true,
        'visualisation.effects.bloom': true,
        'visualisation.effects.ssao': true,
        'visualisation.rendering.pixelRatio': 1.5,
        'visualisation.rendering.targetFPS': 120,
        'visualisation.materials.quality': 'high',
        'visualisation.materials.filtering': 'trilinear',
        'visualisation.materials.anisotropy': 16
      }
    };

    if (presets[preset]) {
      await handleBatchChange(presets[preset]);
    }
  };

  const resetToDefaults = async () => {
    await handleBatchChange({
      'visualisation.enabled': true,
      'visualisation.theme': 'dark',
      'visualisation.quality': 'medium',
      'visualisation.antialiasing': true,
      'visualisation.shadows': true,
      'visualisation.postProcessing': true,
      'visualisation.effects.bloom': true,
      'visualisation.effects.ssao': false,
      'visualisation.camera.fov': 75,
      'visualisation.camera.near': 0.1,
      'visualisation.camera.far': 1000,
      'visualisation.camera.position': [0, 0, 5],
      'visualisation.lighting.ambient.intensity': 0.4,
      'visualisation.lighting.directional.intensity': 0.8,
      'visualisation.lighting.directional.position': [10, 10, 5],
      'visualisation.rendering.pixelRatio': 1.0,
      'visualisation.rendering.targetFPS': 60,
      'visualisation.rendering.adaptiveQuality': true,
      'visualisation.materials.quality': 'medium',
      'visualisation.materials.filtering': 'linear',
      'visualisation.materials.anisotropy': 4
    });
  };

  const qualityPresets = [
    { value: 'low', label: 'Low', icon: '🔋', description: 'Battery friendly' },
    { value: 'medium', label: 'Medium', icon: '⚖️', description: 'Balanced quality' },
    { value: 'high', label: 'High', icon: '✨', description: 'Enhanced visuals' },
    { value: 'ultra', label: 'Ultra', icon: '🚀', description: 'Maximum quality' }
  ];

  const themes = [
    { value: 'light', label: 'Light' },
    { value: 'dark', label: 'Dark' },
    { value: 'auto', label: 'Auto' }
  ];

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Eye className="w-5 h-5" />
              <CardTitle>3D Visualization Settings</CardTitle>
            </div>
            <Badge variant={enabled ? 'default' : 'secondary'}>
              {enabled ? 'Enabled' : 'Disabled'}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Master Toggle */}
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label className="text-sm font-medium">Enable 3D Visualization</Label>
              <p className="text-xs text-muted-foreground">
                Master toggle for 3D rendering and visualization features
              </p>
            </div>
            <Switch
              checked={enabled}
              onCheckedChange={(checked) => handleSettingChange('visualisation.enabled', checked)}
            />
          </div>

          {enabled && (
            <>
              {/* Quality Presets */}
              <div className="space-y-3">
                <div className="border-t pt-4">
                  <Label className="text-sm font-medium mb-3 block">Quality Presets</Label>
                  <div className="grid grid-cols-2 gap-2">
                    {qualityPresets.map((preset) => (
                      <Button
                        key={preset.value}
                        variant={quality === preset.value ? 'default' : 'outline'}
                        className="flex flex-col items-start gap-1 h-auto p-3 text-left"
                        onClick={() => applyQualityPreset(preset.value)}
                      >
                        <div className="flex items-center gap-2">
                          <span className="text-lg">{preset.icon}</span>
                          <span className="font-medium">{preset.label}</span>
                        </div>
                        <span className="text-xs text-muted-foreground">{preset.description}</span>
                      </Button>
                    ))}
                  </div>
                </div>
              </div>

              {/* General Settings */}
              <div className="space-y-4">
                <div className="border-t pt-4">
                  <h3 className="text-sm font-medium mb-3">General</h3>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label className="text-sm">Theme</Label>
                      <Select
                        value={theme}
                        onValueChange={(value) => handleSettingChange('visualisation.theme', value)}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {themes.map((theme) => (
                            <SelectItem key={theme.value} value={theme.value}>
                              {theme.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label className="text-sm">Adaptive Quality</Label>
                        <p className="text-xs text-muted-foreground">Auto-adjust quality for performance</p>
                      </div>
                      <Switch
                        checked={adaptiveQuality}
                        onCheckedChange={(checked) => handleSettingChange('visualisation.rendering.adaptiveQuality', checked)}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Rendering Quality */}
              <div className="space-y-4">
                <div className="border-t pt-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Layers className="w-4 h-4" />
                    <h3 className="text-sm font-medium">Rendering Quality</h3>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label className="text-sm">Anti-aliasing</Label>
                        <p className="text-xs text-muted-foreground">Smooth jagged edges</p>
                      </div>
                      <Switch
                        checked={antialiasing}
                        onCheckedChange={(checked) => handleSettingChange('visualisation.antialiasing', checked)}
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label className="text-sm">Shadows</Label>
                        <p className="text-xs text-muted-foreground">Dynamic shadow rendering</p>
                      </div>
                      <Switch
                        checked={shadows}
                        onCheckedChange={(checked) => handleSettingChange('visualisation.shadows', checked)}
                      />
                    </div>
                  </div>

                  <div className="mt-4 space-y-2">
                    <Label className="text-sm">Pixel Ratio</Label>
                    <div className="space-y-2">
                      <Slider
                        value={[pixelRatio]}
                        onValueChange={([value]) => handleSettingChange('visualisation.rendering.pixelRatio', value)}
                        min={0.5}
                        max={2.0}
                        step={0.1}
                        className="w-full"
                      />
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>0.5x</span>
                        <span>{pixelRatio}x</span>
                        <span>2.0x</span>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label className="text-sm">Target FPS</Label>
                    <div className="flex items-center gap-2">
                      <Input
                        type="number"
                        min="30"
                        max="144"
                        value={targetFPS}
                        onChange={(e) => handleSettingChange('visualisation.rendering.targetFPS', parseInt(e.target.value))}
                        className="w-20"
                      />
                      <span className="text-sm text-muted-foreground">fps</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Post-Processing Effects */}
              <div className="space-y-4">
                <div className="border-t pt-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Zap className="w-4 h-4" />
                    <h3 className="text-sm font-medium">Post-Processing Effects</h3>
                  </div>
                  
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label className="text-sm">Enable Post-Processing</Label>
                        <p className="text-xs text-muted-foreground">Master toggle for effects</p>
                      </div>
                      <Switch
                        checked={postProcessing}
                        onCheckedChange={(checked) => handleSettingChange('visualisation.postProcessing', checked)}
                      />
                    </div>

                    {postProcessing && (
                      <div className="grid grid-cols-2 gap-4 pl-4 border-l-2 border-muted">
                        <div className="flex items-center justify-between">
                          <div className="space-y-0.5">
                            <Label className="text-sm">Bloom Effect</Label>
                            <p className="text-xs text-muted-foreground">Glow around bright objects</p>
                          </div>
                          <Switch
                            checked={bloom}
                            onCheckedChange={(checked) => handleSettingChange('visualisation.effects.bloom', checked)}
                          />
                        </div>

                        <div className="flex items-center justify-between">
                          <div className="space-y-0.5">
                            <Label className="text-sm">SSAO</Label>
                            <p className="text-xs text-muted-foreground">Screen-space ambient occlusion</p>
                          </div>
                          <Switch
                            checked={ssao}
                            onCheckedChange={(checked) => handleSettingChange('visualisation.effects.ssao', checked)}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Camera Settings */}
              <div className="space-y-4">
                <div className="border-t pt-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Camera className="w-4 h-4" />
                    <h3 className="text-sm font-medium">Camera Settings</h3>
                  </div>
                  
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label className="text-sm">Field of View</Label>
                      <div className="space-y-2">
                        <Slider
                          value={[cameraFov]}
                          onValueChange={([value]) => handleSettingChange('visualisation.camera.fov', value)}
                          min={30}
                          max={120}
                          step={1}
                          className="w-full"
                        />
                        <div className="flex justify-between text-xs text-muted-foreground">
                          <span>30°</span>
                          <span>{cameraFov}°</span>
                          <span>120°</span>
                        </div>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label className="text-sm">Near Clipping</Label>
                        <Input
                          type="number"
                          min="0.01"
                          max="10"
                          step="0.01"
                          value={cameraNear}
                          onChange={(e) => handleSettingChange('visualisation.camera.near', parseFloat(e.target.value))}
                          className="w-full"
                        />
                      </div>

                      <div className="space-y-2">
                        <Label className="text-sm">Far Clipping</Label>
                        <Input
                          type="number"
                          min="100"
                          max="10000"
                          step="100"
                          value={cameraFar}
                          onChange={(e) => handleSettingChange('visualisation.camera.far', parseFloat(e.target.value))}
                          className="w-full"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Lighting Settings */}
              <div className="space-y-4">
                <div className="border-t pt-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Sun className="w-4 h-4" />
                    <h3 className="text-sm font-medium">Lighting</h3>
                  </div>
                  
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label className="text-sm">Ambient Light Intensity</Label>
                      <div className="space-y-2">
                        <Slider
                          value={[ambientIntensity]}
                          onValueChange={([value]) => handleSettingChange('visualisation.lighting.ambient.intensity', value)}
                          min={0}
                          max={1}
                          step={0.01}
                          className="w-full"
                        />
                        <div className="flex justify-between text-xs text-muted-foreground">
                          <span>0</span>
                          <span>{ambientIntensity.toFixed(2)}</span>
                          <span>1</span>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <Label className="text-sm">Directional Light Intensity</Label>
                      <div className="space-y-2">
                        <Slider
                          value={[directionalIntensity]}
                          onValueChange={([value]) => handleSettingChange('visualisation.lighting.directional.intensity', value)}
                          min={0}
                          max={2}
                          step={0.01}
                          className="w-full"
                        />
                        <div className="flex justify-between text-xs text-muted-foreground">
                          <span>0</span>
                          <span>{directionalIntensity.toFixed(2)}</span>
                          <span>2</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Material Settings */}
              <div className="space-y-4">
                <div className="border-t pt-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Palette className="w-4 h-4" />
                    <h3 className="text-sm font-medium">Materials & Textures</h3>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label className="text-sm">Material Quality</Label>
                      <Select
                        value={materialQuality}
                        onValueChange={(value) => handleSettingChange('visualisation.materials.quality', value)}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="low">Low</SelectItem>
                          <SelectItem value="medium">Medium</SelectItem>
                          <SelectItem value="high">High</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-2">
                      <Label className="text-sm">Texture Filtering</Label>
                      <Select
                        value={textureFiltering}
                        onValueChange={(value) => handleSettingChange('visualisation.materials.filtering', value)}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="nearest">Nearest</SelectItem>
                          <SelectItem value="linear">Linear</SelectItem>
                          <SelectItem value="trilinear">Trilinear</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label className="text-sm">Anisotropic Filtering</Label>
                    <div className="space-y-2">
                      <Slider
                        value={[anisotropicFiltering]}
                        onValueChange={([value]) => handleSettingChange('visualisation.materials.anisotropy', value)}
                        min={1}
                        max={16}
                        step={1}
                        className="w-full"
                      />
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>1x</span>
                        <span>{anisotropicFiltering}x</span>
                        <span>16x</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}

          {/* Reset Button */}
          <div className="border-t pt-4">
            <Button
              variant="outline"
              onClick={resetToDefaults}
              className="w-full"
              disabled={!enabled}
            >
              <RotateCcw className="w-4 h-4 mr-2" />
              Reset to Defaults
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default VisualizationSettings;