import React, { useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../features/design-system/components/Card';
import { Label } from '../../features/design-system/components/Label';
import { Slider } from '../../features/design-system/components/Slider';
import { Switch } from '../../features/design-system/components/Switch';
import { useSettingsStore } from '../../store/settingsStore';
import { useToast } from '../../features/design-system/components/Toast';
import { Palette, Sun, Sparkles } from 'lucide-react';

export const BackgroundEnvironmentControls: React.FC = () => {
  const { toast } = useToast();
  const { settings, updateSettings } = useSettingsStore();

  // Get current background environment settings
  const backgroundSettings = settings?.visualisation?.rendering || {};
  const glowSettings = settings?.visualisation?.glow || {};

  const handleBackgroundColorChange = useCallback((color: string) => {
    updateSettings((draft) => {
      if (!draft.visualisation) draft.visualisation = {} as any;
      if (!draft.visualisation.rendering) draft.visualisation.rendering = {} as any;
      draft.visualisation.rendering.backgroundColor = color;
    });
    
    toast({
      title: 'Background Updated',
      description: `Background color changed to ${color}`,
    });
  }, [updateSettings, toast]);

  const handleBackgroundOpacityChange = useCallback((opacity: number[]) => {
    const value = opacity[0];
    updateSettings((draft) => {
      if (!draft.visualisation) draft.visualisation = {} as any;
      if (!draft.visualisation.rendering) draft.visualisation.rendering = {} as any;
      (draft.visualisation.rendering as any).backgroundOpacity = value;
    });
  }, [updateSettings]);

  const handleGlowIntensityChange = useCallback((intensity: number[]) => {
    const value = intensity[0];
    updateSettings((draft) => {
      if (!draft.visualisation) draft.visualisation = {} as any;
      if (!draft.visualisation.glow) draft.visualisation.glow = {} as any;
      draft.visualisation.glow.environmentGlowStrength = value;
    });
  }, [updateSettings]);

  const handleAmbientLightChange = useCallback((intensity: number[]) => {
    const value = intensity[0];
    updateSettings((draft) => {
      if (!draft.visualisation) draft.visualisation = {} as any;
      if (!draft.visualisation.rendering) draft.visualisation.rendering = {} as any;
      draft.visualisation.rendering.ambientLightIntensity = value;
    });
  }, [updateSettings]);

  const handleDirectionalLightChange = useCallback((intensity: number[]) => {
    const value = intensity[0];
    updateSettings((draft) => {
      if (!draft.visualisation) draft.visualisation = {} as any;
      if (!draft.visualisation.rendering) draft.visualisation.rendering = {} as any;
      draft.visualisation.rendering.directionalLightIntensity = value;
    });
  }, [updateSettings]);

  const handleGlowEnabledChange = useCallback((enabled: boolean) => {
    updateSettings((draft) => {
      if (!draft.visualisation) draft.visualisation = {} as any;
      if (!draft.visualisation.glow) draft.visualisation.glow = {} as any;
      draft.visualisation.glow.enabled = enabled;
    });
  }, [updateSettings]);


  const handleEnvironmentIntensityChange = useCallback((intensity: number[]) => {
    const value = intensity[0];
    updateSettings((draft) => {
      if (!draft.visualisation) draft.visualisation = {} as any;
      if (!draft.visualisation.rendering) draft.visualisation.rendering = {} as any;
      draft.visualisation.rendering.environmentIntensity = value;
    });
  }, [updateSettings]);

  return (
    <div className="space-y-4">
      {/* Background Controls */}
      <Card className="bg-white/5 border-white/10">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Palette className="h-4 w-4" />
            Background Environment
          </CardTitle>
          <CardDescription className="text-xs text-white/60">
            Control the visual environment backdrop
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Background Color */}
          <div className="space-y-2">
            <Label htmlFor="bg-color" className="text-xs font-medium">
              Background Color
            </Label>
            <div className="flex gap-2">
              <input
                type="color"
                id="bg-color"
                value={(backgroundSettings as any).backgroundColor || '#000000'}
                onChange={(e) => handleBackgroundColorChange(e.target.value)}
                className="w-12 h-8 rounded border border-white/20 bg-transparent cursor-pointer"
              />
              <input
                type="text"
                value={(backgroundSettings as any).backgroundColor || '#000000'}
                onChange={(e) => handleBackgroundColorChange(e.target.value)}
                className="flex-1 px-2 py-1 text-xs bg-white/10 border border-white/20 rounded text-white"
                placeholder="#000000"
              />
            </div>
          </div>

          {/* Background Opacity */}
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <Label className="text-xs font-medium">Background Opacity</Label>
              <span className="text-xs text-white/60">
                {(((backgroundSettings as any).backgroundOpacity || 1) * 100).toFixed(0)}%
              </span>
            </div>
            <Slider
              value={[(backgroundSettings as any).backgroundOpacity || 1]}
              onValueChange={handleBackgroundOpacityChange}
              min={0}
              max={1}
              step={0.01}
              className="w-full"
            />
          </div>
        </CardContent>
      </Card>

      {/* Lighting Controls */}
      <Card className="bg-white/5 border-white/10">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Sun className="h-4 w-4" />
            Environment Lighting
          </CardTitle>
          <CardDescription className="text-xs text-white/60">
            Adjust ambient and directional lighting
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Ambient Light */}
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <Label className="text-xs font-medium">Ambient Light</Label>
              <span className="text-xs text-white/60">
                {((backgroundSettings as any).ambientLightIntensity || 0.5).toFixed(2)}
              </span>
            </div>
            <Slider
              value={[(backgroundSettings as any).ambientLightIntensity || 0.5]}
              onValueChange={handleAmbientLightChange}
              min={0}
              max={2}
              step={0.01}
              className="w-full"
            />
          </div>

          {/* Directional Light */}
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <Label className="text-xs font-medium">Directional Light</Label>
              <span className="text-xs text-white/60">
                {((backgroundSettings as any).directionalLightIntensity || 1).toFixed(2)}
              </span>
            </div>
            <Slider
              value={[(backgroundSettings as any).directionalLightIntensity || 1]}
              onValueChange={handleDirectionalLightChange}
              min={0}
              max={2}
              step={0.01}
              className="w-full"
            />
          </div>

          {/* Environment Intensity */}
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <Label className="text-xs font-medium">Environment Intensity</Label>
              <span className="text-xs text-white/60">
                {((backgroundSettings as any).environmentIntensity || 1).toFixed(2)}
              </span>
            </div>
            <Slider
              value={[(backgroundSettings as any).environmentIntensity || 1]}
              onValueChange={handleEnvironmentIntensityChange}
              min={0}
              max={2}
              step={0.01}
              className="w-full"
            />
          </div>
        </CardContent>
      </Card>

      {/* Glow & Effects Controls */}
      <Card className="bg-white/5 border-white/10">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Sparkles className="h-4 w-4" />
            Environment Effects
          </CardTitle>
          <CardDescription className="text-xs text-white/60">
            Atmospheric glow and bloom effects
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Global Glow Toggle */}
          <div className="flex items-center justify-between">
            <Label htmlFor="glow-enabled" className="text-xs font-medium">
              Enable Glow Effects
            </Label>
            <Switch
              id="glow-enabled"
              checked={(glowSettings as any).enabled || false}
              onCheckedChange={handleGlowEnabledChange}
            />
          </div>

          {/* Glow Intensity */}
          {(glowSettings as any).enabled && (
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <Label className="text-xs font-medium">Environment Glow</Label>
                <span className="text-xs text-white/60">
                  {((glowSettings as any).environmentGlowStrength || 0).toFixed(1)}
                </span>
              </div>
              <Slider
                value={[(glowSettings as any).environmentGlowStrength || 0]}
                onValueChange={handleGlowIntensityChange}
                min={0}
                max={10}
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