import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../../design-system/components/Card';
import { Label } from '../../design-system/components/Label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../../design-system/components/Select';
import { Switch } from '../../design-system/components/Switch';
import { Input } from '../../design-system/components/Input';
import { Button } from '../../design-system/components/Button';
import { Palette, Sun, Moon, Monitor, Paintbrush } from 'lucide-react';
import { useSelectiveSetting, useSettingSetter } from '../../../hooks/useSelectiveSettingsStore';

/**
 * ThemeSelector Settings Panel
 * Provides theme and appearance settings with selective access patterns
 */
export function ThemeSelector() {
  const { set } = useSettingSetter();
  
  // Use selective settings access for theme-related settings
  const theme = useSelectiveSetting<string>('ui.theme') ?? 'system';
  const darkMode = useSelectiveSetting<boolean>('ui.darkMode') ?? false;
  const highContrast = useSelectiveSetting<boolean>('ui.highContrast') ?? false;
  const reducedMotion = useSelectiveSetting<boolean>('ui.reducedMotion') ?? false;
  const colorScheme = useSelectiveSetting<string>('ui.colorScheme') ?? 'default';
  
  // Color customization
  const primaryColor = useSelectiveSetting<string>('ui.colors.primary') ?? '#3b82f6';
  const secondaryColor = useSelectiveSetting<string>('ui.colors.secondary') ?? '#6b7280';
  const accentColor = useSelectiveSetting<string>('ui.colors.accent') ?? '#8b5cf6';
  
  // Typography
  const fontSize = useSelectiveSetting<number>('ui.typography.fontSize') ?? 14;
  const fontFamily = useSelectiveSetting<string>('ui.typography.fontFamily') ?? 'system';

  const handleSettingChange = async (path: string, value: any) => {
    await set(path, value);
  };

  const themes = [
    { value: 'light', label: 'Light', icon: Sun },
    { value: 'dark', label: 'Dark', icon: Moon },
    { value: 'system', label: 'System', icon: Monitor }
  ];

  const colorSchemes = [
    { value: 'default', label: 'Default' },
    { value: 'blue', label: 'Blue' },
    { value: 'green', label: 'Green' },
    { value: 'purple', label: 'Purple' },
    { value: 'orange', label: 'Orange' },
    { value: 'custom', label: 'Custom' }
  ];

  const fontFamilies = [
    { value: 'system', label: 'System Default' },
    { value: 'inter', label: 'Inter' },
    { value: 'roboto', label: 'Roboto' },
    { value: 'opensans', label: 'Open Sans' },
    { value: 'monospace', label: 'Monospace' }
  ];

  const resetToDefaults = async () => {
    await Promise.all([
      set('ui.theme', 'system'),
      set('ui.darkMode', false),
      set('ui.highContrast', false),
      set('ui.reducedMotion', false),
      set('ui.colorScheme', 'default'),
      set('ui.colors.primary', '#3b82f6'),
      set('ui.colors.secondary', '#6b7280'),
      set('ui.colors.accent', '#8b5cf6'),
      set('ui.typography.fontSize', 14),
      set('ui.typography.fontFamily', 'system')
    ]);
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Palette className="w-5 h-5" />
            <CardTitle>Theme & Appearance</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Theme Selection */}
          <div className="space-y-2">
            <Label className="text-sm font-medium">Theme</Label>
            <div className="grid grid-cols-3 gap-2">
              {themes.map((themeOption) => {
                const Icon = themeOption.icon;
                return (
                  <Button
                    key={themeOption.value}
                    variant={theme === themeOption.value ? 'default' : 'outline'}
                    className="flex items-center gap-2 justify-start"
                    onClick={() => handleSettingChange('ui.theme', themeOption.value)}
                  >
                    <Icon className="w-4 h-4" />
                    {themeOption.label}
                  </Button>
                );
              })}
            </div>
          </div>

          {/* Accessibility Options */}
          <div className="space-y-4">
            <div className="border-t pt-4">
              <h3 className="text-sm font-medium mb-3">Accessibility</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label className="text-sm">High Contrast</Label>
                    <p className="text-xs text-muted-foreground">
                      Increase contrast for better visibility
                    </p>
                  </div>
                  <Switch
                    checked={highContrast}
                    onCheckedChange={(checked) => handleSettingChange('ui.highContrast', checked)}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label className="text-sm">Reduced Motion</Label>
                    <p className="text-xs text-muted-foreground">
                      Minimize animations and transitions
                    </p>
                  </div>
                  <Switch
                    checked={reducedMotion}
                    onCheckedChange={(checked) => handleSettingChange('ui.reducedMotion', checked)}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Color Scheme */}
          <div className="space-y-2">
            <Label className="text-sm font-medium">Color Scheme</Label>
            <Select
              value={colorScheme}
              onValueChange={(value) => handleSettingChange('ui.colorScheme', value)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select color scheme" />
              </SelectTrigger>
              <SelectContent>
                {colorSchemes.map((scheme) => (
                  <SelectItem key={scheme.value} value={scheme.value}>
                    {scheme.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Custom Colors */}
          {colorScheme === 'custom' && (
            <div className="space-y-4">
              <div className="border-t pt-4">
                <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
                  <Paintbrush className="w-4 h-4" />
                  Custom Colors
                </h3>
                <div className="grid grid-cols-1 gap-3">
                  <div className="flex items-center gap-3">
                    <Label className="text-sm min-w-[80px]">Primary</Label>
                    <div className="flex items-center gap-2 flex-1">
                      <Input
                        type="color"
                        value={primaryColor}
                        onChange={(e) => handleSettingChange('ui.colors.primary', e.target.value)}
                        className="w-12 h-8 p-1 border rounded"
                      />
                      <Input
                        type="text"
                        value={primaryColor}
                        onChange={(e) => handleSettingChange('ui.colors.primary', e.target.value)}
                        className="font-mono text-sm flex-1"
                        placeholder="#3b82f6"
                      />
                    </div>
                  </div>

                  <div className="flex items-center gap-3">
                    <Label className="text-sm min-w-[80px]">Secondary</Label>
                    <div className="flex items-center gap-2 flex-1">
                      <Input
                        type="color"
                        value={secondaryColor}
                        onChange={(e) => handleSettingChange('ui.colors.secondary', e.target.value)}
                        className="w-12 h-8 p-1 border rounded"
                      />
                      <Input
                        type="text"
                        value={secondaryColor}
                        onChange={(e) => handleSettingChange('ui.colors.secondary', e.target.value)}
                        className="font-mono text-sm flex-1"
                        placeholder="#6b7280"
                      />
                    </div>
                  </div>

                  <div className="flex items-center gap-3">
                    <Label className="text-sm min-w-[80px]">Accent</Label>
                    <div className="flex items-center gap-2 flex-1">
                      <Input
                        type="color"
                        value={accentColor}
                        onChange={(e) => handleSettingChange('ui.colors.accent', e.target.value)}
                        className="w-12 h-8 p-1 border rounded"
                      />
                      <Input
                        type="text"
                        value={accentColor}
                        onChange={(e) => handleSettingChange('ui.colors.accent', e.target.value)}
                        className="font-mono text-sm flex-1"
                        placeholder="#8b5cf6"
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Typography */}
          <div className="space-y-4">
            <div className="border-t pt-4">
              <h3 className="text-sm font-medium mb-3">Typography</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label className="text-sm">Font Family</Label>
                  <Select
                    value={fontFamily}
                    onValueChange={(value) => handleSettingChange('ui.typography.fontFamily', value)}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select font" />
                    </SelectTrigger>
                    <SelectContent>
                      {fontFamilies.map((font) => (
                        <SelectItem key={font.value} value={font.value}>
                          {font.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label className="text-sm">Font Size</Label>
                  <div className="flex items-center gap-2">
                    <Input
                      type="number"
                      min="10"
                      max="24"
                      value={fontSize}
                      onChange={(e) => handleSettingChange('ui.typography.fontSize', parseInt(e.target.value))}
                      className="w-20"
                    />
                    <span className="text-sm text-muted-foreground">px</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Reset Button */}
          <div className="border-t pt-4">
            <Button
              variant="outline"
              onClick={resetToDefaults}
              className="w-full"
            >
              Reset to Defaults
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default ThemeSelector;