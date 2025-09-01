import React, { useState } from 'react';
import { Button } from '../../design-system/components/Button';
import { Card, CardContent, CardHeader, CardTitle } from '../../design-system/components/Card';
import { Label } from '../../design-system/components/Label';
import { Switch } from '../../design-system/components/Switch';
import { Badge } from '../../design-system/components/Badge';
import { Keyboard, Settings } from 'lucide-react';
import { useSelectiveSetting, useSettingSetter } from '../../../hooks/useSelectiveSettingsStore';
import { useKeyboardShortcutsList, formatShortcut } from '../../../hooks/useKeyboardShortcuts';
import { KeyboardShortcutsModal } from '../../../components/KeyboardShortcutsModal';

/**
 * KeyboardShortcuts Settings Panel
 * Provides settings for keyboard shortcut configuration with selective access patterns
 */
export function KeyboardShortcuts() {
  const [showModal, setShowModal] = useState(false);
  const { set } = useSettingSetter();
  
  // Use selective settings access for keyboard shortcut related settings
  const shortcutsEnabled = useSelectiveSetting<boolean>('system.keyboard.enabled') ?? true;
  const globalShortcuts = useSelectiveSetting<boolean>('system.keyboard.globalShortcuts') ?? true;
  const modalShortcuts = useSelectiveSetting<boolean>('system.keyboard.modalShortcuts') ?? true;
  const navigationShortcuts = useSelectiveSetting<boolean>('system.keyboard.navigation') ?? true;
  const debugShortcuts = useSelectiveSetting<boolean>('system.keyboard.debug') ?? false;
  
  // Get the list of registered shortcuts for display
  const shortcuts = useKeyboardShortcutsList();

  const handleToggleSetting = async (path: string, value: boolean) => {
    await set(path, value);
  };

  const shortcutCategories = [
    {
      title: 'Global Shortcuts',
      enabled: globalShortcuts,
      path: 'system.keyboard.globalShortcuts',
      description: 'System-wide shortcuts that work everywhere',
      shortcuts: shortcuts.filter(s => s.category === 'Global' || s.category === 'System')
    },
    {
      title: 'Modal Shortcuts', 
      enabled: modalShortcuts,
      path: 'system.keyboard.modalShortcuts',
      description: 'Shortcuts for modal dialogs and overlays',
      shortcuts: shortcuts.filter(s => s.category === 'Modal')
    },
    {
      title: 'Navigation Shortcuts',
      enabled: navigationShortcuts,
      path: 'system.keyboard.navigation',
      description: 'Shortcuts for navigating the interface',
      shortcuts: shortcuts.filter(s => s.category === 'Navigation')
    },
    {
      title: 'Debug Shortcuts',
      enabled: debugShortcuts,
      path: 'system.keyboard.debug',
      description: 'Developer shortcuts for debugging',
      shortcuts: shortcuts.filter(s => s.category === 'Debug')
    }
  ];

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Keyboard className="w-5 h-5" />
            <CardTitle>Keyboard Shortcuts</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Master enable/disable */}
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label className="text-sm font-medium">Enable Keyboard Shortcuts</Label>
              <p className="text-xs text-muted-foreground">
                Master toggle for all keyboard shortcuts
              </p>
            </div>
            <Switch
              checked={shortcutsEnabled}
              onCheckedChange={(checked) => handleToggleSetting('system.keyboard.enabled', checked)}
            />
          </div>

          {/* Show shortcuts reference */}
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label className="text-sm font-medium">Shortcuts Reference</Label>
              <p className="text-xs text-muted-foreground">
                View all available keyboard shortcuts
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowModal(true)}
              disabled={!shortcutsEnabled}
            >
              View All
            </Button>
          </div>

          {/* Shortcut Categories */}
          {shortcutsEnabled && (
            <div className="space-y-4">
              <div className="border-t pt-4">
                <h3 className="text-sm font-medium mb-3">Shortcut Categories</h3>
                <div className="space-y-3">
                  {shortcutCategories.map((category) => (
                    <div key={category.path} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="space-y-0.5 flex-1">
                          <Label className="text-sm font-medium">{category.title}</Label>
                          <p className="text-xs text-muted-foreground">
                            {category.description}
                          </p>
                          {category.shortcuts.length > 0 && (
                            <div className="flex flex-wrap gap-1 mt-1">
                              {category.shortcuts.slice(0, 3).map((shortcut, idx) => (
                                <Badge
                                  key={idx}
                                  variant="secondary"
                                  className="text-xs font-mono"
                                >
                                  {formatShortcut(shortcut.key)}
                                </Badge>
                              ))}
                              {category.shortcuts.length > 3 && (
                                <Badge variant="outline" className="text-xs">
                                  +{category.shortcuts.length - 3} more
                                </Badge>
                              )}
                            </div>
                          )}
                        </div>
                        <Switch
                          checked={category.enabled}
                          onCheckedChange={(checked) => handleToggleSetting(category.path, checked)}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Keyboard Shortcuts Modal */}
      <KeyboardShortcutsModal
        isOpen={showModal}
        onClose={() => setShowModal(false)}
      />
    </div>
  );
}

export default KeyboardShortcuts;