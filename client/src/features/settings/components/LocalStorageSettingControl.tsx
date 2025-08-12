/**
 * Setting control component that uses localStorage instead of backend sync
 */

import React, { useState, useEffect, useCallback } from 'react';
import { Switch } from '@/features/design-system/components/Switch';
import { Slider } from '@/features/design-system/components/Slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/features/design-system/components/Select';
import { Input } from '@/features/design-system/components/Input';
import { clientDebugState } from '@/utils/clientDebugState';
import { createLogger } from '@/utils/logger';

const logger = createLogger('LocalStorageSettingControl');

interface LocalStorageSettingControlProps {
  setting: {
    label: string;
    type: 'toggle' | 'slider' | 'select' | 'text' | 'number';
    path: string;
    description?: string;
    min?: number;
    max?: number;
    step?: number;
    unit?: string;
    options?: Array<{ value: string; label: string }>;
    localStorage: boolean;
  };
}

export const LocalStorageSettingControl: React.FC<LocalStorageSettingControlProps> = ({ setting }) => {
  const [value, setValue] = useState<any>(null);

  // Extract the key from the path (e.g., 'debug.enabled' -> 'enabled')
  const getDebugKey = useCallback(() => {
    if (setting.path.startsWith('debug.')) {
      return setting.path.replace('debug.', '');
    }
    return setting.path;
  }, [setting.path]);

  // Load initial value from localStorage
  useEffect(() => {
    const loadValue = () => {
      const key = getDebugKey();
      
      // Check if this is a debug setting
      if (setting.path.startsWith('debug.')) {
        // Use clientDebugState for debug settings
        const debugKey = key as any;
        const val = clientDebugState.get(debugKey);
        setValue(val);
      } else {
        // Direct localStorage access for non-debug settings
        const stored = localStorage.getItem(setting.path);
        if (stored !== null) {
          if (setting.type === 'toggle') {
            setValue(stored === 'true');
          } else if (setting.type === 'number' || setting.type === 'slider') {
            setValue(parseFloat(stored));
          } else {
            setValue(stored);
          }
        } else {
          // Set default values
          setValue(setting.type === 'toggle' ? false : 
                  setting.type === 'number' || setting.type === 'slider' ? setting.min || 0 :
                  '');
        }
      }
    };

    loadValue();

    // Subscribe to changes if it's a debug setting
    if (setting.path.startsWith('debug.')) {
      const key = getDebugKey() as any;
      const unsubscribe = clientDebugState.subscribe(key, (newValue) => {
        setValue(newValue);
      });
      return unsubscribe;
    }

    // Listen for storage changes from other tabs
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === setting.path) {
        loadValue();
      }
    };

    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, [setting.path, setting.type, setting.min, getDebugKey]);

  // Handle value changes
  const handleChange = useCallback((newValue: any) => {
    setValue(newValue);
    
    const key = getDebugKey();
    
    if (setting.path.startsWith('debug.')) {
      // Use clientDebugState for debug settings
      const debugKey = key as any;
      clientDebugState.set(debugKey, newValue);
    } else {
      // Direct localStorage for non-debug settings
      localStorage.setItem(setting.path, String(newValue));
      
      // Dispatch storage event for same-tab updates
      window.dispatchEvent(new StorageEvent('storage', {
        key: setting.path,
        newValue: String(newValue),
        oldValue: localStorage.getItem(setting.path),
        storageArea: localStorage,
      }));
    }
    
    logger.debug(`Setting ${setting.path} changed to ${newValue}`);
  }, [setting.path, getDebugKey]);

  // Render different control types
  const renderControl = () => {
    switch (setting.type) {
      case 'toggle':
        return (
          <Switch
            checked={value || false}
            onCheckedChange={handleChange}
            aria-label={setting.label}
          />
        );

      case 'slider':
        return (
          <div className="flex items-center gap-2">
            <Slider
              value={[value || setting.min || 0]}
              onValueChange={([val]) => handleChange(val)}
              min={setting.min}
              max={setting.max}
              step={setting.step}
              className="flex-1"
            />
            <span className="text-sm text-muted-foreground min-w-[3rem] text-right">
              {value || setting.min || 0}{setting.unit || ''}
            </span>
          </div>
        );

      case 'select':
        return (
          <Select value={value || ''} onValueChange={handleChange}>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select an option" />
            </SelectTrigger>
            <SelectContent>
              {setting.options?.map(option => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        );

      case 'text':
        return (
          <Input
            type="text"
            value={value || ''}
            onChange={(e) => handleChange(e.target.value)}
            placeholder={setting.label}
          />
        );

      case 'number':
        return (
          <Input
            type="number"
            value={value || 0}
            onChange={(e) => handleChange(parseFloat(e.target.value) || 0)}
            min={setting.min}
            max={setting.max}
            step={setting.step}
          />
        );

      default:
        return null;
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="space-y-0.5">
          <label className="text-sm font-medium">{setting.label}</label>
          {setting.description && (
            <p className="text-xs text-muted-foreground">{setting.description}</p>
          )}
        </div>
        <div className="flex-shrink-0 ml-4">
          {renderControl()}
        </div>
      </div>
    </div>
  );
};