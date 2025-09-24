// Unified Settings Panel - The single control center for all settings
import React, { useState, useEffect, useMemo } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../../../design-system/components/Tabs';
import { Input } from '../../../design-system/components/Input';
import { Button } from '../../../design-system/components/Button';
import { ScrollArea } from '../../../design-system/components/ScrollArea';
import { 
  Settings, 
  Monitor, 
  Palette, 
  Activity, 
  Database, 
  Code, 
  Shield, 
  Headphones,
  Search,
  Save,
  RotateCcw,
  Download,
  Upload,
  Undo2 as Undo,
  Redo2 as Redo,
  Brain,
  Eye,
  BarChart3,
  Smartphone
} from 'lucide-react';
import { useSettingsStore, settingsSelectors } from '../../../../store/settingsStore';
import { settingsUIDefinition } from '../../config/settingsUIDefinition';
import { SettingControlComponent } from '../SettingControlComponent';
import { toast } from '../../../../utils/toast';
import { createLogger } from '../../../../utils/loggerConfig';

interface SettingsPanelRedesignProps {
  isOpen?: boolean;
  onClose?: () => void;
}

export const SettingsPanelRedesign: React.FC<SettingsPanelRedesignProps> = ({ 
  isOpen = true, 
  onClose 
}) => {
  const [activeTab, setActiveTab] = useState('visualization');
  const [searchQuery, setSearchQuery] = useState('');
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  
  // Settings store
  const { 
    settings, 
    saving, 
    loading, 
    updateSettings,
    saveSettings,
    resetSettings,
    exportToFile,
    loadFromFile,
    hasUnsavedChanges: checkUnsavedChanges
  } = useSettingsStore();
  
  // Undo/redo support (disabled - not implemented yet)
  const canUndo = false;
  const canRedo = false;
  const undo = () => toast.info('Undo not yet implemented');
  const redo = () => toast.info('Redo not yet implemented');
  
  // Check for unsaved changes
  useEffect(() => {
    const interval = setInterval(() => {
      setHasUnsavedChanges(checkUnsavedChanges());
    }, 1000);
    return () => clearInterval(interval);
  }, [checkUnsavedChanges]);
  
  // Filter settings based on search
  const filteredUIDefinition = useMemo(() => {
    if (!searchQuery) return settingsUIDefinition;
    
    const query = searchQuery.toLowerCase();
    const filtered: typeof settingsUIDefinition = {};
    
    Object.entries(settingsUIDefinition).forEach(([category, categoryDef]) => {
      const filteredSubsections: typeof categoryDef.subsections = {};
      
      Object.entries(categoryDef.subsections || {}).forEach(([sectionKey, section]) => {
        const filteredSettings: Record<string, any> = {};
        
        Object.entries(section.settings || {}).forEach(([settingKey, setting]) => {
          if (setting.label.toLowerCase().includes(query) ||
              setting.path.toLowerCase().includes(query) ||
              setting.description?.toLowerCase().includes(query)) {
            filteredSettings[settingKey] = setting;
          }
        });
        
        if (Object.keys(filteredSettings).length > 0) {
          filteredSubsections[sectionKey] = {
            ...section,
            settings: filteredSettings
          };
        }
      });
      
      if (Object.keys(filteredSubsections).length > 0) {
        filtered[category] = {
          ...categoryDef,
          subsections: filteredSubsections
        };
      }
    });
    
    return filtered;
  }, [searchQuery]);
  
  // Handle file import
  const handleFileImport = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      await loadFromFile(file);
      event.target.value = ''; // Reset input
    }
  };
  
  // Tab definitions - comprehensive settings categories
  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: Monitor, category: 'dashboard' },
    { id: 'visualization', label: 'Visualization', icon: Eye, category: 'visualization' },
    { id: 'physics', label: 'Physics', icon: Activity, category: 'physics' },
    { id: 'analytics', label: 'Analytics', icon: BarChart3, category: 'analytics' },
    { id: 'xr', label: 'XR/AR', icon: Smartphone, category: 'xr' },
    { id: 'performance', label: 'Performance', icon: Activity, category: 'performance' },
    { id: 'data', label: 'Data', icon: Database, category: 'integrations' },
    { id: 'system', label: 'System', icon: Settings, category: 'system' },
    { id: 'ai', label: 'AI Services', icon: Brain, category: 'ai' },
    { id: 'developer', label: 'Developer', icon: Code, category: 'developer' },
    { id: 'auth', label: 'Auth', icon: Shield, category: 'auth' },
  ];
  
  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center gap-2">
          <Settings className="w-5 h-5" />
          <h2 className="text-lg font-semibold">Control Center</h2>
          {hasUnsavedChanges && (
            <span className="text-xs text-orange-500 ml-2">(Unsaved changes)</span>
          )}
        </div>
        
        <div className="flex items-center gap-2">
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-2 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              type="text"
              placeholder="Search settings..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-8 w-64"
            />
          </div>
          
          {/* Undo/Redo */}
          <div className="flex gap-1">
            <Button
              variant="ghost"
              size="icon"
              onClick={undo}
              disabled={!canUndo}
              title="Undo"
            >
              <Undo className="w-4 h-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={redo}
              disabled={!canRedo}
              title="Redo"
            >
              <Redo className="w-4 h-4" />
            </Button>
          </div>
          
          {/* Import/Export */}
          <div className="flex gap-1">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => document.getElementById('import-settings')?.click()}
              title="Import settings"
            >
              <Upload className="w-4 h-4" />
            </Button>
            <input
              id="import-settings"
              type="file"
              accept=".json"
              onChange={handleFileImport}
              className="hidden"
            />
            <Button
              variant="ghost"
              size="icon"
              onClick={exportToFile}
              title="Export settings"
            >
              <Download className="w-4 h-4" />
            </Button>
          </div>
          
          {/* Save/Reset */}
          <div className="flex gap-1">
            <Button
              variant="ghost"
              size="icon"
              onClick={saveSettings}
              disabled={saving || !hasUnsavedChanges}
              title="Save settings"
            >
              <Save className="w-4 h-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={resetSettings}
              title="Reset to defaults"
            >
              <RotateCcw className="w-4 h-4" />
            </Button>
          </div>
          
          {onClose && (
            <Button variant="ghost" size="sm" onClick={onClose}>
              Close
            </Button>
          )}
        </div>
      </div>
      
      {/* Content */}
      <div className="flex-1 overflow-hidden">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full">
          <TabsList className="px-4 w-full justify-start">
            {tabs.map(tab => (
              <TabsTrigger key={tab.id} value={tab.id} className="flex items-center gap-2">
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </TabsTrigger>
            ))}
          </TabsList>
          
          <ScrollArea className="flex-1 h-[calc(100%-3rem)]">
            {tabs.map(tab => {
              const categoryDef = filteredUIDefinition[tab.category];
              if (!categoryDef) return null;
              
              return (
                <TabsContent key={tab.id} value={tab.id} className="p-4 space-y-4">
                  {/* Tab description */}
                  {categoryDef.description && (
                    <div className="text-sm text-muted-foreground mb-4">
                      {categoryDef.description}
                    </div>
                  )}
                  
                  {/* Settings sections */}
                  {Object.entries(categoryDef.subsections || {}).map(([sectionKey, section]) => (
                    <div key={sectionKey} className="space-y-2">
                      <h3 className="text-sm font-semibold">{section.label}</h3>
                      {section.description && (
                        <p className="text-xs text-muted-foreground">{section.description}</p>
                      )}
                      <div className="space-y-2">
                        {Object.values(section.settings || {}).map(setting => (
                          <SettingControlComponent
                            key={setting.path}
                            path={setting.path}
                            settingDef={setting}
                            value={getSettingValue(settings, setting.path)}
                            onChange={(value) => {
                              updateSettings((draft) => {
                                setSettingValue(draft, setting.path, value);
                              });
                            }}
                          />
                        ))}
                      </div>
                    </div>
                  ))}
                </TabsContent>
              );
            })}
          </ScrollArea>
        </Tabs>
      </div>
      
      {/* Status bar */}
      {(loading || saving) && (
        <div className="px-4 py-2 border-t text-xs text-muted-foreground">
          {loading && 'Loading settings...'}
          {saving && 'Saving settings...'}
        </div>
      )}
    </div>
  );
};

// Helper to get nested value by path
function getSettingValue(obj: any, path: string): any {
  return path.split('.').reduce((acc, part) => acc?.[part], obj);
}

// Helper to set nested value by path
function setSettingValue(obj: any, path: string, value: any): void {
  const parts = path.split('.');
  const last = parts.pop()!;
  const target = parts.reduce((acc, part) => {
    if (!acc[part]) acc[part] = {};
    return acc[part];
  }, obj);
  target[last] = value;
}