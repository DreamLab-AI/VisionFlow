// Unified Settings Panel - The single control center for all settings
import React, { useState, useEffect, useMemo, useCallback, Suspense } from 'react';
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
  Redo2 as Redo
} from 'lucide-react';
import { useSettingsStore } from '../../../../store/settingsStore';
import { useSelectiveSetting, useSettingSetter } from '../../../../hooks/useSelectiveSettingsStore';
import { useSettingsHistory } from '../../hooks/useSettingsHistory';
import { settingsUIDefinition } from '../../config/settingsUIDefinition';
import { SettingControlComponent } from '../SettingControlComponent';
import { toast } from '../../../../utils/toast';
import { logger } from '../../../../utils/logger';

// Skeleton component for loading state
const SettingsTabSkeleton: React.FC = () => (
  <div className="space-y-4 animate-pulse">
    <div className="h-4 bg-muted rounded w-3/4"></div>
    <div className="space-y-2">
      <div className="h-3 bg-muted rounded w-1/2"></div>
      <div className="h-8 bg-muted rounded"></div>
      <div className="h-8 bg-muted rounded"></div>
      <div className="h-8 bg-muted rounded"></div>
    </div>
  </div>
);

// Lazy-loaded settings tab component
interface LazySettingsTabProps {
  tabId: string;
  category: string;
  filteredUIDefinition: any;
  updateSettings: (updater: (draft: any) => void) => void;
}

const LazySettingsTab: React.FC<LazySettingsTabProps> = ({
  tabId,
  category,
  filteredUIDefinition,
  updateSettings
}) => {
  const { loadSection } = useSettingsStore();
  
  useEffect(() => {
    // Load the settings section when the tab becomes active
    loadSection(category).catch(error => {
      logger.error(`Failed to load settings section ${category}:`, error);
    });
  }, [category, loadSection]);

  const categoryDef = filteredUIDefinition[category];
  if (!categoryDef) return null;

  return (
    <>
      {/* Tab description */}
      {categoryDef.description && (
        <div className="text-sm text-muted-foreground mb-4">
          {categoryDef.description}
        </div>
      )}
      
      {/* Settings sections */}
      {Object.entries(categoryDef.subsections || {}).map(([sectionKey, section]: [string, any]) => (
        <div key={sectionKey} className="space-y-2">
          <h3 className="text-sm font-semibold">{section.label}</h3>
          {section.description && (
            <p className="text-xs text-muted-foreground">{section.description}</p>
          )}
          <div className="space-y-2">
            {Object.values(section.settings || {}).map((setting: any) => (
              <SettingControlComponent
                key={setting.path}
                path={setting.path}
                settingDef={setting}
                value={useSettingsStore.getState().get(setting.path)}
                onChange={(value) => {
                  // Use path-based setter directly
                  updateSettings((draft) => {
                    setSettingValue(draft, setting.path, value);
                  });
                }}
              />
            ))}
          </div>
        </div>
      ))}
    </>
  );
};

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
  const [showResetConfirmation, setShowResetConfirmation] = useState(false);
  
  // Settings store - using selective access for performance
  const loading = useSelectiveSetting<boolean>('system.debug.enabled') === undefined; // Derive loading from essential settings
  const saving = useSettingsStore(state => state.saving);
  const hasUnsavedChanges = useSettingsStore(state => state.hasUnsavedChanges);
  const { batchedSet } = useSettingSetter();
  const saveSettings = useSettingsStore(state => state.saveSettings);
  const resetSettingsStore = useSettingsStore(state => state.resetSettings);
  const updateSettings = useSettingsStore(state => state.updateSettings);
  
  // Only get settings when needed for filtering - avoid full object subscription
  const getSettingsForFiltering = useSettingsStore(state => state.partialSettings);

  // Undo/redo support using useSettingsHistory hook
  const { 
    canUndo, 
    canRedo, 
    undo, 
    redo 
  } = useSettingsHistory();

  // File operations (placeholder implementations for now)
  const exportToFile = () => toast.info('Export not yet implemented')
  const loadFromFile = () => toast.info('Import not yet implemented')
  
  // Reset with confirmation
  const resetSettings = useCallback(async () => {
    if (showResetConfirmation) {
      try {
        await resetSettingsStore()
        setShowResetConfirmation(false)
      } catch (error) {
        logger.error('Failed to reset settings:', error)
      }
    } else {
      setShowResetConfirmation(true)
      setTimeout(() => setShowResetConfirmation(false), 5000)
    }
  }, [showResetConfirmation, resetSettingsStore])
  
  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Only handle shortcuts if no input is focused
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
        return;
      }

      if (event.ctrlKey || event.metaKey) {
        switch (event.key.toLowerCase()) {
          case 's':
            event.preventDefault();
            if (hasUnsavedChanges && !saving) {
              saveSettings();
            }
            break;
          case 'z':
            if (event.shiftKey) {
              // Ctrl+Shift+Z = Redo (alternative to Ctrl+Y)
              event.preventDefault();
              if (canRedo) {
                redo();
              }
            } else {
              // Ctrl+Z = Undo
              event.preventDefault();
              if (canUndo) {
                undo();
              }
            }
            break;
          case 'y':
            // Ctrl+Y = Redo
            event.preventDefault();
            if (canRedo) {
              redo();
            }
            break;
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [hasUnsavedChanges, saving, saveSettings, canUndo, canRedo, undo, redo]);
  
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
  
  // Tab definitions
  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: Monitor, category: 'dashboard' },
    { id: 'visualization', label: 'Visualization', icon: Palette, category: 'visualization' },
    { id: 'physics', label: 'Physics', icon: Activity, category: 'physics' },
    { id: 'analytics', label: 'Analytics', icon: Activity, category: 'analytics' },
    { id: 'xr', label: 'XR/AR', icon: Headphones, category: 'xr' },
    { id: 'performance', label: 'Performance', icon: Activity, category: 'performance' },
    { id: 'data', label: 'Data', icon: Database, category: 'integrations' },
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
            <div className="flex items-center gap-1 ml-2">
              <div className="w-2 h-2 bg-orange-500 rounded-full animate-pulse"></div>
              <span className="text-xs text-orange-500">Auto-saving...</span>
            </div>
          )}
          {showResetConfirmation && (
            <div className="flex items-center gap-1 ml-2">
              <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
              <span className="text-xs text-red-500">Click reset again to confirm</span>
            </div>
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
              title="Undo (Ctrl+Z)"
            >
              <Undo className="w-4 h-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={redo}
              disabled={!canRedo}
              title="Redo (Ctrl+Y)"
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
              title="Force save pending changes (Ctrl+S)"
              className={hasUnsavedChanges ? "text-orange-500 hover:text-orange-600" : ""}
            >
              <Save className="w-4 h-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={resetSettings}
              title={showResetConfirmation ? "Click again to confirm reset" : "Reset to defaults"}
              className={showResetConfirmation ? "text-red-500 hover:text-red-600 animate-pulse" : ""}
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
            {tabs.map(tab => (
              <TabsContent key={tab.id} value={tab.id} className="p-4 space-y-4">
                <Suspense fallback={<SettingsTabSkeleton />}>
                  <LazySettingsTab 
                    tabId={tab.id}
                    category={tab.category}
                    filteredUIDefinition={filteredUIDefinition}
                    updateSettings={updateSettings}
                  />
                </Suspense>
              </TabsContent>
            ))}
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