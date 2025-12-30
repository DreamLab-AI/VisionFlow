// Unified Settings Panel - The single control center for all settings (ENHANCED WITH SEARCH)
import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../../../design-system/components/Tabs';
import { Input } from '../../../design-system/components/Input';
import { Button } from '../../../design-system/components/Button';
import { ScrollArea } from '../../../design-system/components/ScrollArea';
import Settings from 'lucide-react/dist/esm/icons/settings';
import Monitor from 'lucide-react/dist/esm/icons/monitor';
import Palette from 'lucide-react/dist/esm/icons/palette';
import Activity from 'lucide-react/dist/esm/icons/activity';
import Database from 'lucide-react/dist/esm/icons/database';
import Code from 'lucide-react/dist/esm/icons/code';
import Shield from 'lucide-react/dist/esm/icons/shield';
import Headphones from 'lucide-react/dist/esm/icons/headphones';
import Search from 'lucide-react/dist/esm/icons/search';
import Save from 'lucide-react/dist/esm/icons/save';
import RotateCcw from 'lucide-react/dist/esm/icons/rotate-ccw';
import Download from 'lucide-react/dist/esm/icons/download';
import Upload from 'lucide-react/dist/esm/icons/upload';
import Undo from 'lucide-react/dist/esm/icons/undo-2';
import Redo from 'lucide-react/dist/esm/icons/redo-2';
import Brain from 'lucide-react/dist/esm/icons/brain';
import Eye from 'lucide-react/dist/esm/icons/eye';
import BarChart3 from 'lucide-react/dist/esm/icons/bar-chart-3';
import Smartphone from 'lucide-react/dist/esm/icons/smartphone';
import Zap from 'lucide-react/dist/esm/icons/zap';
import Network from 'lucide-react/dist/esm/icons/network';
import HeartPulse from 'lucide-react/dist/esm/icons/heart-pulse';
import { useSettingsStore } from '../../../../store/settingsStore';
import { settingsUIDefinition } from '../../config/settingsUIDefinition';
import { SettingControlComponent } from '../SettingControlComponent';
import { SettingsSearch } from '../SettingsSearch';
import { toast } from '../../../design-system/components/Toast';
import { createLogger } from '../../../../utils/loggerConfig';
import { AgentControlPanel } from './AgentControlPanel';
import { PhysicsControlPanel } from '../../../physics/components/PhysicsControlPanel';
import { SemanticAnalysisPanel } from '../../../analytics/components/SemanticAnalysisPanel';
import { InferencePanel } from '../../../ontology/components/InferencePanel';
import { HealthDashboard } from '../../../monitoring/components/HealthDashboard';
import {
  buildSearchIndex,
  searchSettings,
  SearchableSettingField,
  SearchResult
} from '../../../../utils/settingsSearch';

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
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);


  const settings = useSettingsStore(state => state.settings);
  const updateSettings = useSettingsStore(state => state.updateSettings);
  const resetSettings = useSettingsStore(state => state.resetSettings);
  const exportSettings = useSettingsStore(state => state.exportSettings);
  const importSettings = useSettingsStore(state => state.importSettings);

  // Stub functions for missing store methods
  const saving = false;
  const loading = false;
  const saveSettings = async () => { /* Settings are auto-saved */ };
  const exportToFile = async () => {
    try {
      const json = await exportSettings();
      const blob = new Blob([json], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'settings.json';
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error('Export failed:', e);
    }
  };
  const loadFromFile = async (file: File) => {
    try {
      const text = await file.text();
      await importSettings(text);
    } catch (e) {
      console.error('Import failed:', e);
    }
  };
  const checkUnsavedChanges = () => false;

  
  const canUndo = false;
  const canRedo = false;
  const undo = () => console.log('Undo not yet implemented');
  const redo = () => console.log('Redo not yet implemented');

  
  const searchIndex = useMemo(() => {
    return buildSearchIndex(settingsUIDefinition);
  }, []);

  
  const totalSettingsCount = useMemo(() => searchIndex.length, [searchIndex]);

  
  const handleSearch = useCallback((query: string) => {
    setSearchQuery(query);

    if (!query.trim()) {
      setSearchResults([]);
      return;
    }

    
    const results = searchSettings(searchIndex, query, {
      minScore: 15, 
      maxResults: 100,
      includeAdvanced: true,
      includePowerUser: true
    });

    setSearchResults(results);

    
    if (results.length > 0) {
      console.debug(`Search: "${query}" found ${results.length} results (avg score: ${
        (results.reduce((sum, r) => sum + r.score, 0) / results.length).toFixed(1)
      })`);
    }
  }, [searchIndex]);

  
  useEffect(() => {
    const interval = setInterval(() => {
      setHasUnsavedChanges(checkUnsavedChanges());
    }, 1000);
    return () => clearInterval(interval);
  }, [checkUnsavedChanges]);

  
  const filteredUIDefinition = useMemo(() => {
    if (!searchQuery || searchResults.length === 0) {
      return settingsUIDefinition;
    }

    
    const matchingPaths = new Set(searchResults.map(r => r.path));
    const filtered: typeof settingsUIDefinition = {};

    Object.entries(settingsUIDefinition).forEach(([category, categoryDef]) => {
      const filteredSubsections: typeof categoryDef.subsections = {};

      Object.entries(categoryDef.subsections || {}).forEach(([sectionKey, section]) => {
        const filteredSettings: Record<string, any> = {};
        const sectionTyped = section as { settings?: Record<string, { path: string }> };

        Object.entries(sectionTyped.settings || {}).forEach(([settingKey, setting]) => {
          if (matchingPaths.has(setting.path)) {
            filteredSettings[settingKey] = setting;
          }
        });

        if (Object.keys(filteredSettings).length > 0) {
          filteredSubsections[sectionKey] = {
            ...(section as object),
            settings: filteredSettings
          } as typeof section;
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
  }, [searchQuery, searchResults]);

  
  const handleFileImport = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      await loadFromFile(file);
      event.target.value = ''; 
    }
  };

  
  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: Monitor, category: 'dashboard' },
    { id: 'visualization', label: 'Visualization', icon: Eye, category: 'visualization' },
    { id: 'physics', label: 'Physics Control', icon: Zap, category: 'physics', isCustomPanel: true },
    { id: 'analytics', label: 'Semantic Analysis', icon: Network, category: 'analytics', isCustomPanel: true },
    { id: 'inference', label: 'Ontology Inference', icon: Brain, category: 'inference', isCustomPanel: true },
    { id: 'health', label: 'System Health', icon: HeartPulse, category: 'health', isCustomPanel: true },
    { id: 'agents', label: 'Agents', icon: Brain, category: 'agents', isCustomPanel: true },
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
      {}
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center gap-2">
          <Settings className="w-5 h-5" />
          <h2 className="text-lg font-semibold">Control Center</h2>
          {hasUnsavedChanges && (
            <span className="text-xs text-orange-500 ml-2">(Unsaved changes)</span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {}
          <SettingsSearch
            onSearch={handleSearch}
            resultCount={searchQuery ? searchResults.length : undefined}
            totalCount={totalSettingsCount}
            placeholder={`Search ${totalSettingsCount} settings...`}
            className="w-80"
          />

          {}
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

          {}
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

          {}
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

      {}
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

              if (tab.isCustomPanel) {
                // Render custom panel components
                if (tab.id === 'physics') {
                  return (
                    <TabsContent key={tab.id} value={tab.id} className="p-4">
                      <PhysicsControlPanel />
                    </TabsContent>
                  );
                }
                if (tab.id === 'analytics') {
                  return (
                    <TabsContent key={tab.id} value={tab.id} className="p-4">
                      <SemanticAnalysisPanel />
                    </TabsContent>
                  );
                }
                if (tab.id === 'inference') {
                  return (
                    <TabsContent key={tab.id} value={tab.id} className="p-4">
                      <InferencePanel />
                    </TabsContent>
                  );
                }
                if (tab.id === 'health') {
                  return (
                    <TabsContent key={tab.id} value={tab.id} className="p-4">
                      <HealthDashboard />
                    </TabsContent>
                  );
                }
                if (tab.id === 'agents') {
                  return (
                    <TabsContent key={tab.id} value={tab.id} className="p-4">
                      <AgentControlPanel />
                    </TabsContent>
                  );
                }
                return null;
              }

              const categoryDef = filteredUIDefinition[tab.category];
              if (!categoryDef) return null;

              return (
                <TabsContent key={tab.id} value={tab.id} className="p-4 space-y-4">
                  {}
                  {categoryDef.description && (
                    <div className="text-sm text-muted-foreground mb-4">
                      {categoryDef.description}
                    </div>
                  )}

                  {/* Subsections within category */}
                  {Object.entries(categoryDef.subsections || {}).map(([sectionKey, sectionUntyped]) => {
                    const section = sectionUntyped as { label: string; description?: string; settings?: Record<string, any> };
                    return (
                      <div key={sectionKey} className="space-y-2">
                        <h3 className="text-sm font-semibold">{section.label}</h3>
                        {section.description && (
                          <p className="text-xs text-muted-foreground">{section.description}</p>
                        )}
                        <div className="space-y-2">
                          {Object.values(section.settings || {}).map((settingUntyped) => {
                            const setting = settingUntyped as { path: string; [key: string]: any };
                            return (
                              <SettingControlComponent
                                key={setting.path}
                                path={setting.path}
                                settingDef={setting as any}
                                value={getSettingValue(settings, setting.path)}
                                onChange={(value) => {
                                  updateSettings((draft) => {
                                    setSettingValue(draft, setting.path, value);
                                  });
                                }}
                              />
                            );
                          })}
                        </div>
                      </div>
                    );
                  })}
                </TabsContent>
              );
            })}
          </ScrollArea>
        </Tabs>
      </div>

      {}
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
