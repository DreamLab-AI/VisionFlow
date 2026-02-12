// Unified Settings Panel - The single control center for all settings (ENHANCED WITH SEARCH)
import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../../../design-system/components/Tabs';
import { Input } from '../../../design-system/components/Input';
import { Button } from '../../../design-system/components/Button';
import { ScrollArea } from '../../../design-system/components/ScrollArea';
import { Settings, Activity, Database, Code, Search, Save, RotateCcw, Download, Upload, Undo2 as Undo, Redo2 as Redo, Eye, Smartphone, Zap, Network, Brain, HeartPulse } from 'lucide-react';
import { useSettingsStore } from '../../../../store/settingsStore';
import { settingsUIDefinition } from '../../config/settingsUIDefinition';
import { SettingControlComponent } from '../SettingControlComponent';
import { SettingsSearch } from '../SettingsSearch';
import { toast } from '../../../design-system/components/Toast';
import { createLogger } from '../../../../utils/loggerConfig';

const logger = createLogger('SettingsPanelRedesign');

import { AgentControlPanel } from './AgentControlPanel';
import { PhysicsControlPanel } from '../../../physics/components/PhysicsControlPanel';
import { SemanticAnalysisPanel } from '../../../analytics/components/SemanticAnalysisPanel';
import { InferencePanel } from '../../../ontology/components/InferencePanel';
import { HealthDashboard } from '../../../monitoring/components/HealthDashboard';
import { PerformanceControlPanel } from './PerformanceControlPanel';
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

  const batchUpdate = useSettingsStore(state => state.batchUpdate);

  // Settings are auto-saved to server on each change
  const saving = false;
  const loading = false;
  const saveSettings = async () => { /* Settings are auto-saved via updateSettings */ };
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
      logger.error('Export failed:', e);
    }
  };
  const loadFromFile = async (file: File) => {
    try {
      const text = await file.text();
      await importSettings(text);
    } catch (e) {
      logger.error('Import failed:', e);
    }
  };
  const checkUnsavedChanges = () => false;

  // Undo/redo history using snapshots of changed paths
  const MAX_HISTORY = 50;
  const historyRef = useRef<{ past: string[]; future: string[] }>({ past: [], future: [] });
  const lastSettingsRef = useRef<string>(JSON.stringify(settings));
  const isUndoRedoRef = useRef(false);
  const [historyVersion, setHistoryVersion] = useState(0);

  // Track settings changes to build undo history
  useEffect(() => {
    if (isUndoRedoRef.current) {
      isUndoRedoRef.current = false;
      lastSettingsRef.current = JSON.stringify(settings);
      return;
    }
    const currentSnapshot = JSON.stringify(settings);
    if (currentSnapshot !== lastSettingsRef.current) {
      historyRef.current.past.push(lastSettingsRef.current);
      historyRef.current.future = [];
      if (historyRef.current.past.length > MAX_HISTORY) {
        historyRef.current.past.shift();
      }
      lastSettingsRef.current = currentSnapshot;
      setHistoryVersion(v => v + 1);
    }
  }, [settings]);

  const canUndo = historyRef.current.past.length > 0;
  const canRedo = historyRef.current.future.length > 0;

  const undo = useCallback(() => {
    if (historyRef.current.past.length === 0) return;
    const previous = historyRef.current.past.pop()!;
    historyRef.current.future.push(JSON.stringify(settings));
    isUndoRedoRef.current = true;
    const restored = JSON.parse(previous);
    updateSettings(() => Object.assign({}, restored));
    setHistoryVersion(v => v + 1);
    logger.debug('Settings undone');
  }, [settings, updateSettings]);

  const redo = useCallback(() => {
    if (historyRef.current.future.length === 0) return;
    const next = historyRef.current.future.pop()!;
    historyRef.current.past.push(JSON.stringify(settings));
    isUndoRedoRef.current = true;
    const restored = JSON.parse(next);
    updateSettings(() => Object.assign({}, restored));
    setHistoryVersion(v => v + 1);
    logger.debug('Settings redone');
  }, [settings, updateSettings]);

  
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
      logger.debug(`Search: "${query}" found ${results.length} results (avg score: ${
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

  // Consolidated from 14 → 9 tabs.
  // Removed: Dashboard (stub → merged into Performance), Auth (→ System),
  //   System Health (→ Performance), Ontology Inference (→ Analytics).
  // Distinct icons: no more triple Brain.
  const tabs = [
    { id: 'visualization', label: 'Visualization', icon: Eye, category: 'visualization' },
    { id: 'physics', label: 'Physics & Layout', icon: Zap, category: 'physics', isCustomPanel: true },
    { id: 'performance', label: 'Performance', icon: Activity, category: 'performance', isCustomPanel: true },
    { id: 'analytics', label: 'Analytics', icon: Network, category: 'analytics', isCustomPanel: true },
    { id: 'agents', label: 'Agents', icon: HeartPulse, category: 'agents', isCustomPanel: true },
    { id: 'xr', label: 'XR/AR', icon: Smartphone, category: 'xr' },
    { id: 'system', label: 'System', icon: Settings, category: 'system' },
    { id: 'ai', label: 'AI Services', icon: Brain, category: 'ai' },
    { id: 'developer', label: 'Developer', icon: Code, category: 'developer' },
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
                if (tab.id === 'physics') {
                  return (
                    <TabsContent key={tab.id} value={tab.id} className="p-4">
                      <PhysicsControlPanel />
                    </TabsContent>
                  );
                }
                if (tab.id === 'performance') {
                  // Merged: Performance monitoring + Quality Gates + System Health
                  const perfDef = filteredUIDefinition['performance'];
                  const qgDef = filteredUIDefinition['qualityGates'];
                  return (
                    <TabsContent key={tab.id} value={tab.id} className="p-4 space-y-6">
                      <PerformanceControlPanel />
                      <hr className="border-border" />
                      <HealthDashboard />
                      {/* Quality Gates settings */}
                      {qgDef && Object.entries(qgDef.subsections || {}).map(([sectionKey, sectionUntyped]) => {
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
                      {/* Performance monitoring settings */}
                      {perfDef && Object.entries(perfDef.subsections || {}).map(([sectionKey, sectionUntyped]) => {
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
                }
                if (tab.id === 'analytics') {
                  // Merged: SemanticAnalysisPanel + InferencePanel + Node Filter settings
                  const analyticsDef = filteredUIDefinition['analytics'];
                  return (
                    <TabsContent key={tab.id} value={tab.id} className="p-4 space-y-6">
                      <SemanticAnalysisPanel />
                      <hr className="border-border" />
                      <h3 className="text-sm font-semibold">Ontology Inference</h3>
                      <InferencePanel />
                      {/* Node Filter and metrics settings from analytics category */}
                      {analyticsDef && Object.entries(analyticsDef.subsections || {}).map(([sectionKey, sectionUntyped]) => {
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
