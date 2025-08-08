/**
 * Visualization Config Provider
 * Provides centralized configuration management for all visualization components
 * Uses settingsStore as the single source of truth
 */

import React, { createContext, useContext, useCallback } from 'react';
import { useSettingsStore } from '../store/settingsStore';
import { Settings } from '../features/settings/config/settings';

interface VisualizationConfigContextValue {
  settings: Settings;
  updateSettings: (updater: (draft: Settings) => void) => void;
  getSetting: <T>(path: string) => T;
  setSetting: <T>(path: string, value: T) => void;
}

const VisualizationConfigContext = createContext<VisualizationConfigContextValue | null>(null);

export const useVisualizationConfig = () => {
  const context = useContext(VisualizationConfigContext);
  if (!context) {
    throw new Error('useVisualizationConfig must be used within VisualizationConfigProvider');
  }
  return context;
};

export const VisualizationConfigProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const settings = useSettingsStore(state => state.settings);
  const updateSettings = useSettingsStore(state => state.updateSettings);
  const getSetting = useSettingsStore(state => state.get);
  const setSetting = useSettingsStore(state => state.set);
  
  const value: VisualizationConfigContextValue = {
    settings,
    updateSettings,
    getSetting,
    setSetting
  };
  
  return (
    <VisualizationConfigContext.Provider value={value}>
      {children}
    </VisualizationConfigContext.Provider>
  );
};