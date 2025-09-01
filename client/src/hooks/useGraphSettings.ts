import { useSelectiveSetting, useSettingSetter } from './useSelectiveSettingsStore';
import { GraphSettings } from '../types/generated/settings';
import { SettingsPath } from '../types/generated/settings';

/**
 * Hook to get settings for a specific graph using selective loading
 * @param graphName - Either 'logseq' or 'visionflow'
 * @returns The settings for the specified graph
 */
export function useGraphSettings(graphName: 'logseq' | 'visionflow'): GraphSettings | undefined {
  const path = `visualisation.graphs.${graphName}` as SettingsPath;
  return useSelectiveSetting<GraphSettings>(path);
}

/**
 * Hook to update settings for a specific graph using granular updates
 * @param graphName - Either 'logseq' or 'visionflow'
 * @returns Functions to set individual graph settings
 */
export function useUpdateGraphSettings(graphName: 'logseq' | 'visionflow') {
  const { set, batchSet } = useSettingSetter();
  
  return {
    set: <T>(settingPath: string, value: T) => {
      const fullPath = `visualisation.graphs.${graphName}.${settingPath}` as SettingsPath;
      return set(fullPath, value);
    },
    batchSet: (updates: Record<string, any>) => {
      const pathValuePairs = Object.entries(updates).map(([settingPath, value]) => ({
        path: `visualisation.graphs.${graphName}.${settingPath}`,
        value
      }));
      return batchSet(pathValuePairs);
    }
  };
}

/**
 * Hook to get a specific setting from a graph using selective loading
 * @param graphName - Either 'logseq' or 'visionflow'
 * @param settingPath - Path within the graph settings (e.g., 'nodes.baseColor')
 * @returns The value of the setting
 */
export function useGraphSetting<T>(
  graphName: 'logseq' | 'visionflow',
  settingPath: string
): T | undefined {
  const fullPath = `visualisation.graphs.${graphName}.${settingPath}` as SettingsPath;
  return useSelectiveSetting<T>(fullPath);
}

/**
 * Hook to set a specific setting for a graph
 * @param graphName - Either 'logseq' or 'visionflow' 
 * @param settingPath - Path within the graph settings (e.g., 'nodes.baseColor')
 * @returns Function to set the setting value
 */
export function useSetGraphSetting<T>(
  graphName: 'logseq' | 'visionflow',
  settingPath: string
) {
  const { set } = useSettingSetter();
  const fullPath = `visualisation.graphs.${graphName}.${settingPath}` as SettingsPath;
  
  return (value: T) => set(fullPath, value);
}