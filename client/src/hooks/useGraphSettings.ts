import { useSelectiveSetting, useSettingSetter } from './useSelectiveSettingsStore';
import { GraphSettings } from '../features/settings/config/settings';

/**
 * Hook to get settings for a specific graph (optimized with selective subscription)
 * @param graphName - Either 'logseq' or 'visionflow'
 * @returns The settings for the specified graph
 */
export function useGraphSettings(graphName: 'logseq' | 'visionflow'): GraphSettings | undefined {
  // Use selective hook to subscribe only to the specific graph settings path
  const graphSettings = useSelectiveSetting<GraphSettings | undefined>(
    `visualisation.graphs.${graphName}`,
    {
      enableCache: true,
      enableDeduplication: true,
      fallbackToStore: true
    }
  );
  
  return graphSettings;
}

/**
 * Hook to update settings for a specific graph (optimized with batched updates)
 * @param graphName - Either 'logseq' or 'visionflow'
 * @returns A function to update the graph settings
 */
export function useUpdateGraphSettings(graphName: 'logseq' | 'visionflow') {
  const { immediateSet } = useSettingSetter();
  
  return (updater: (draft: GraphSettings) => void) => {
    immediateSet((draft) => {
      // Ensure the graphs structure exists
      if (!draft.visualisation.graphs) {
        draft.visualisation.graphs = {
          logseq: {} as GraphSettings,
          visionflow: {} as GraphSettings,
        };
      }
      
      // Update the specific graph settings
      updater(draft.visualisation.graphs[graphName]);
    });
  };
}

/**
 * Hook to get a specific setting from a graph (highly optimized with direct path subscription)
 * @param graphName - Either 'logseq' or 'visionflow'
 * @param path - Path within the graph settings (e.g., 'nodes.baseColor')
 * @returns The value of the setting
 */
export function useGraphSetting<T>(
  graphName: 'logseq' | 'visionflow',
  path: string
): T {
  // Use selective hook with the complete path for maximum performance
  const fullPath = `visualisation.graphs.${graphName}.${path}`;
  
  const value = useSelectiveSetting<T>(fullPath, {
    enableCache: true,
    enableDeduplication: true,
    fallbackToStore: true
  });
  
  return value;
}