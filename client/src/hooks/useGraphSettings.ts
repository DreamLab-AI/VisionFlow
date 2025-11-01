import { useSelectiveSetting, useSettingSetter } from './useSelectiveSettingsStore';
import { GraphSettings } from '../features/settings/config/settings';


export function useGraphSettings(graphName: 'logseq' | 'visionflow'): GraphSettings | undefined {
  
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


export function useUpdateGraphSettings(graphName: 'logseq' | 'visionflow') {
  const { immediateSet } = useSettingSetter();
  
  return (updater: (draft: GraphSettings) => void) => {
    immediateSet((draft) => {
      
      if (!draft.visualisation.graphs) {
        draft.visualisation.graphs = {
          logseq: {} as GraphSettings,
          visionflow: {} as GraphSettings,
        };
      }
      
      
      updater(draft.visualisation.graphs[graphName]);
    });
  };
}


export function useGraphSetting<T>(
  graphName: 'logseq' | 'visionflow',
  path: string
): T {
  
  const fullPath = `visualisation.graphs.${graphName}.${path}`;
  
  const value = useSelectiveSetting<T>(fullPath, {
    enableCache: true,
    enableDeduplication: true,
    fallbackToStore: true
  });
  
  return value;
}