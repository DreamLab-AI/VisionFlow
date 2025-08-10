// Note: zustand needs to be installed: npm install zustand
// import { create } from 'zustand';
// import { devtools, subscribeWithSelector } from 'zustand/middleware';
import { Settings, PhysicsSettings, NodeSettings, EdgeSettings, LabelSettings } from '../config/settings';
import { MinimalSettingsAPI } from '../config/minimalSettings';

interface SettingsState {
  // Current settings
  settings: Settings | null;
  
  // Loading states
  loading: boolean;
  saving: boolean;
  error: string | null;
  
  // Current graph
  currentGraph: 'logseq' | 'visionflow';
  
  // Actions
  loadSettings: () => Promise<void>;
  updateSettings: (updates: Partial<Settings>) => Promise<void>;
  updatePhysics: (physics: Partial<PhysicsSettings>) => Promise<void>;
  updateNodes: (nodes: Partial<NodeSettings>) => Promise<void>;
  updateEdges: (edges: Partial<EdgeSettings>) => Promise<void>;
  updateLabels: (labels: Partial<LabelSettings>) => Promise<void>;
  switchGraph: (graph: 'logseq' | 'visionflow') => Promise<void>;
  clearError: () => void;
}

// Temporary implementation until zustand is installed
let currentState: SettingsState = {
  settings: null,
  loading: false,
  saving: false,
  error: null,
  currentGraph: 'logseq',
  loadSettings: async () => {},
  updateSettings: async () => {},
  updatePhysics: async () => {},
  updateNodes: async () => {},
  updateEdges: async () => {},
  updateLabels: async () => {},
  switchGraph: async () => {},
  clearError: () => {},
};

export const useSettingsStore = () => currentState;

// Original zustand implementation (to be restored after installing zustand):
/*
export const useSettingsStore = create<SettingsState>()(
  devtools(
    subscribeWithSelector((set, get) => ({
      settings: null,
      loading: false,
      saving: false,
      error: null,
      currentGraph: 'logseq',
      
      loadSettings: async () => {
        set({ loading: true, error: null });
        try {
          const response = await fetch('/api/settings');
          if (!response.ok) throw new Error('Failed to load settings');
          const settings = await response.json();
          set({ settings, loading: false });
        } catch (error) {
          set({ 
            error: error instanceof Error ? error.message : 'Failed to load settings',
            loading: false 
          });
        }
      },
      
      updateSettings: async (updates) => {
        const { settings, currentGraph } = get();
        if (!settings) return;
        
        set({ saving: true, error: null });
        try {
          const response = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(updates),
          });
          
          if (!response.ok) throw new Error('Failed to update settings');
          const updatedSettings = await response.json();
          set({ settings: updatedSettings, saving: false });
        } catch (error) {
          set({ 
            error: error instanceof Error ? error.message : 'Failed to update settings',
            saving: false 
          });
        }
      },
      
      updatePhysics: async (physics) => {
        const { settings, currentGraph } = get();
        if (!settings) return;
        
        set({ saving: true, error: null });
        try {
          // Update the physics for the current graph
          const updates = {
            visualisation: {
              graphs: {
                [currentGraph]: {
                  physics: {
                    ...settings.visualisation.graphs[currentGraph].physics,
                    ...physics,
                  }
                }
              }
            }
          };
          
          const response = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(updates),
          });
          
          if (!response.ok) throw new Error('Failed to update physics');
          const updatedSettings = await response.json();
          set({ settings: updatedSettings, saving: false });
        } catch (error) {
          set({ 
            error: error instanceof Error ? error.message : 'Failed to update physics',
            saving: false 
          });
        }
      },
      
      updateNodes: async (nodes) => {
        const { settings, currentGraph } = get();
        if (!settings) return;
        
        set({ saving: true, error: null });
        try {
          const updates = {
            visualisation: {
              graphs: {
                [currentGraph]: {
                  nodes: {
                    ...settings.visualisation.graphs[currentGraph].nodes,
                    ...nodes,
                  }
                }
              }
            }
          };
          
          const response = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(updates),
          });
          
          if (!response.ok) throw new Error('Failed to update nodes');
          const updatedSettings = await response.json();
          set({ settings: updatedSettings, saving: false });
        } catch (error) {
          set({ 
            error: error instanceof Error ? error.message : 'Failed to update nodes',
            saving: false 
          });
        }
      },
      
      updateEdges: async (edges) => {
        const { settings, currentGraph } = get();
        if (!settings) return;
        
        set({ saving: true, error: null });
        try {
          const updates = {
            visualisation: {
              graphs: {
                [currentGraph]: {
                  edges: {
                    ...settings.visualisation.graphs[currentGraph].edges,
                    ...edges,
                  }
                }
              }
            }
          };
          
          const response = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(updates),
          });
          
          if (!response.ok) throw new Error('Failed to update edges');
          const updatedSettings = await response.json();
          set({ settings: updatedSettings, saving: false });
        } catch (error) {
          set({ 
            error: error instanceof Error ? error.message : 'Failed to update edges',
            saving: false 
          });
        }
      },
      
      updateLabels: async (labels) => {
        const { settings, currentGraph } = get();
        if (!settings) return;
        
        set({ saving: true, error: null });
        try {
          const updates = {
            visualisation: {
              graphs: {
                [currentGraph]: {
                  labels: {
                    ...settings.visualisation.graphs[currentGraph].labels,
                    ...labels,
                  }
                }
              }
            }
          };
          
          const response = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(updates),
          });
          
          if (!response.ok) throw new Error('Failed to update labels');
          const updatedSettings = await response.json();
          set({ settings: updatedSettings, saving: false });
        } catch (error) {
          set({ 
            error: error instanceof Error ? error.message : 'Failed to update labels',
            saving: false 
          });
        }
      },
      
      switchGraph: async (graph) => {
        set({ currentGraph: graph, saving: true, error: null });
        try {
          const response = await fetch(`/api/settings/graph/${graph}`, {
            method: 'POST',
          });
          
          if (!response.ok) throw new Error('Failed to switch graph');
          await get().loadSettings(); // Reload settings after switch
          set({ saving: false });
        } catch (error) {
          set({ 
            error: error instanceof Error ? error.message : 'Failed to switch graph',
            saving: false 
          });
        }
      },
      
      clearError: () => set({ error: null }),
    })),
    {
      name: 'settings-store',
    }
  )
);
*/