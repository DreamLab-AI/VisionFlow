import { useEffect, useRef, useState, useContext } from 'react';
import { graphDataManager } from '../../features/graph/managers/graphDataManager';
import { useSettingsStore } from '../../store/settingsStore';
import type { GraphData } from '../../features/graph/managers/graphDataManager';

/**
 * Hook to bridge React state with Babylon.js immersive experience
 * Manages data flow between existing data managers and the 3D scene
 */
export const useImmersiveData = (initialData?: any) => {
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [nodePositions, setNodePositions] = useState<Float32Array | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  // For now, we'll use placeholder data for bots
  // This can be connected later when BotsDataProvider is properly set up
  const botsData = null;

  // Subscribe to settings store
  const settings = useSettingsStore(state => state.settings);

  useEffect(() => {
    console.log('useImmersiveData: Setting up data subscriptions');

    const subscriptions: (() => void)[] = [];

    try {
      // Subscribe to graph data changes
      const graphDataUnsubscribe = graphDataManager.onGraphDataChange((data: GraphData) => {
        console.log('Graph data updated:', data);
        setGraphData(data);
        setIsLoading(false);
        setError(null); // Clear any previous errors
      });
      subscriptions.push(graphDataUnsubscribe);

      // Subscribe to position updates
      const positionUnsubscribe = graphDataManager.onPositionUpdate((positions: Float32Array) => {
        console.log('Position data updated, length:', positions?.length);
        setNodePositions(positions);
      });
      subscriptions.push(positionUnsubscribe);

      // Get initial graph data (manager initializes automatically)
      const initializeData = async () => {
        try {
          // The graph data manager initializes automatically in its constructor
          // Just get current graph data if available
          const currentData = await graphDataManager.getGraphData();
          if (currentData && currentData.nodes && currentData.nodes.length > 0) {
            console.log('Initial graph data available:', currentData);
            setGraphData(currentData);
            setIsLoading(false);
          } else {
            console.log('Waiting for graph data...');
            setIsLoading(false); // Still set loading to false
            // Data will come through subscription
          }
        } catch (err) {
          console.error('Failed to get initial graph data:', err);
          // Don't set error here, just wait for data through subscriptions
          // Many components work without initial data
          setIsLoading(false);
        }
      };

      initializeData();

    } catch (err) {
      console.error('Error setting up data subscriptions:', err);
      setError('Failed to set up data connections');
      setIsLoading(false);
    }

    return () => {
      // Cleanup subscriptions
      subscriptions.forEach(unsub => {
        if (typeof unsub === 'function') {
          unsub();
        }
      });
    };
  }, []);

  const updateNodePosition = (nodeId: string, position: { x: number; y: number; z: number }) => {
    // Update node position in the physics simulation
    if (graphDataManager) {
      graphDataManager.updateUserDrivenNodePosition(nodeId, position);
    }
  };

  const selectNode = (nodeId: string | null) => {
    setSelectedNode(nodeId);
    if (nodeId && graphDataManager) {
      // Optionally highlight the node in the main graph
      graphDataManager.highlightNode(nodeId);
    }
  };

  const pinNode = (nodeId: string) => {
    if (graphDataManager) {
      graphDataManager.pinNode(nodeId);
    }
  };

  const unpinNode = (nodeId: string) => {
    if (graphDataManager) {
      graphDataManager.unpinNode(nodeId);
    }
  };

  return {
    graphData,
    nodePositions,
    botsData,
    settings,
    isLoading,
    error,
    selectedNode,
    updateNodePosition,
    selectNode,
    pinNode,
    unpinNode
  };
};