import { useEffect, useRef, useState, useContext } from 'react';
import { graphDataManager } from '../../features/graph/managers/graphDataManager';
import { useSettingsStore } from '../../store/settingsStore';
import type { GraphData } from '../../features/graph/managers/graphDataManager';


export const useImmersiveData = (initialData?: any) => {
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [nodePositions, setNodePositions] = useState<Float32Array | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  
  
  const botsData = null;

  
  const settings = useSettingsStore(state => state.settings);

  useEffect(() => {
    console.log('useImmersiveData: Setting up data subscriptions');

    const subscriptions: (() => void)[] = [];

    try {
      
      const graphDataUnsubscribe = graphDataManager.onGraphDataChange((data: GraphData) => {
        console.log('Graph data updated:', data);
        setGraphData(data);
        setIsLoading(false);
        setError(null); 
      });
      subscriptions.push(graphDataUnsubscribe);

      
      const positionUnsubscribe = graphDataManager.onPositionUpdate((positions: ArrayBuffer | Float32Array) => {
        const floatPositions = positions instanceof Float32Array ? positions : new Float32Array(positions);
        console.log('Position data updated, length:', floatPositions?.length);
        setNodePositions(floatPositions);
      });
      subscriptions.push(positionUnsubscribe);

      
      const initializeData = async () => {
        try {
          
          
          const currentData = await graphDataManager.getGraphData();
          if (currentData && currentData.nodes && currentData.nodes.length > 0) {
            console.log('Initial graph data available:', currentData);
            setGraphData(currentData);
            setIsLoading(false);
          } else {
            console.log('Waiting for graph data...');
            setIsLoading(false); 
            
          }
        } catch (err) {
          console.error('Failed to get initial graph data:', err);
          
          
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
      
      subscriptions.forEach(unsub => {
        if (typeof unsub === 'function') {
          unsub();
        }
      });
    };
  }, []);

  const updateNodePosition = (nodeId: string, position: { x: number; y: number; z: number }) => {

    if (graphDataManager) {
      // @ts-ignore - method may not exist in current GraphDataManager interface
      graphDataManager.updateUserDrivenNodePosition?.(nodeId, position) ??
        // @ts-ignore - updateNodePositions signature may vary
        graphDataManager.updateNodePositions?.([{ id: nodeId, position }] as any);
    }
  };

  const selectNode = (nodeId: string | null) => {
    setSelectedNode(nodeId);
    if (nodeId && graphDataManager) {
      // @ts-ignore - method may not exist in current GraphDataManager interface
      graphDataManager.highlightNode?.(nodeId);
    }
  };

  const pinNode = (nodeId: string) => {
    if (graphDataManager) {
      // @ts-ignore - method may not exist in current GraphDataManager interface
      graphDataManager.pinNode?.(nodeId);
    }
  };

  const unpinNode = (nodeId: string) => {
    if (graphDataManager) {
      // @ts-ignore - method may not exist in current GraphDataManager interface
      graphDataManager.unpinNode?.(nodeId);
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