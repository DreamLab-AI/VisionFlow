import { useEffect, useState } from 'react';
import { graphDataManager } from '../../features/graph/managers/graphDataManager';
import { useSettingsStore } from '../../store/settingsStore';
import type { GraphData } from '../../features/graph/managers/graphDataManager';
import { createLogger } from '../../utils/loggerConfig';

const logger = createLogger('useImmersiveData');


export const useImmersiveData = (initialData?: any) => {
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [nodePositions, setNodePositions] = useState<Float32Array | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  
  
  const botsData = null;

  
  const settings = useSettingsStore(state => state.settings);

  useEffect(() => {
    logger.debug('Setting up data subscriptions');

    const subscriptions: (() => void)[] = [];

    try {
      
      const graphDataUnsubscribe = graphDataManager.onGraphDataChange((data: GraphData) => {
        logger.debug('Graph data updated:', data);
        setGraphData(data);
        setIsLoading(false);
        setError(null); 
      });
      subscriptions.push(graphDataUnsubscribe);

      
      const positionUnsubscribe = graphDataManager.onPositionUpdate((positions: ArrayBuffer | Float32Array) => {
        const floatPositions = positions instanceof Float32Array ? positions : new Float32Array(positions);
        logger.debug('Position data updated, length:', floatPositions?.length);
        setNodePositions(floatPositions);
      });
      subscriptions.push(positionUnsubscribe);

      
      const initializeData = async () => {
        try {
          
          
          const currentData = await graphDataManager.getGraphData();
          if (currentData && currentData.nodes && currentData.nodes.length > 0) {
            logger.debug('Initial graph data available:', currentData);
            setGraphData(currentData);
            setIsLoading(false);
          } else {
            logger.debug('Waiting for graph data...');
            setIsLoading(false); 
            
          }
        } catch (err) {
          logger.error('Failed to get initial graph data:', err);
          
          
          setIsLoading(false);
        }
      };

      initializeData();

    } catch (err) {
      logger.error('Error setting up data subscriptions:', err);
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

  const updateNodePosition = (_nodeId: string, _position: { x: number; y: number; z: number }) => {
    // updateUserDrivenNodePosition and pinNode/unpinNode live on graphWorkerProxy,
    // not graphDataManager. Immersive mode should use the worker proxy directly
    // when this integration is wired up.
    logger.warn('updateNodePosition called but graphDataManager does not support direct position updates');
  };

  const selectNode = (nodeId: string | null) => {
    setSelectedNode(nodeId);
    if (nodeId) {
      logger.debug('Node selected:', nodeId);
    }
  };

  const pinNode = (_nodeId: string) => {
    logger.warn('pinNode called but graphDataManager does not support pinning directly');
  };

  const unpinNode = (_nodeId: string) => {
    logger.warn('unpinNode called but graphDataManager does not support unpinning directly');
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