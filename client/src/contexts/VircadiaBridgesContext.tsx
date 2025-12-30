

import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { useVircadia } from './VircadiaContext';
import { BotsVircadiaBridge } from '../services/bridges/BotsVircadiaBridge';
import { GraphVircadiaBridge, type UserSelectionEvent, type AnnotationEvent } from '../services/bridges/GraphVircadiaBridge';
import { EntitySyncManager } from '../services/vircadia/EntitySyncManager';
// Note: AvatarManager removed - requires camera which context doesn't have
import { CollaborativeGraphSync } from '../services/vircadia/CollaborativeGraphSync';
import { createLogger } from '../utils/loggerConfig';
import type { BotsAgent, BotsEdge } from '../features/bots/types/BotsTypes';

const logger = createLogger('VircadiaBridgesContext');

interface VircadiaBridgesContextValue {
  
  botsBridge: BotsVircadiaBridge | null;
  syncAgentsToVircadia: (agents: BotsAgent[], edges: BotsEdge[]) => void;
  startBotsAutoSync: (getAgentsCallback: () => { agents: BotsAgent[]; edges: BotsEdge[] }) => void;
  stopBotsAutoSync: () => void;

  
  graphBridge: GraphVircadiaBridge | null;
  syncGraphToVircadia: (nodes: any[], edges: any[]) => void;
  broadcastSelection: (nodeIds: string[]) => void;
  addAnnotation: (nodeId: string, text: string, position: { x: number; y: number; z: number }) => Promise<string>;
  removeAnnotation: (annotationId: string) => Promise<void>;
  activeUsers: Array<{ userId: string; username: string; selectedNodes: string[] }>;
  annotations: AnnotationEvent[];

  
  isInitialized: boolean;
  error: Error | null;
}

const VircadiaBridgesContext = createContext<VircadiaBridgesContextValue | null>(null);

interface VircadiaBridgesProviderProps {
  children: React.ReactNode;
  scene?: any; 
  enableBotsBridge?: boolean;
  enableGraphBridge?: boolean;
}

export const VircadiaBridgesProvider: React.FC<VircadiaBridgesProviderProps> = ({
  children,
  scene,
  enableBotsBridge = true,
  enableGraphBridge = true
}) => {
  const { client, isConnected } = useVircadia();
  const [botsBridge, setBotsBridge] = useState<BotsVircadiaBridge | null>(null);
  const [graphBridge, setGraphBridge] = useState<GraphVircadiaBridge | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [activeUsers, setActiveUsers] = useState<Array<{ userId: string; username: string; selectedNodes: string[] }>>([]);
  const [annotations, setAnnotations] = useState<AnnotationEvent[]>([]);

  
  useEffect(() => {
    if (!client || !isConnected) {
      setIsInitialized(false);
      return;
    }

    const initializeBridges = async () => {
      try {
        logger.info('Initializing Vircadia bridges...');

        
        if (enableBotsBridge) {
          // EntitySyncManager takes (client, config?) - not scene
          const entitySync = new EntitySyncManager(client, {
            syncGroup: 'public.NORMAL',
            enableRealTimePositions: true
          });

          // AvatarManager requires camera - pass null since we don't have it here
          // BotsVircadiaBridge accepts null for avatars
          const bBridge = new BotsVircadiaBridge(client, entitySync, null);
          await bBridge.initialize();
          setBotsBridge(bBridge);

          logger.info('BotsVircadiaBridge initialized');
        }

        
        if (enableGraphBridge && scene) {
          const collab = new CollaborativeGraphSync(scene, client);
          const gBridge = new GraphVircadiaBridge(scene, client, collab);
          await gBridge.initialize();

          
          gBridge.onRemoteSelection((event: UserSelectionEvent) => {
            logger.debug('Remote selection:', event);
            updateActiveUsers();
          });

          gBridge.onAnnotation((event: AnnotationEvent) => {
            logger.info('Remote annotation:', event);
            updateAnnotations();
          });

          setGraphBridge(gBridge);

          logger.info('GraphVircadiaBridge initialized');
        }

        setIsInitialized(true);
        setError(null);
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        logger.error('Failed to initialize bridges:', error);
        setError(error);
        setIsInitialized(false);
      }
    };

    initializeBridges();

    
    return () => {
      if (botsBridge) {
        botsBridge.dispose();
        setBotsBridge(null);
      }
      if (graphBridge) {
        graphBridge.dispose();
        setGraphBridge(null);
      }
      setIsInitialized(false);
    };
  }, [client, isConnected, scene, enableBotsBridge, enableGraphBridge]);

  
  const updateActiveUsers = useCallback(() => {
    if (graphBridge) {
      setActiveUsers(graphBridge.getActiveUsers());
    }
  }, [graphBridge]);

  
  const updateAnnotations = useCallback(() => {
    if (graphBridge) {
      setAnnotations(graphBridge.getAnnotations());
    }
  }, [graphBridge]);

  
  useEffect(() => {
    if (!graphBridge) return;

    const interval = setInterval(updateActiveUsers, 2000);
    return () => clearInterval(interval);
  }, [graphBridge, updateActiveUsers]);

  
  const syncAgentsToVircadia = useCallback((agents: BotsAgent[], edges: BotsEdge[]) => {
    if (botsBridge) {
      botsBridge.syncAgentsToVircadia(agents, edges);
    }
  }, [botsBridge]);

  const startBotsAutoSync = useCallback((getAgentsCallback: () => { agents: BotsAgent[]; edges: BotsEdge[] }) => {
    if (botsBridge) {
      botsBridge.startAutoSync(getAgentsCallback);
    }
  }, [botsBridge]);

  const stopBotsAutoSync = useCallback(() => {
    if (botsBridge) {
      botsBridge.stopAutoSync();
    }
  }, [botsBridge]);

  
  const syncGraphToVircadia = useCallback((nodes: any[], edges: any[]) => {
    if (graphBridge) {
      graphBridge.syncGraphToVircadia(nodes, edges);
    }
  }, [graphBridge]);

  const broadcastSelection = useCallback((nodeIds: string[]) => {
    if (graphBridge) {
      graphBridge.broadcastLocalSelection(nodeIds);
    }
  }, [graphBridge]);

  const addAnnotation = useCallback(async (
    nodeId: string,
    text: string,
    position: { x: number; y: number; z: number }
  ): Promise<string> => {
    if (!graphBridge) {
      throw new Error('Graph bridge not initialized');
    }
    const id = await graphBridge.addAnnotation(nodeId, text, position);
    updateAnnotations();
    return id;
  }, [graphBridge, updateAnnotations]);

  const removeAnnotation = useCallback(async (annotationId: string): Promise<void> => {
    if (graphBridge) {
      await graphBridge.removeAnnotation(annotationId);
      updateAnnotations();
    }
  }, [graphBridge, updateAnnotations]);

  const value: VircadiaBridgesContextValue = {
    botsBridge,
    syncAgentsToVircadia,
    startBotsAutoSync,
    stopBotsAutoSync,
    graphBridge,
    syncGraphToVircadia,
    broadcastSelection,
    addAnnotation,
    removeAnnotation,
    activeUsers,
    annotations,
    isInitialized,
    error
  };

  return (
    <VircadiaBridgesContext.Provider value={value}>
      {children}
    </VircadiaBridgesContext.Provider>
  );
};


export const useVircadiaBridges = (): VircadiaBridgesContextValue => {
  const context = useContext(VircadiaBridgesContext);
  if (!context) {
    throw new Error('useVircadiaBridges must be used within VircadiaBridgesProvider');
  }
  return context;
};


export const useBotsBridge = () => {
  const { botsBridge, syncAgentsToVircadia, startBotsAutoSync, stopBotsAutoSync, isInitialized } = useVircadiaBridges();
  return { botsBridge, syncAgentsToVircadia, startBotsAutoSync, stopBotsAutoSync, isInitialized };
};


export const useGraphBridge = () => {
  const {
    graphBridge,
    syncGraphToVircadia,
    broadcastSelection,
    addAnnotation,
    removeAnnotation,
    activeUsers,
    annotations,
    isInitialized
  } = useVircadiaBridges();

  return {
    graphBridge,
    syncGraphToVircadia,
    broadcastSelection,
    addAnnotation,
    removeAnnotation,
    activeUsers,
    annotations,
    isInitialized
  };
};
