import React, { useEffect } from 'react';
import { createLogger, createErrorMetadata } from '../utils/loggerConfig';
import { debugState } from '../utils/clientDebugState';
import { useSettingsStore } from '../store/settingsStore';
import WebSocketService from '../services/WebSocketService';
import { graphWorkerProxy } from '../features/graph/managers/graphWorkerProxy';
import { graphDataManager } from '../features/graph/managers/graphDataManager';
import { innovationManager } from '../features/graph/innovations/index';

// Load and initialize all services
const loadServices = async (): Promise<void> => {
  if (debugState.isEnabled()) {
    logger.info('Initializing services...');
  }

  try {
    
    if (debugState.isEnabled()) {
      logger.info('Using Nostr authentication system');
    }

    
    try {
      console.log('[AppInitializer] Starting Innovation Manager initialization...');
      const initPromise = innovationManager.initialize({
        enableSync: true,
        enableComparison: true,
        enableAnimations: true,
        enableAI: true,
        enableAdvancedInteractions: true,
        performanceMode: 'balanced'
      });
      
      
      const timeoutPromise = new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Innovation Manager initialization timeout')), 5000)
      );
      
      await Promise.race([initPromise, timeoutPromise]);

      console.log('[AppInitializer] Innovation Manager initialized successfully');
      if (debugState.isEnabled()) {
        logger.info('Innovation Manager initialized successfully');
        const status = innovationManager.getStatus();
        logger.debug('Innovation Manager status:', status);
      }
    } catch (innovationError) {
      console.error('[AppInitializer] Innovation Manager initialization failed:', innovationError);
      logger.error('Error initializing Innovation Manager:', createErrorMetadata(innovationError));
      
    }

  } catch (error) {
    logger.error('Error initializing services:', createErrorMetadata(error));
  }
}

const logger = createLogger('AppInitializer');

interface AppInitializerProps {
  onInitialized: () => void;
  onError: (error: Error) => void;
}

const AppInitializer: React.FC<AppInitializerProps> = ({ onInitialized, onError }) => {
  const { settings, initialize } = useSettingsStore();

  useEffect(() => {
    const initApp = async () => {
      
      await loadServices();

      if (debugState.isEnabled()) {
        logger.info('Starting application initialization...');
        }

        try {
          console.log('[AppInitializer] Step 1: Initializing graphWorkerProxy');
          
          await graphWorkerProxy.initialize();
          console.log('[AppInitializer] Step 2: graphWorkerProxy initialized, calling settings initialize');
          const settings = await initialize();
          console.log('[AppInitializer] Step 3: Settings initialized:', settings ? 'success' : 'null');

          
          if (settings?.system?.debug) {
            try {
              const debugSettings = settings.system.debug;
              debugState.enableDebug(debugSettings.enabled);
              if (debugSettings.enabled) {
                debugState.enableDataDebug(debugSettings.enableDataDebug);
                debugState.enablePerformanceDebug(debugSettings.enablePerformanceDebug);
              }
            } catch (debugError) {
              logger.warn('Error applying debug settings:', createErrorMetadata(debugError));
            }
          }

          
          if (typeof WebSocketService !== 'undefined' && typeof graphDataManager !== 'undefined') {
            try {
              
              await initializeWebSocket(settings);
              
            } catch (wsError) {
              logger.error('WebSocket initialization failed, continuing with UI only:', createErrorMetadata(wsError));
              
            }
          } else {
            logger.warn('WebSocket services not available, continuing with UI only');
          }

          
          try {
            console.log('[AppInitializer] About to fetch initial graph data via REST API');
            logger.info('Fetching initial graph data via REST API');
            const graphData = await graphDataManager.fetchInitialData();
            console.log(`[AppInitializer] Successfully fetched ${graphData.nodes.length} nodes, ${graphData.edges.length} edges`);
            if (debugState.isDataDebugEnabled()) {
              logger.debug('Initial graph data fetched successfully');
            }
          } catch (fetchError) {
            console.error('[AppInitializer] Failed to fetch initial graph data:', fetchError);
            logger.error('Failed to fetch initial graph data:', createErrorMetadata(fetchError));
            
            const emptyGraph = {
              nodes: [],
              edges: []
            };
            console.log('[AppInitializer] Initializing with empty graph due to fetch failure');
            await graphDataManager.setGraphData(emptyGraph);
          }

          console.log('[AppInitializer] About to call onInitialized');
          if (debugState.isEnabled()) {
            logger.info('Application initialized successfully');
          }

          
          onInitialized();
          console.log('[AppInitializer] onInitialized called successfully');

      } catch (error) {
          logger.error('Failed to initialize application components:', createErrorMetadata(error as Error));
          onError(error as Error);
      }
    };

    initApp();
  }, []);

  
  const initializeWebSocket = async (settings: any): Promise<void> => {
    try {
      const websocketService = WebSocketService.getInstance();

      
      websocketService.onBinaryMessage((data) => {
        if (data instanceof ArrayBuffer) {
          try {
            
            if (debugState.isDataDebugEnabled()) {
              logger.debug(`Received binary data from WebSocket: ${data.byteLength} bytes`);
            }

            
            graphDataManager.updateNodePositions(data).then(() => {
              if (debugState.isDataDebugEnabled()) {
                logger.debug(`Processed binary position update: ${data.byteLength} bytes`);
              }
            }).catch(error => {
              logger.error('Failed to process binary position update via worker:', createErrorMetadata(error));
            });
          } catch (error) {
            logger.error('Failed to process binary position update:', createErrorMetadata(error));

            
            if (debugState.isEnabled()) {
              
              logger.debug(`Binary data size: ${data.byteLength} bytes`);

              
              try {
                const view = new DataView(data);
                const hexBytes = [];
                const maxBytesToShow = Math.min(16, data.byteLength);

                for (let i = 0; i < maxBytesToShow; i++) {
                  hexBytes.push(view.getUint8(i).toString(16).padStart(2, '0'));
                }

                logger.debug(`First ${maxBytesToShow} bytes: ${hexBytes.join(' ')}`);

                
                if (data.byteLength >= 2) {
                  const firstByte = view.getUint8(0);
                  const secondByte = view.getUint8(1);
                  if (firstByte === 0x78 && (secondByte === 0x01 || secondByte === 0x9C || secondByte === 0xDA)) {
                    logger.debug('Data appears to be zlib compressed (has zlib header)');
                  }
                }
              } catch (e) {
                logger.debug('Could not display binary data preview');
              }

              
              const nodeSize = 26; 
              if (data.byteLength % nodeSize !== 0) {
                logger.debug(`Invalid data length: not a multiple of ${nodeSize} bytes per node (remainder: ${data.byteLength % nodeSize})`);
              }
            }
          }
        }
      });

      
      websocketService.onConnectionStatusChange((connected) => {
        if (debugState.isEnabled()) {
          logger.info(`WebSocket connection status changed: ${connected}`);
        }

        
        if (connected) {
          try {
            if (websocketService.isReady()) {
              
              logger.info('WebSocket is connected and fully established - enabling binary updates');
              graphDataManager.setBinaryUpdatesEnabled(true);

              
              logger.info('Sending subscribe_position_updates message to server');
              websocketService.sendMessage('subscribe_position_updates', {
                binary: true,
                interval: settings?.system?.websocket?.updateRate || 60
              });

              if (debugState.isDataDebugEnabled()) {
                logger.debug('Binary updates enabled and subscribed to position updates');
              }
            } else {
              logger.info('WebSocket connected but not fully established yet - waiting for readiness');

              
              
              graphDataManager.enableBinaryUpdates();

              
              const unsubscribe = websocketService.onMessage((message) => {
                if (message.type === 'connection_established') {
                  
                  logger.info('Connection established message received, sending subscribe_position_updates');
                  websocketService.sendMessage('subscribe_position_updates', {
                    binary: true,
                    interval: settings?.system?.websocket?.updateRate || 60
                  });
                  unsubscribe(); 

                  if (debugState.isDataDebugEnabled()) {
                    logger.debug('Connection established, subscribed to position updates');
                  }
                }
              });
            }
          } catch (connectionError) {
            logger.error('Error during WebSocket status change handling:', createErrorMetadata(connectionError));
          }
        }
      });

      
      if (websocketService) {
        const wsAdapter = {
          send: (data: ArrayBuffer) => {
            websocketService.sendRawBinaryData(data);
          },
          isReady: () => websocketService.isReady()
        };
        graphDataManager.setWebSocketService(wsAdapter);
      }

      try {
        
        await websocketService.connect();
      } catch (connectError) {
        logger.error('Failed to connect to WebSocket:', createErrorMetadata(connectError));
      }
    } catch (error) {
      logger.error('Failed during WebSocket/data initialization:', createErrorMetadata(error));
      throw error;
    }
  };

  return null; 
};

export default AppInitializer;
