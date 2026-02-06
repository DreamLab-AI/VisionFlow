import React, { useEffect } from 'react';
import { createLogger, createErrorMetadata } from '../utils/loggerConfig';
import { debugState } from '../utils/clientDebugState';
import { useSettingsStore } from '../store/settingsStore';
import { useWorkerErrorStore } from '../store/workerErrorStore';
import { webSocketService } from '../store/websocketStore';
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
      logger.info('Starting Innovation Manager initialization...');
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

      logger.info('Innovation Manager initialized successfully');
      if (debugState.isEnabled()) {
        logger.info('Innovation Manager initialized successfully');
        const status = innovationManager.getStatus();
        logger.debug('Innovation Manager status:', status);
      }
    } catch (innovationError) {
      logger.error('Innovation Manager initialization failed:', createErrorMetadata(innovationError));
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

        // Set up retry handler for worker initialization
        const initializeWorker = async (): Promise<boolean> => {
          try {
            logger.info('Step 1: Initializing graphWorkerProxy');
            await graphWorkerProxy.initialize();
            logger.info('Step 1b: graphWorkerProxy initialized, ensuring graphDataManager worker connection');

            // Ensure graphDataManager is connected to the worker now that it's ready
            const workerReady = await graphDataManager.ensureWorkerReady();
            logger.info(`Step 1c: graphDataManager worker ready: ${workerReady}`);

            if (!workerReady) {
              throw new Error('Graph worker failed to become ready after initialization');
            }

            return true;
          } catch (workerError) {
            logger.error('Worker initialization failed:', createErrorMetadata(workerError));
            const errorMessage = workerError instanceof Error ? workerError.message : String(workerError);

            // Check for SharedArrayBuffer-related issues
            let details = errorMessage;
            if (typeof SharedArrayBuffer === 'undefined') {
              details = 'SharedArrayBuffer is not available. This is required for the graph engine to function properly.';
            } else if (errorMessage.includes('Worker') || errorMessage.includes('worker')) {
              details = `Worker initialization error: ${errorMessage}`;
            }

            useWorkerErrorStore.getState().setWorkerError(
              'The graph visualization engine failed to initialize.',
              details
            );

            // Continue without worker - graceful degradation
            logger.warn('Continuing without fully initialized worker');
            return false;
          }
        };

        // Store retry handler
        useWorkerErrorStore.getState().setRetryHandler(async () => {
          const success = await initializeWorker();
          if (!success) {
            throw new Error('Worker initialization retry failed');
          }
        });

        try {
          await initializeWorker();

          logger.info('Step 2: graphWorkerProxy initialized, calling settings initialize');
          await initialize();
          logger.info('Step 3: Settings initialized');

          // Access settings from the store after initialization
          const currentSettings = useSettingsStore.getState().settings as any;
          if (currentSettings?.system?.debug) {
            try {
              const debugSettings = currentSettings.system.debug;
              debugState.enableDebug(debugSettings.enabled);
              if (debugSettings.enabled) {
                debugState.enableDataDebug(debugSettings.enableDataDebug);
                debugState.enablePerformanceDebug(debugSettings.enablePerformanceDebug);
              }
            } catch (debugError) {
              logger.warn('Error applying debug settings:', createErrorMetadata(debugError));
            }
          }

          
          if (typeof graphDataManager !== 'undefined') {
            try {
              
              await initializeWebSocket(settings);
              
            } catch (wsError) {
              logger.error('WebSocket initialization failed, continuing with UI only:', createErrorMetadata(wsError));
              
            }
          } else {
            logger.warn('WebSocket services not available, continuing with UI only');
          }

          
          try {
            logger.info('About to fetch initial graph data via REST API');
            logger.info('Fetching initial graph data via REST API');
            const graphData = await graphDataManager.fetchInitialData();
            logger.info(`Successfully fetched ${graphData.nodes.length} nodes, ${graphData.edges.length} edges`);
            if (debugState.isDataDebugEnabled()) {
              logger.debug('Initial graph data fetched successfully');
            }
          } catch (fetchError) {
            logger.error('Failed to fetch initial graph data:', createErrorMetadata(fetchError));
            logger.error('Failed to fetch initial graph data:', createErrorMetadata(fetchError));
            
            const emptyGraph = {
              nodes: [],
              edges: []
            };
            logger.info('Initializing with empty graph due to fetch failure');
            await graphDataManager.setGraphData(emptyGraph);
          }

          logger.info('About to call onInitialized');
          if (debugState.isEnabled()) {
            logger.info('Application initialized successfully');
          }

          
          onInitialized();
          logger.info('onInitialized called successfully');

      } catch (error) {
          logger.error('Failed to initialize application components:', createErrorMetadata(error as Error));
          onError(error as Error);
      }
    };

    initApp();
  }, []);

  
  const initializeWebSocket = async (settings: any): Promise<void> => {
    try {
      const websocketService = webSocketService;

      
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
                if ((message as any).type === 'connection_established') {
                  
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
