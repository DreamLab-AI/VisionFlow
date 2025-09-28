import React, { useEffect, useRef, useCallback, useState } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { AuthGatedVoiceButton } from '../components/AuthGatedVoiceButton';
import { AuthGatedVoiceIndicator } from '../components/AuthGatedVoiceIndicator';
import { ARGraphViewport } from '../features/graph/components/ARGraphViewport';
import { createLogger, createErrorMetadata } from '../utils/loggerConfig';
import { useXRCore } from '../features/xr/providers/XRCoreProvider';
import { useApplicationMode } from '../contexts/ApplicationModeContext';
import { useSettingsStore } from '../store/settingsStore';
import { webSocketService } from '../services/WebSocketService';
import { graphDataManager } from '../features/graph/managers/graphDataManager';
import { parseBinaryNodeData, BinaryNodeData } from '../types/binaryProtocol';
import { BotsDataProvider } from '../features/bots/contexts/BotsDataContext';
import { Quest3FullscreenHandler } from '../features/xr/components/Quest3FullscreenHandler';
import * as THREE from 'three';

const logger = createLogger('Quest3AR');

/**
 * Unified Quest 3 AR Graph Renderer
 * Integrates with centralized data management while providing optimized rendering
 */
const ARGraphRenderer: React.FC = () => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const { camera } = useThree();
  const [nodeData, setNodeData] = useState<BinaryNodeData[]>([]);
  const settings = useSettingsStore((state) => state.settings);
  
  // Subscribe to graph data updates via the centralized graphDataManager
  useEffect(() => {
    const unsubscribe = graphDataManager.subscribeToUpdates((data) => {
      // Convert graph data manager format to our binary node format
      const nodes = graphDataManager.getVisibleNodes();
      const nodeArray: BinaryNodeData[] = nodes.map(node => ({
        nodeId: parseInt(node.id, 10) || 0,
        position: node.position,
        velocity: (node as any).velocity || { x: 0, y: 0, z: 0 },
        ssspDistance: Infinity, // Default SSSP distance
        ssspParent: -1 // Default SSSP parent
      }));
      setNodeData(nodeArray);
    });

    return unsubscribe;
  }, []);

  // Also listen directly to WebSocket binary updates for real-time performance
  useEffect(() => {
    const unsubscribe = webSocketService.onBinaryMessage((data: ArrayBuffer) => {
      try {
        // Use centralized binary parsing
        const parsedNodes = parseBinaryNodeData(data);
        setNodeData(parsedNodes);
      } catch (error) {
        logger.error('Error parsing binary node data in AR renderer:', createErrorMetadata(error));
      }
    });

    return unsubscribe;
  }, []);
  
  useFrame(() => {
    if (!meshRef.current || nodeData.length === 0) return;
    
    const mesh = meshRef.current;
    const nodeCount = nodeData.length;
    
    // Update instance matrices from centralized data
    for (let i = 0; i < Math.min(nodeCount, mesh.count); i++) {
      const node = nodeData[i];
      const { x, y, z } = node.position;
      
      // Level-of-detail based on distance to camera
      const distance = Math.sqrt(
        Math.pow(x - camera.position.x, 2) +
        Math.pow(y - camera.position.y, 2) +
        Math.pow(z - camera.position.z, 2)
      );
      
      // Apply AR-optimized LOD settings
      const maxDistance = 100; // Default AR render distance
      const scale = distance > maxDistance ? 0.3 : distance > 50 ? 0.6 : 1.0;
      
      mesh.setMatrixAt(i, new THREE.Matrix4().makeTranslation(x, y, z).scale(new THREE.Vector3(scale, scale, scale)));
      
      // Color based on node type or ID (could be enhanced with graph metadata)
      const hue = (node.nodeId % 12) / 12;
      const color = new THREE.Color().setHSL(hue, 0.7, 0.5);
      mesh.setColorAt(i, color);
    }
    
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  });
  
  const nodeCount = Math.max(nodeData.length, 1000); // Pre-allocate for performance
  
  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, nodeCount]}>
      <sphereGeometry args={[1, 8, 6]} />
      <meshPhysicalMaterial 
        color="white" 
        metalness={0.7} 
        roughness={0.3}
        transparent
        opacity={0.8}
      />
    </instancedMesh>
  );
};

/**
 * Unified Quest 3 AR Layout
 * Combines the best of both previous implementations:
 * - Uses XRCoreProvider integration from Quest3ARLayout
 * - Integrates with centralized WebSocketService and graphDataManager
 * - Provides optimized AR rendering with Three.js
 * - Maintains voice interaction capabilities
 */
const Quest3AR: React.FC = () => {
  const { isSessionActive, sessionType, startSession, isXRSupported } = useXRCore();
  const { setMode } = useApplicationMode();
  const settings = useSettingsStore((state) => state.settings);
  const [isConnected, setIsConnected] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Performance optimization refs
  const frameRef = useRef<number>();
  const lastUpdateRef = useRef<number>(0);
  const [updateRate, setUpdateRate] = useState(72); // Quest 3 native refresh rate
  
  // Quest 3 detection
  const isQuest3 = useRef(false);
  
  useEffect(() => {
    const userAgent = navigator.userAgent.toLowerCase();
    isQuest3.current = userAgent.includes('quest 3') || 
                      userAgent.includes('meta quest 3') ||
                      (userAgent.includes('oculus') && userAgent.includes('quest'));
    
    if (isQuest3.current) {
      logger.info('Meta Quest 3 detected - applying AR optimizations');
      setUpdateRate(72); // Quest 3 native refresh rate
    }
  }, []);

  // Auto-start AR session if conditions are met
  useEffect(() => {
    const initializeAR = async () => {
      try {
        if (!isQuest3.current && !window.location.search.includes('force=quest3')) {
          logger.warn('Not Quest 3 device, add ?force=quest3 to test');
          setIsReady(true); // Still allow fallback mode
          return;
        }

        logger.info('Quest 3 detected - initializing unified AR mode', {
          isXRSupported,
          hasStartSession: !!startSession
        });
        
        // Check if XR is supported before trying to start session
        if (!isXRSupported) {
          logger.warn('WebXR not supported on this device/browser');
          // Don't set error, just continue without XR
        }
        
        // Don't auto-start XR session - let user trigger it manually
        logger.info('XR support available, user can start session manually');
        
        // Ensure WebSocket connection through centralized service
        if (!webSocketService.isReady()) {
          await webSocketService.connect();
        }
        
        setIsReady(true);
        logger.info('Quest 3 AR initialized successfully');
        
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Unknown error';
        logger.error('Failed to initialize Quest 3 AR:', err);
        setError(`Initialization failed: ${errorMessage}`);
        setIsReady(true); // Fallback to standard mode
      }
    };

    initializeAR();
  }, [isSessionActive, startSession]);

  // Set XR mode when AR session is active
  useEffect(() => {
    if (isSessionActive) {
      setMode('xr');
      logger.info('Quest 3 AR Layout activated - entering XR mode');
    }
  }, [isSessionActive, setMode]);
  
  // Monitor WebSocket connection through centralized service
  useEffect(() => {
    const unsubscribe = webSocketService.onConnectionStatusChange((connected) => {
      setIsConnected(connected);
    });
    
    return unsubscribe;
  }, []);

  // Optimized render loop
  const renderLoop = useCallback(() => {
    const now = performance.now();
    const deltaTime = now - lastUpdateRef.current;
    
    if (deltaTime >= 1000 / updateRate) {
      lastUpdateRef.current = now;
      // Update logic handled by React Three Fiber and graphDataManager
    }
    
    frameRef.current = requestAnimationFrame(renderLoop);
  }, [updateRate]);
  
  useEffect(() => {
    if (isSessionActive) {
      frameRef.current = requestAnimationFrame(renderLoop);
    }
    
    return () => {
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current);
      }
    };
  }, [isSessionActive, renderLoop]);

  // Skip loading screen - go straight to AR viewport
  // if (!isReady) {
  //   return (
  //     <div style={{
  //       position: 'fixed',
  //       top: 0,
  //       left: 0,
  //       right: 0,
  //       bottom: 0,
  //       background: 'linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%)',
  //       color: 'white',
  //       display: 'flex',
  //       flexDirection: 'column',
  //       alignItems: 'center',
  //       justifyContent: 'center',
  //       fontSize: '24px',
  //       fontFamily: 'system-ui'
  //     }}>
  //       <div className="mb-4">🥽</div>
  //       <div>Initializing Quest 3 AR...</div>
  //       <div style={{ fontSize: '14px', opacity: 0.7, marginTop: '16px' }}>
  //         Preparing immersive AR experience
  //       </div>
  //     </div>
  //   );
  // }

  // If in XR session, use minimal AR view with Three.js Canvas
  if (isSessionActive) {
    return (
      <div style={{
        width: '100vw',
        height: '100vh',
        position: 'relative',
        overflow: 'hidden',
        backgroundColor: 'transparent', // For AR passthrough
        margin: 0,
        padding: 0
      }}>
        {/* Three.js AR Scene - Clean AR Passthrough using settings.yaml */}
        <Canvas
          camera={{ position: [0, 0, 5], fov: 75 }}
          gl={{
            antialias: false, // Performance optimization for Quest 3
            alpha: true, // Required for AR passthrough
            preserveDrawingBuffer: true,
            xr: { // WebXR configuration
              enabled: true,
              mode: 'immersive-ar' // AR passthrough mode
            }
          }}
          frameloop="always"
          dpr={0.8} // Lower DPR for Quest 3 performance
          style={{
            width: '100%',
            height: '100%',
            position: 'absolute',
            top: 0,
            left: 0
          }}
        >
          <ambientLight intensity={settings?.visualisation?.rendering?.ambientLightIntensity ?? 0.5} />
          <directionalLight position={[10, 10, 5]} intensity={settings?.visualisation?.rendering?.directionalLightIntensity ?? 0.01263736} />
          <ARGraphRenderer />
        </Canvas>
      </div>
    );
  }


  // Add error handling
  if (error) {
    return (
      <div style={{
        width: '100vw',
        height: '100vh',
        backgroundColor: 'red',
        color: 'white',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '24px',
        textAlign: 'center',
        padding: '20px'
      }}>
        <div>
          <h1>Error in Quest3AR</h1>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  // Debug info to check if component is mounting
  return (
    <div style={{
      width: '100vw',
      height: '100vh',
      backgroundColor: 'black',
      color: 'white',
      position: 'relative'
    }}>
      {/* Debug info */}
      <div style={{
        position: 'absolute',
        top: '10px',
        left: '10px',
        backgroundColor: 'rgba(255, 0, 0, 0.8)',
        padding: '10px',
        zIndex: 1000,
        fontSize: '16px'
      }}>
        <div>Quest3AR Component Loaded</div>
        <div>Protocol: {window.location.protocol}</div>
        <div>WebXR in navigator: {'xr' in navigator ? 'YES' : 'NO'}</div>
        <div>WebXR Supported: {isXRSupported ? 'YES' : 'NO'}</div>
        <div>Session Active: {isSessionActive ? 'YES' : 'NO'}</div>
        <div>Start Session Available: {startSession ? 'YES' : 'NO'}</div>
        <div>Is Ready: {isReady ? 'YES' : 'NO'}</div>
        <div>Is Quest 3: {isQuest3.current ? 'YES' : 'NO'}</div>
        <div>Settings: {settings ? 'Loaded' : 'Not loaded'}</div>
        {/* Manual XR button */}
        {'xr' in navigator && (
          <button
            onClick={async () => {
              try {
                const session = await navigator.xr.requestSession('immersive-ar');
                console.log('XR Session started:', session);
                setError('XR Session started successfully!');
              } catch (e) {
                console.error('XR Session failed:', e);
                setError(`XR Failed: ${e.message}`);
              }
            }}
            style={{
              marginTop: '10px',
              padding: '5px 10px',
              backgroundColor: 'blue',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer'
            }}
          >
            Test WebXR
          </button>
        )}
      </div>
      
      {/* Render the AR viewport */}
      <ARGraphViewport />
    </div>
  );
};

// Wrap Quest3AR with BotsDataProvider to ensure all child components have access to bots data
const Quest3ARWithProviders: React.FC = () => {
  return (
    <BotsDataProvider>
      <Quest3AR />
    </BotsDataProvider>
  );
};

export default Quest3ARWithProviders;