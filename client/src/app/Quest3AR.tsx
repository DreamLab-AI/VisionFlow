import React, { useEffect, useRef, useCallback, useState } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { AuthGatedVoiceButton } from '../components/AuthGatedVoiceButton';
import { AuthGatedVoiceIndicator } from '../components/AuthGatedVoiceIndicator';
import GraphViewport from '../features/graph/components/GraphViewport';
import { createLogger, createErrorMetadata } from '../utils/loggerConfig';
import { useXRCore } from '../features/xr/providers/XRCoreProvider';
import { useApplicationMode } from '../contexts/ApplicationModeContext';
import { useSettingsStore } from '../store/settingsStore';
import { webSocketService } from '../services/WebSocketService';
import { graphDataManager } from '../features/graph/managers/graphDataManager';
import { parseBinaryNodeData, BinaryNodeData } from '../types/binaryProtocol';
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
  const { isSessionActive, sessionType, startSession } = useXRCore();
  const { setMode } = useApplicationMode();
  const settings = useSettingsStore((state) => state.settings);
  const [isConnected, setIsConnected] = useState(false);
  const [isReady, setIsReady] = useState(false);
  
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
      if (!isQuest3.current && !window.location.search.includes('force=quest3')) {
        logger.warn('Not Quest 3 device, add ?force=quest3 to test');
        setIsReady(true); // Still allow fallback mode
        return;
      }

      try {
        logger.info('Quest 3 detected - initializing unified AR mode');
        
        // Use the centralized XR session management
        if (!isSessionActive) {
          await startSession();
        }
        
        // Ensure WebSocket connection through centralized service
        if (!webSocketService.isReady()) {
          await webSocketService.connect();
        }
        
        setIsReady(true);
        logger.info('Quest 3 AR initialized successfully');
        
      } catch (error) {
        logger.error('Failed to initialize Quest 3 AR:', error);
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

  if (!isReady) {
    return (
      <div style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: 'linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%)',
        color: 'white',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '24px',
        fontFamily: 'system-ui'
      }}>
        <div className="mb-4">🥽</div>
        <div>Initializing Quest 3 AR...</div>
        <div style={{ fontSize: '14px', opacity: 0.7, marginTop: '16px' }}>
          Preparing immersive AR experience
        </div>
      </div>
    );
  }

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
        {/* Three.js AR Scene */}
        <Canvas
          camera={{ position: [0, 0, 5], fov: 75 }}
          gl={{
            antialias: false, // Performance optimization for Quest 3
            alpha: true,
            preserveDrawingBuffer: true
          }}
          frameloop="demand"
          dpr={0.8} // Lower DPR for Quest 3 performance
          style={{
            width: '100%',
            height: '100%',
            position: 'absolute',
            top: 0,
            left: 0,
            zIndex: 1
          }}
        >
          <ambientLight intensity={0.5} />
          <directionalLight position={[10, 10, 5]} intensity={1} />
          <ARGraphRenderer />
        </Canvas>

        {/* AR-optimized voice controls */}
        <div style={{
          position: 'fixed',
          bottom: '40px',
          left: '50%',
          transform: 'translateX(-50%)',
          zIndex: 1000,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '16px',
          pointerEvents: 'auto'
        }}>
          <AuthGatedVoiceButton
            size="lg"
            variant="primary"
            className="bg-blue-500 bg-opacity-90 backdrop-blur-md border-2 border-white border-opacity-30 shadow-lg"
          />
          <AuthGatedVoiceIndicator
            className="max-w-sm text-center bg-black bg-opacity-70 backdrop-blur-md rounded-xl p-3 border border-white border-opacity-20 text-white"
            showTranscription={true}
            showStatus={true}
          />
        </div>

        {/* AR session status indicator */}
        <div style={{
          position: 'fixed',
          top: '20px',
          left: '20px',
          zIndex: 1000,
          backgroundColor: 'rgba(0, 255, 0, 0.8)',
          color: 'black',
          padding: '8px 12px',
          borderRadius: '20px',
          fontSize: '14px',
          fontWeight: 'bold',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.3)',
          pointerEvents: 'none'
        }}>
          {isQuest3.current ? 'Quest 3' : 'XR'} AR Active • {updateRate}fps
        </div>

        {/* Development debug info */}
        {process.env.NODE_ENV === 'development' && (
          <div style={{
            position: 'fixed',
            top: '60px',
            left: '20px',
            zIndex: 999,
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            color: 'white',
            padding: '12px',
            borderRadius: '8px',
            fontSize: '12px',
            fontFamily: 'monospace',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            pointerEvents: 'none',
            maxWidth: '300px'
          }}>
            <div>Session Type: {sessionType}</div>
            <div>Unified AR: Active</div>
            <div>Voice Controls: Available</div>
            <div>Device: {isQuest3.current ? 'Meta Quest 3' : 'Generic XR'}</div>
            <div>Update Rate: {updateRate} fps</div>
            <div>WebSocket: {isConnected ? 'Connected' : 'Disconnected'}</div>
            <div>Data Source: Centralized (graphDataManager)</div>
            <div>Binary Protocol: Integrated</div>
          </div>
        )}
      </div>
    );
  }

  // Fallback: Use GraphViewport for non-XR mode with AR readiness UI
  return (
    <div style={{
      width: '100vw',
      height: '100vh',
      position: 'relative',
      overflow: 'hidden',
      backgroundColor: 'transparent',
      margin: 0,
      padding: 0
    }}>
      {/* Full-screen graph viewport for fallback */}
      <div style={{
        width: '100vw',
        height: '100vh',
        position: 'absolute',
        top: 0,
        left: 0,
        zIndex: 1
      }}>
        <GraphViewport />
      </div>

      {/* AR readiness overlay */}
      <div style={{
        position: 'fixed',
        top: '20px',
        right: '20px',
        zIndex: 1000,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        color: 'white',
        padding: '16px',
        borderRadius: '12px',
        textAlign: 'center',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255, 255, 255, 0.2)',
      }}>
        <div style={{ fontSize: '16px', marginBottom: '8px' }}>🥽 AR Ready</div>
        {isQuest3.current ? (
          <button
            onClick={() => startSession().catch(logger.error)}
            style={{
              backgroundColor: '#3B82F6',
              color: 'white',
              border: 'none',
              padding: '8px 16px',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '14px'
            }}
          >
            Enter AR
          </button>
        ) : (
          <div style={{ fontSize: '12px', opacity: 0.7 }}>
            Quest 3 required
          </div>
        )}
      </div>

      {/* Voice controls for fallback mode */}
      <div style={{
        position: 'fixed',
        bottom: '40px',
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 1000,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '16px',
        pointerEvents: 'auto'
      }}>
        <AuthGatedVoiceButton
          size="lg"
          variant="primary"
          className="bg-blue-500 bg-opacity-90 backdrop-blur-md border-2 border-white border-opacity-30 shadow-lg"
        />
        <AuthGatedVoiceIndicator
          className="max-w-sm text-center bg-black bg-opacity-70 backdrop-blur-md rounded-xl p-3 border border-white border-opacity-20 text-white"
          showTranscription={true}
          showStatus={true}
        />
      </div>
    </div>
  );
};

export default Quest3AR;