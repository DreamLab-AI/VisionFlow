/**
 * Quest 3 Direct AR Entry
 * Minimal, immediate AR experience with no UI chrome
 * Auto-loads server settings and enters AR mode on detection
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { webSocketService } from '../services/WebSocketService';
import { createLogger } from '../utils/logger';

const logger = createLogger('Quest3DirectAR');

// Minimal graph renderer - just positions from server
const ARGraphRenderer: React.FC<{ nodes: Float32Array; colors: Float32Array }> = ({ nodes, colors }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const { camera } = useThree();
  
  useFrame(() => {
    if (!meshRef.current || nodes.length === 0) return;
    
    const mesh = meshRef.current;
    const nodeCount = nodes.length / 4; // x,y,z,t per node
    
    // Update instance matrices from server positions
    for (let i = 0; i < nodeCount; i++) {
      const x = nodes[i * 4];
      const y = nodes[i * 4 + 1];
      const z = nodes[i * 4 + 2];
      
      // Simple LOD based on distance
      const distance = Math.sqrt(
        Math.pow(x - camera.position.x, 2) +
        Math.pow(y - camera.position.y, 2) +
        Math.pow(z - camera.position.z, 2)
      );
      
      const scale = distance > 100 ? 0.5 : 1.0;
      
      mesh.setMatrixAt(i, new THREE.Matrix4().makeTranslation(x, y, z).scale(new THREE.Vector3(scale, scale, scale)));
      
      // Set color from server
      if (colors.length > i * 4) {
        mesh.setColorAt(i, new THREE.Color(colors[i * 4], colors[i * 4 + 1], colors[i * 4 + 2]));
      }
    }
    
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  });
  
  const nodeCount = nodes.length / 4;
  
  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, nodeCount]}>
      <sphereGeometry args={[1, 8, 6]} />
      <meshPhysicalMaterial color="white" metalness={0.7} roughness={0.3} />
    </instancedMesh>
  );
};

// Main Quest 3 Direct AR component
export const Quest3DirectAR: React.FC = () => {
  const [xrSession, setXrSession] = useState<XRSession | null>(null);
  const [nodes, setNodes] = useState<Float32Array>(new Float32Array());
  const [colors, setColors] = useState<Float32Array>(new Float32Array());
  const [isReady, setIsReady] = useState(false);
  const animationFrameRef = useRef<number>();
  
  // Auto-detect Quest 3 and enter AR
  useEffect(() => {
    const initQuest3AR = async () => {
      // Check if Quest 3
      const userAgent = navigator.userAgent.toLowerCase();
      const isQuest3 = userAgent.includes('quest 3') || 
                       userAgent.includes('meta quest 3') ||
                       (userAgent.includes('oculus') && userAgent.includes('quest'));
      
      if (!isQuest3 && !window.location.search.includes('force=quest3')) {
        logger.warn('Not Quest 3 device, add ?force=quest3 to test');
        return;
      }
      
      logger.info('Quest 3 detected - initializing direct AR mode');
      
      try {
        // 1. Load Quest 3 optimized settings from server
        const settingsResponse = await fetch('/api/quest3/defaults');
        const settings = await settingsResponse.json();
        logger.info('Loaded Quest 3 settings:', settings);
        
        // 2. Apply settings globally
        if (settings.success && settings.settings) {
          // Apply performance settings
          window.devicePixelRatio = settings.settings.performance.renderScale;
          
          // Store for later use
          (window as any).__quest3Settings = settings.settings;
        }
        
        // 3. Check WebXR support
        if (!navigator.xr) {
          throw new Error('WebXR not supported');
        }
        
        const isSupported = await navigator.xr.isSessionSupported('immersive-ar');
        if (!isSupported) {
          throw new Error('AR not supported on this device');
        }
        
        // 4. Request AR session with Quest 3 optimized features
        const session = await navigator.xr.requestSession('immersive-ar', {
          requiredFeatures: ['local-floor', 'hand-tracking'],
          optionalFeatures: ['bounded-floor', 'depth-sensing', 'light-estimation']
        });
        
        logger.info('AR session started successfully');
        setXrSession(session);
        
        // 5. Connect WebSocket for position streaming
        webSocketService.connect();
        
        // 6. Set up binary data handler
        webSocketService.on('binary-data', (data: ArrayBuffer) => {
          // Parse binary positions (28 bytes per node: id(4) + position(12) + velocity(12))
          const view = new DataView(data);
          const nodeCount = data.byteLength / 28;
          
          const newNodes = new Float32Array(nodeCount * 4); // x,y,z,t
          const newColors = new Float32Array(nodeCount * 4); // r,g,b,a
          
          for (let i = 0; i < nodeCount; i++) {
            const offset = i * 28;
            
            // Skip ID (4 bytes)
            // Read position (12 bytes)
            newNodes[i * 4] = view.getFloat32(offset + 4, true);     // x
            newNodes[i * 4 + 1] = view.getFloat32(offset + 8, true);  // y  
            newNodes[i * 4 + 2] = view.getFloat32(offset + 12, true); // z
            newNodes[i * 4 + 3] = 0; // t (time, not used yet)
            
            // Generate color based on node ID for now
            const nodeId = view.getUint32(offset, true);
            const hue = (nodeId % 12) / 12;
            
            // HSL to RGB conversion
            const h = hue * 360;
            const s = 0.7;
            const l = 0.5;
            const c = (1 - Math.abs(2 * l - 1)) * s;
            const x = c * (1 - Math.abs((h / 60) % 2 - 1));
            const m = l - c / 2;
            
            let r = 0, g = 0, b = 0;
            if (h < 60) { r = c; g = x; b = 0; }
            else if (h < 120) { r = x; g = c; b = 0; }
            else if (h < 180) { r = 0; g = c; b = x; }
            else if (h < 240) { r = 0; g = x; b = c; }
            else if (h < 300) { r = x; g = 0; b = c; }
            else { r = c; g = 0; b = x; }
            
            newColors[i * 4] = r + m;
            newColors[i * 4 + 1] = g + m;
            newColors[i * 4 + 2] = b + m;
            newColors[i * 4 + 3] = 1.0;
          }
          
          setNodes(newNodes);
          setColors(newColors);
        });
        
        // 7. Request initial data
        webSocketService.send({ type: 'request-positions' });
        
        setIsReady(true);
        
        // 8. Set up AR render loop
        const onXRFrame: XRFrameRequestCallback = (time, frame) => {
          // AR frame processing if needed
          animationFrameRef.current = session.requestAnimationFrame(onXRFrame);
        };
        
        animationFrameRef.current = session.requestAnimationFrame(onXRFrame);
        
        // Handle session end
        session.addEventListener('end', () => {
          logger.info('AR session ended');
          setXrSession(null);
          setIsReady(false);
          if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current);
          }
        });
        
      } catch (error) {
        logger.error('Failed to initialize Quest 3 AR:', error);
        // Fallback to standard view
        setIsReady(true);
      }
    };
    
    // Start immediately
    initQuest3AR();
    
    // Cleanup
    return () => {
      if (xrSession) {
        xrSession.end();
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      webSocketService.disconnect();
    };
  }, []);
  
  // Minimal UI - just the AR scene
  if (!isReady) {
    return (
      <div style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: 'black',
        color: 'white',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '24px',
        fontFamily: 'system-ui'
      }}>
        Initializing AR...
      </div>
    );
  }
  
  return (
    <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0 }}>
      <Canvas
        camera={{ position: [0, 0, 5], fov: 75 }}
        gl={{
          antialias: false, // Performance optimization for Quest 3
          alpha: true,
          preserveDrawingBuffer: true
        }}
        frameloop="demand" // Only render when needed
        dpr={0.8} // Lower DPR for Quest 3 performance
      >
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 5]} intensity={1} />
        
        {nodes.length > 0 && (
          <ARGraphRenderer nodes={nodes} colors={colors} />
        )}
      </Canvas>
    </div>
  );
};

export default Quest3DirectAR;