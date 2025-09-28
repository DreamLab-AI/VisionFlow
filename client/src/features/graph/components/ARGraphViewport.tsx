import React, { useRef, useCallback, useEffect, useState, Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import GraphManager from './GraphManager';
import { BotsVisualization } from '../../bots/components/BotsVisualizationFixed';
import CameraController from '../../visualisation/components/CameraController';
import { useSettingsStore } from '../../../store/settingsStore';
import { createLogger } from '../../../utils/loggerConfig';
import * as THREE from 'three';

const logger = createLogger('ARGraphViewport'); // Clean AR viewport

/**
 * Clean AR-optimized graph viewport without any UI or hologram elements
 * Loads all settings from settings.yaml
 */
export const ARGraphViewport: React.FC = () => {
  const settings = useSettingsStore(state => state.settings);
  const renderingSettings = settings?.visualisation?.rendering;
  
  const [graphCenter, setGraphCenter] = useState<[number, number, number]>([0, 0, 0]);
  const [graphSize, setGraphSize] = useState(50);
  const [isNodeDragging, setIsNodeDragging] = useState(false);
  
  useEffect(() => {
    logger.info('ARGraphViewport mounted', { settings: renderingSettings });
  }, []);
  
  // Light refs for layer control
  const [ambientRef, setAmbientRef] = useState<any>(null);
  const [dirRef, setDirRef] = useState<any>(null);
  const [pointRef, setPointRef] = useState<any>(null);

  // Camera settings for AR
  const near = 0.1;
  const far = 1500;
  const backgroundColor = renderingSettings?.backgroundColor || '#000033'; // Dark blue background for visibility

  useEffect(() => {
    // Enable lights only for default layer and layer 1
    [ambientRef, dirRef, pointRef].forEach((l) => {
      if (l) {
        l.layers.enable(0); // Default layer
        l.layers.enable(1); // Bloom layer 1
      }
    });
  }, [ambientRef, dirRef, pointRef]);

  return (
    <div style={{ 
      width: '100vw', 
      height: '100vh', 
      position: 'fixed',
      top: 0,
      left: 0,
      backgroundColor: '#000033'
    }}>
      <Canvas
        camera={{ near, far }}
        gl={{
          antialias: renderingSettings?.enableAntialiasing ?? false,
          alpha: true, // Always transparent for AR
          powerPreference: 'high-performance',
          logarithmicDepthBuffer: false,
          toneMapping: THREE.ACESFilmicToneMapping,
          preserveDrawingBuffer: true,
          outputColorSpace: THREE.SRGBColorSpace
        }}
        dpr={[1, 2]}
        shadows={renderingSettings?.enableShadows ?? false}
      >
        <color attach="background" args={[backgroundColor]} />
        <CameraController center={graphCenter} size={graphSize} />
        
        {/* Lighting from settings.yaml */}
        <ambientLight 
          ref={setAmbientRef} 
          intensity={renderingSettings?.ambientLightIntensity ?? 0.15} 
        />
        <directionalLight
          ref={setDirRef}
          position={[
            renderingSettings?.directionalLightPosition?.[0] ?? 10,
            renderingSettings?.directionalLightPosition?.[1] ?? 10,
            renderingSettings?.directionalLightPosition?.[2] ?? 5
          ]}
          intensity={renderingSettings?.directionalLightIntensity ?? 0.4}
          castShadow={renderingSettings?.enableShadows ?? false}
        />
        <pointLight
          ref={setPointRef}
          position={[
            renderingSettings?.pointLightPosition?.[0] ?? -10,
            renderingSettings?.pointLightPosition?.[1] ?? -10,
            renderingSettings?.pointLightPosition?.[2] ?? -5
          ]}
          intensity={renderingSettings?.pointLightIntensity ?? 0.5}
        />

        <OrbitControls
          makeDefault
          enableDamping
          dampingFactor={0.05}
          minDistance={1}
          maxDistance={far / 2}
          target={graphCenter}
          enabled={!isNodeDragging}
          enableRotate={!isNodeDragging}
          enablePan={!isNodeDragging}
          enableZoom={!isNodeDragging}
          mouseButtons={isNodeDragging ? {} : undefined}
        />

        <Suspense fallback={null}>
          {/* Test cube to verify rendering */}
          <mesh position={[0, 0, 0]}>
            <boxGeometry args={[2, 2, 2]} />
            <meshStandardMaterial color="red" />
          </mesh>
          
          {/* Graph rendering only - no hologram */}
          <GraphManager onDragStateChange={(isDragging) => {
            setIsNodeDragging(isDragging);
          }} />

          {/* VisionFlow visualization if enabled */}
          <BotsVisualization />
        </Suspense>
      </Canvas>
    </div>
  );
};