import React, { Suspense, useEffect, useState, useMemo, useRef } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, Stats } from '@react-three/drei';
import { EffectComposer } from '@react-three/postprocessing';
import * as THREE from 'three';
import { AtmosphericGlow } from '../../visualisation/effects/AtmosphericGlow';
import { graphDataManager } from '../managers/graphDataManager';
import GraphManager from './GraphManager';
import CameraController from '../../visualisation/components/CameraController';
import { useSettingsStore } from '../../../store/settingsStore';
import { createLogger } from '../../../utils/logger';
import { debugState } from '../../../utils/clientDebugState';
import { BotsVisualization } from '../../bots/components';
import { EnhancedHologramSystem } from '../../visualisation/renderers/EnhancedHologramSystem';
import { WorldClassHologram, EnergyFieldParticles } from '../../visualisation/components/WorldClassHologram';

// Ensure Three.js types are properly loaded if not globally done
// import '../../../types/react-three-fiber.d.ts';

const logger = createLogger('GraphViewport');

const AtmosphericGlowWrapper = () => {
  const { camera, size } = useThree();
  const glowSettings = useSettingsStore(state => state.settings?.visualisation?.glow);
  
  // Use default values if glow settings are not available
  const defaultGlow = {
    baseColor: "#00ffff",
    intensity: 2.0,
    radius: 0.85,
    threshold: 0.15,
    diffuseStrength: 1.5,
    atmosphericDensity: 0.8,
    volumetricIntensity: 1.2
  };
  
  const settings = glowSettings || defaultGlow;

  return (
    <AtmosphericGlow
      glowColor={settings.baseColor}
      intensity={settings.intensity}
      radius={settings.radius}
      threshold={settings.threshold}
      diffuseStrength={settings.diffuseStrength}
      atmosphericDensity={settings.atmosphericDensity}
      volumetricIntensity={settings.volumetricIntensity}
      camera={camera}
      resolution={size}
    />
  );
};

const GraphViewport: React.FC = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [graphCenter, setGraphCenter] = useState<[number, number, number]>([0, 0, 0]);
  const [graphSize, setGraphSize] = useState(50); // Default size
  const [isNodeDragging, setIsNodeDragging] = useState(false); // <--- Add this state
  const [ambientRef, setAmbientRef] = useState<THREE.AmbientLight | null>(null);
  const [dirRef, setDirRef] = useState<THREE.DirectionalLight | null>(null);
  const [pointRef, setPointRef] = useState<THREE.PointLight | null>(null);

  // Settings for camera and visuals
  const settings = useSettingsStore(state => state.settings);
  const initialized = useSettingsStore(state => state.initialized);
  const [viewportRefresh, setViewportRefresh] = useState(0);

  // Subscribe to viewport updates for real-time changes
  useEffect(() => {
    const unsubscribe = useSettingsStore.getState().subscribe(
      'viewport.update',
      () => {
        logger.debug('Viewport update triggered');
        setViewportRefresh(prev => prev + 1); // Force re-render
      },
      false
    );

    return unsubscribe;
  }, []);

  const cameraSettings = settings?.visualisation?.camera;
  const renderingSettings = settings?.visualisation?.rendering;
  const glowSettings = settings?.visualisation?.glow;
  const debugSettings = settings?.system?.debug;
  const nodeSettings = settings?.visualisation?.graphs?.logseq?.nodes || settings?.visualisation?.nodes;
  const hologramEnabled = nodeSettings?.enableHologram || false;

  const fov = cameraSettings?.fov ?? 75;
  const near = cameraSettings?.near ?? 0.1;
  const far = cameraSettings?.far ?? 2000;

  // Memoize cameraPosition to ensure stable reference unless underlying values change
  const cameraPosition = useMemo(() => (
    cameraSettings?.position
      ? [cameraSettings.position.x, cameraSettings.position.y, cameraSettings.position.z]
      : [0, 10, 50] // Default camera position
  ), [cameraSettings?.position]);

  const enableGlow = glowSettings?.enabled ?? true;


  useEffect(() => {
    const initializeGraph = async () => {
      setIsLoading(true);
      setError(null);
      try {
        logger.debug('Fetching initial graph data...');
        await graphDataManager.fetchInitialData();
        logger.debug('Graph data fetched.');
        const data = await graphDataManager.getGraphData();

        if (!data || !data.nodes || data.nodes.length === 0) {
          logger.warn('No graph data or empty nodes received.');
          setGraphCenter([0,0,0]);
          setGraphSize(50); // Default size for empty graph
          setIsLoading(false);
          return;
        }

        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

        data.nodes.forEach((node) => {
          if (node.position) {
            minX = Math.min(minX, node.position.x);
            maxX = Math.max(maxX, node.position.x);
            minY = Math.min(minY, node.position.y);
            maxY = Math.max(maxY, node.position.y);
            minZ = Math.min(minZ, node.position.z);
            maxZ = Math.max(maxZ, node.position.z);
          }
        });

        const centerX = (minX === Infinity || maxX === -Infinity) ? 0 : (maxX + minX) / 2;
        const centerY = (minY === Infinity || maxY === -Infinity) ? 0 : (maxY + minY) / 2;
        const centerZ = (minZ === Infinity || maxZ === -Infinity) ? 0 : (maxZ + minZ) / 2;

        const width = (minX === Infinity || maxX === -Infinity) ? 0 : maxX - minX;
        const height = (minY === Infinity || maxY === -Infinity) ? 0 : maxY - minY;
        const depth = (minZ === Infinity || maxZ === -Infinity) ? 0 : maxZ - minZ;

        const maxDimension = Math.max(width, height, depth, 1); // Ensure maxDimension is at least 1

        setGraphCenter([centerX, centerY, centerZ]);
        setGraphSize(maxDimension > 0 ? maxDimension : 50);
        logger.debug('Graph initialized and centered.', { center: [centerX, centerY, centerZ], size: maxDimension });

      } catch (err) {
        logger.error('Failed to fetch initial graph data:', err);
        setError(err instanceof Error ? err.message : 'An unknown error occurred during data fetch.');
      } finally {
        setIsLoading(false);
      }
    };
    initializeGraph();
  }, []);

  useEffect(() => {
    // Enable lights only for default layer and layer 1
    // SelectiveBloom has issues with layer 2
    [ambientRef, dirRef, pointRef].forEach((l) => {
      if (l) {
        l.layers.enable(0); // Default layer
        l.layers.enable(1); // Bloom layer 1
        // Don't enable layer 2 for lights to avoid SelectiveBloom issues
      }
    });
  }, [ambientRef, dirRef, pointRef]);

  if (isLoading) {
    return <div style={{ padding: '2rem', color: '#ccc', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>Loading graph data...</div>;
  }

  if (error) {
    return <div style={{ padding: '2rem', color: 'red', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>Error loading graph data: {error}</div>;
  }

  const backgroundColor = renderingSettings?.backgroundColor ?? '#000000';

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <Canvas
        style={{ display: 'block', width: '100%', height: '100%' }}
        camera={{
          fov: fov,
          near: near,
          far: far,
          position: cameraPosition as [number, number, number],
        }}
        onCreated={({ gl, camera, scene }) => {
          if (debugState.isEnabled()) {
            logger.info('Canvas created', {
              cameraPosition: camera.position.toArray(),
              cameraFov: camera.fov,
              sceneChildren: scene.children.length,
            });
          }
          // Force a render
          gl.render(scene, camera);
        }}
        gl={{ 
          antialias: true, 
          alpha: true, // Enable alpha for proper transparency
          powerPreference: 'high-performance', 
          logarithmicDepthBuffer: false,
          toneMapping: THREE.ACESFilmicToneMapping, // Better tone mapping for bloom
          preserveDrawingBuffer: true,
          outputColorSpace: THREE.SRGBColorSpace // Ensure proper color space
        }}
        dpr={[1, 2]} // Pixel ratio for sharpness
        shadows // Enable shadows
      >
        <color attach="background" args={[backgroundColor]} />
        <CameraController center={graphCenter} size={graphSize} />
        

        <ambientLight ref={setAmbientRef} intensity={renderingSettings?.ambientLightIntensity ?? 0.6} />
        <directionalLight
          ref={setDirRef}
          position={[
            renderingSettings?.directionalLightPosition?.[0] ?? 10,
            renderingSettings?.directionalLightPosition?.[1] ?? 10,
            renderingSettings?.directionalLightPosition?.[2] ?? 5
          ]}
          intensity={renderingSettings?.directionalLightIntensity ?? 1}
          castShadow
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
          maxDistance={far / 2} // Max distance related to camera far plane
          target={graphCenter}
          enabled={!isNodeDragging} // <--- Control OrbitControls here
        />

            <Suspense fallback={null}>
              {/* Using GraphManager for all graph rendering */}
              <GraphManager />

            {/* World-class hologram effects controlled by hologram toggle */}
            {hologramEnabled && (
              <>
                <WorldClassHologram
                  enabled={hologramEnabled}
                  position={graphCenter}
                />
                <EnergyFieldParticles
                  count={1000}
                  bounds={graphSize * 2}
                  color={nodeSettings?.baseColor || '#00ffff'}
                />
                <EnhancedHologramSystem
                  position={graphCenter}
                  scale={graphSize > 0 ? graphSize / 100 : 1}
                />
              </>
            )}

              {/* VisionFlow visualization re-enabled in same origin space */}
              <BotsVisualization />
            </Suspense>

          {/* Debug visualizations based on debug settings */}
          {debugSettings?.enabled && (
            <>
              <Stats />
              <axesHelper args={[50]} />
              <gridHelper args={[200, 20]} />
              {debugSettings.enablePhysicsDebug && (
                <mesh>
                  <boxGeometry args={[5, 5, 5]} />
                  <meshBasicMaterial color="yellow" wireframe />
                </mesh>
              )}
            </>
          )}
          
          {/* Additional debug info logging */}
          {debugSettings?.enableNodeDebug && debugState.isEnabled() && 
            logger.info('Node debug enabled - Graph state:', { 
              graphSize, 
              graphCenter, 
              nodeCount: graphDataManager.getGraphData().then(d => d.nodes.length) 
            })
          }
          
          {debugSettings?.enableShaderDebug && debugState.isEnabled() &&
            logger.info('Shader debug enabled - Rendering state:', {
              enableGlow,
              hologramEnabled,
              renderingSettings
            })
          }

          {enableGlow && (
            <EffectComposer
              multisampling={0}
              autoClear={false}
              enabled={true}
            >
              <AtmosphericGlowWrapper />
            </EffectComposer>
          )}
      </Canvas>
    </div>
  );
};

export default GraphViewport;