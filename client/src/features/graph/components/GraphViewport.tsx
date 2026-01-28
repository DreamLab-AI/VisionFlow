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
import { createLogger } from '../../../utils/loggerConfig';
import { debugState } from '../../../utils/clientDebugState';
import { BotsVisualization } from '../../bots/components';
import { HologramContent } from '../../visualisation/components/HolographicDataSphere';

// Ensure Three.js types are properly loaded if not globally done
// import '../../../types/react-three-fiber.d.ts';

const logger = createLogger('GraphViewport');

const AtmosphericGlowWrapper = () => {
  const { camera, size } = useThree();
  const glowSettings = useSettingsStore(state => state.settings?.visualisation?.glow);
  
  
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
  const [graphSize, setGraphSize] = useState(50); 
  const [isNodeDragging, setIsNodeDragging] = useState(false);
  
  // Performance: Removed drag state logging (was causing console spam)
  const [ambientRef, setAmbientRef] = useState<THREE.AmbientLight | null>(null);
  const [dirRef, setDirRef] = useState<THREE.DirectionalLight | null>(null);
  const [pointRef, setPointRef] = useState<THREE.PointLight | null>(null);

  
  const settings = useSettingsStore(state => state.settings);
  const initialized = useSettingsStore(state => state.initialized);
  const [viewportRefresh, setViewportRefresh] = useState(0);

  
  useEffect(() => {
    const unsubscribe = useSettingsStore.getState().subscribe(
      'viewport.update',
      () => {
        logger.debug('Viewport update triggered');
        setViewportRefresh(prev => prev + 1); 
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
  const enableHologram = settings?.visualisation?.graphs?.logseq?.nodes?.enableHologram ?? false;

  const fov = cameraSettings?.fov ?? 75;
  const near = cameraSettings?.near ?? 0.1;
  const far = cameraSettings?.far ?? 2000;

  
  const cameraPosition = useMemo(() => (
    cameraSettings?.position
      ? [cameraSettings.position.x, cameraSettings.position.y, cameraSettings.position.z]
      : [0, 10, 50] 
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
          setGraphSize(50); 
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

        const maxDimension = Math.max(width, height, depth, 1); 

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
    
    
    [ambientRef, dirRef, pointRef].forEach((l) => {
      if (l) {
        l.layers.enable(0); 
        l.layers.enable(1); 
        
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
              cameraFov: 'fov' in camera ? (camera as THREE.PerspectiveCamera).fov : undefined,
              sceneChildren: scene.children.length,
            });
          }
          
          gl.render(scene, camera);
        }}
        gl={{ 
          antialias: true, 
          alpha: true, 
          powerPreference: 'high-performance', 
          logarithmicDepthBuffer: false,
          toneMapping: THREE.ACESFilmicToneMapping, 
          preserveDrawingBuffer: true,
          outputColorSpace: THREE.SRGBColorSpace 
        }}
        dpr={[1, 2]} 
        shadows 
      >
        <color attach="background" args={[backgroundColor]} />
        <CameraController center={graphCenter} size={graphSize} />
        

        <ambientLight ref={setAmbientRef} intensity={renderingSettings?.ambientLightIntensity ?? 0.6} />
        <directionalLight
          ref={setDirRef}
          position={[10, 10, 5]}
          intensity={renderingSettings?.directionalLightIntensity ?? 1}
          castShadow
        />
        <pointLight
          ref={setPointRef}
          position={[-10, -10, -5]}
          intensity={renderingSettings?.ambientLightIntensity ?? 0.5}
        />

        <OrbitControls
          key="main-orbit-controls"
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
              {}
              <GraphManager onDragStateChange={(isDragging) => {
                console.log('GraphViewport: onDragStateChange called with', isDragging);
                setIsNodeDragging(isDragging);
              }} />

            {}
            {enableHologram && (
              <HologramContent
                opacity={0.15}
                layer={2}
                renderOrder={-1}
                includeSwarm={false}
                enableDepthFade={true}
                fadeStart={1600}
                fadeEnd={4000}
              />
            )}

              {}
              <BotsVisualization />
            </Suspense>

          {}
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

          {}
          {debugSettings?.enableNodeDebug && debugState.isEnabled() && (() => {
            logger.info('Node debug enabled - Graph state:', {
              graphSize,
              graphCenter,
              nodeCount: graphDataManager.getGraphData().then(d => d.nodes.length)
            });
            return null;
          })()}

          {debugSettings?.enableShaderDebug && debugState.isEnabled() && (() => {
            logger.info('Shader debug enabled - Rendering state:', {
              enableGlow,
              enableHologram,
              renderingSettings
            });
            return null;
          })()}

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