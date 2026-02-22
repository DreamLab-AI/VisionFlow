import React, { useRef, useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stats, Environment } from '@react-three/drei';
import { createGemRenderer } from '../../../rendering/rendererFactory';
import { createLogger } from '../../../utils/loggerConfig';

const logger = createLogger('GraphCanvas');

// GraphManager for rendering the actual graph
import GraphManager from './GraphManager';
// Post-processing effects - unified gem post-processing (WebGPU + WebGL bloom)
import { GemPostProcessing } from '../../../rendering/GemPostProcessing';
// Bots visualization for agent graph
import { BotsVisualization } from '../../bots/components';
// Agent action connections visualization
import { AgentActionVisualization } from '../../visualisation/components/AgentActionVisualization';
// SpacePilot Integration - using simpler version that works with useFrame
import SpacePilotSimpleIntegration from '../../visualisation/components/SpacePilotSimpleIntegration';
// Head Tracking for Parallax
import { HeadTrackedParallaxController } from '../../visualisation/components/HeadTrackedParallaxController';
// XR Support - causes graph to disappear
// import XRController from '../../xr/components/XRController';
// import XRVisualisationConnector from '../../xr/components/XRVisualisationConnector';

// Scene ambient effects (particles, fog, glow ring)
import WasmSceneEffects from '../../visualisation/components/WasmSceneEffects';

// Store and utils
import { useSettingsStore } from '../../../store/settingsStore';
import { graphDataManager, type GraphData } from '../managers/graphDataManager';

// Main GraphCanvas component
const GraphCanvas: React.FC = () => {

    const containerRef = useRef<HTMLDivElement>(null);
    const orbitControlsRef = useRef<any>(null);
    // Narrow selectors — prevent full Canvas tree re-render on unrelated settings changes.
    // Each selector returns a primitive or stable nested ref so Zustand's Object.is comparison
    // only triggers re-renders when that specific value actually changes.
    const showStats = useSettingsStore(s => s.settings?.system?.debug?.enablePerformanceDebug ?? false);
    const enableGlow = useSettingsStore(s => s.settings?.visualisation?.glow?.enabled !== false);
    const ambientLightIntensity = useSettingsStore(s => s.settings?.visualisation?.rendering?.ambientLightIntensity ?? 0.4);
    const directionalLightIntensity = useSettingsStore(s => s.settings?.visualisation?.rendering?.directionalLightIntensity ?? 0.8);
    const sceneEffects = useSettingsStore(s => s.settings?.visualisation?.sceneEffects);
    
    // Lightweight subscription: only track counts to avoid storing full graph data in two places
    const [nodeCount, setNodeCount] = useState(0);
    const [edgeCount, setEdgeCount] = useState(0);
    const [canvasReady, setCanvasReady] = useState(false);

    useEffect(() => {
        let mounted = true;

        const handleGraphData = (data: GraphData) => {
            if (mounted) {
                setNodeCount(data.nodes.length);
                setEdgeCount(data.edges.length);
            }
        };

        const unsubscribe = graphDataManager.onGraphDataChange(handleGraphData);

        graphDataManager.getGraphData().then((data) => {
            if (mounted) {
                setNodeCount(data.nodes.length);
                setEdgeCount(data.edges.length);
            }
        }).catch((error) => {
            logger.error('Failed to load initial graph data:', error);
        });

        return () => {
            mounted = false;
            unsubscribe();
        };
    }, []);

    return (
        <div 
            ref={containerRef}
            style={{ 
                position: 'fixed',
                top: 0,
                left: 0,
                width: '100vw', 
                height: '100vh',
                backgroundColor: '#000033',
                zIndex: 0
            }}
        >
            {}
            {showStats && (
                <div style={{
                    position: 'absolute',
                    top: '10px',
                    left: '10px',
                    color: 'white',
                    backgroundColor: 'rgba(0, 0, 0, 0.6)',
                    padding: '5px 10px',
                    zIndex: 1000,
                    fontSize: '12px'
                }}>
                    Nodes: {nodeCount} | Edges: {edgeCount} | Ready: {canvasReady ? 'Yes' : 'No'}
                </div>
            )}

            <Canvas
                gl={createGemRenderer}
                dpr={[1, 2]}
                camera={{
                    fov: 75,
                    near: 0.1,
                    far: 2000,
                    position: [20, 15, 20]
                }}
                onCreated={({ gl, camera, scene }) => {
                    gl.setClearColor(0x000033, 1);
                    setCanvasReady(true);
                }}
            >
                {/* Lighting tuned for gem refraction -- driven by settings */}
                <ambientLight intensity={ambientLightIntensity} />
                <directionalLight position={[10, 10, 10]} intensity={directionalLightIntensity} />
                <directionalLight position={[-5, -5, -10]} intensity={0.3} />

                {/* Environment map for PBR glass material reflections */}
                <Environment preset="studio" background={false} resolution={512} />

                {/* Scene ambient effects (WASM particles, wisps, atmosphere) */}
                <WasmSceneEffects
                    enabled={sceneEffects?.enabled !== false}
                    particleCount={sceneEffects?.particleCount ?? 256}
                    intensity={sceneEffects?.particleOpacity ?? 0.6}
                    particleDrift={sceneEffects?.particleDrift ?? 0.5}
                    wispsEnabled={sceneEffects?.wispsEnabled !== false}
                    wispCount={sceneEffects?.wispCount ?? 48}
                    wispDriftSpeed={sceneEffects?.wispDriftSpeed ?? 1.0}
                    atmosphereEnabled={sceneEffects?.fogEnabled !== false}
                    atmosphereResolution={sceneEffects?.atmosphereResolution ?? 128}
                />

                {}
                {canvasReady && nodeCount > 0 && (
                    <GraphManager />
                )}
                
                {}
                
                {}
                <BotsVisualization />

                {/* Agent Action Connections - ephemeral animated connections */}
                <AgentActionVisualization showStats={showStats} />

                {}
                <OrbitControls
                    ref={orbitControlsRef}
                    enablePan={true}
                    enableZoom={true}
                    enableRotate={true}
                    zoomSpeed={0.8}
                    panSpeed={0.8}
                    rotateSpeed={0.8}
                />
                {}
                <SpacePilotSimpleIntegration orbitControlsRef={orbitControlsRef} />

                {}
                <HeadTrackedParallaxController />

                {}
                {}
                {}
                
                {}
                <GemPostProcessing enabled={enableGlow} />
                
                {}
                {showStats && <Stats />}
            </Canvas>
        </div>
    );
};

export default GraphCanvas;