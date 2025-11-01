import React, { useRef, useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stats } from '@react-three/drei';
import * as THREE from 'three';

// GraphManager for rendering the actual graph
import GraphManager from './GraphManager';
// Post-processing effects - using modern R3F selective bloom
import { SelectiveBloom } from '../../../rendering/SelectiveBloom';
// Bots visualization for agent graph
import { BotsVisualization } from '../../bots/components';
// SpacePilot Integration - using simpler version that works with useFrame
import SpacePilotSimpleIntegration from '../../visualisation/components/SpacePilotSimpleIntegration';
// Head Tracking for Parallax
import { HeadTrackedParallaxController } from '../../visualisation/components/HeadTrackedParallaxController';
// Hologram environment removed
// XR Support - causes graph to disappear
// import XRController from '../../xr/components/XRController';
// import XRVisualisationConnector from '../../xr/components/XRVisualisationConnector';

// Store and utils
import { useSettingsStore } from '../../../store/settingsStore';
import { graphDataManager, type GraphData } from '../managers/graphDataManager';
import { createLogger } from '../../../utils/loggerConfig';
import { HologramContent } from '../../visualisation/components/HolographicDataSphere';

const logger = createLogger('GraphCanvas');

// Main GraphCanvas component
const GraphCanvas: React.FC = () => {
    
    const containerRef = useRef<HTMLDivElement>(null);
    const orbitControlsRef = useRef<any>(null);
    const { settings } = useSettingsStore();
    const showStats = settings?.system?.debug?.enablePerformanceDebug ?? false;
    const xrEnabled = settings?.xr?.enabled !== false;
    const enableBloom = settings?.visualisation?.bloom?.enabled ?? false;
    const enableGlow = settings?.visualisation?.glow?.enabled ?? false;
    const useMultiLayerBloom = enableBloom || enableGlow; 
    const enableHologram = settings?.visualisation?.graphs?.logseq?.nodes?.enableHologram ?? false;
    
    
    const [graphData, setGraphData] = useState<GraphData>({ nodes: [], edges: [] });
    const [canvasReady, setCanvasReady] = useState(false);

    
    useEffect(() => {
        let mounted = true;
        
        
        const handleGraphData = (data: GraphData) => {
            if (mounted) {
                setGraphData(data);
            }
        };

        const unsubscribe = graphDataManager.onGraphDataChange(handleGraphData);
        
        
        graphDataManager.getGraphData().then((data) => {
            if (mounted) {
                setGraphData(data);
            }
        }).catch((error) => {
            console.error('[GraphCanvas] Failed to load initial graph data:', error);
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
                    backgroundColor: 'rgba(255, 0, 0, 0.5)',
                    padding: '5px 10px',
                    zIndex: 1000,
                    fontSize: '12px'
                }}>
                    Nodes: {graphData.nodes.length} | Edges: {graphData.edges.length} | Ready: {canvasReady ? 'Yes' : 'No'}
                </div>
            )}

            <Canvas
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
                {}
                <ambientLight intensity={0.15} />
                <directionalLight position={[10, 10, 10]} intensity={0.4} />
                
                {}
                {enableHologram && (
                  <HologramContent
                    opacity={0.1}
                    layer={2}
                    renderOrder={-1}
                    includeSwarm={false}
                    enableDepthFade={true}
                    fadeStart={2000}
                    fadeEnd={5000}
                  />
                )}
                
                {}
                {canvasReady && graphData.nodes.length > 0 && (
                    <GraphManager graphData={graphData} />
                )}
                
                {}
                
                {}
                <BotsVisualization />
                
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
                <SelectiveBloom enabled={enableBloom || enableGlow} />
                
                {}
                {showStats && <Stats />}
            </Canvas>
        </div>
    );
};

export default GraphCanvas;