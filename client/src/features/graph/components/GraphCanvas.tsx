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
import { AgentPollingStatus } from '../../bots/components/AgentPollingStatus';
// SpacePilot Integration - using simpler version that works with useFrame
import SpacePilotSimpleIntegration from '../../visualisation/components/SpacePilotSimpleIntegration';
// Consolidated hologram environment
import HologramEnvironment from '../../visualisation/components/HologramEnvironment';
// XR Support - causes graph to disappear
// import XRController from '../../xr/components/XRController';
// import XRVisualisationConnector from '../../xr/components/XRVisualisationConnector';

// Store and utils
import { useSettingsStore } from '../../../store/settingsStore';
import { graphDataManager, type GraphData } from '../managers/graphDataManager';
import { createLogger } from '../../../utils/loggerConfig';

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
    const enableHologram = settings?.visualisation?.graphs?.logseq?.nodes?.enableHologram !== false;
    const useMultiLayerBloom = enableBloom || enableGlow; // Use multi-layer if either is enabled
    
    // Graph data state
    const [graphData, setGraphData] = useState<GraphData>({ nodes: [], edges: [] });
    const [canvasReady, setCanvasReady] = useState(false);

    // Subscribe to graph data updates
    useEffect(() => {
        let mounted = true;
        
        
        const handleGraphData = (data: GraphData) => {
            if (mounted) {
                setGraphData(data);
            }
        };

        const unsubscribe = graphDataManager.onGraphDataChange(handleGraphData);
        
        // Get initial data
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
            {/* Debug indicator */}
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
            
            {/* Agent Polling Status Overlay */}
            <AgentPollingStatus />
            
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
                {/* Basic lighting */}
                <ambientLight intensity={0.5} />
                <directionalLight position={[10, 10, 10]} intensity={0.8} />
                
                {/* Consolidated hologram environment - all effects in one component on Layer 2 */}
                <HologramEnvironment 
                    enabled={enableHologram}
                    position={[0, 0, 0]}
                    useDiffuseEffects={true}
                />
                
                {/* Graph Manager - only render when we have data and canvas is ready */}
                {canvasReady && graphData.nodes.length > 0 && (
                    <GraphManager graphData={graphData} />
                )}
                
                {/* Fallback cube removed - was showing when graph data was loading */}
                
                {/* BotsVisualization for agent graph */}
                <BotsVisualization />
                
                {/* Camera controls with SpacePilot integration */}
                <OrbitControls
                    ref={orbitControlsRef}
                    enablePan={true}
                    enableZoom={true}
                    enableRotate={true}
                    zoomSpeed={0.8}
                    panSpeed={0.8}
                    rotateSpeed={0.8}
                />
                {/* Using the simpler SpacePilot integration that works with useFrame */}
                <SpacePilotSimpleIntegration />
                
                {/* XR Support - causes graph to disappear */}
                {/* {xrEnabled && <XRController />} */}
                {/* {xrEnabled && <XRVisualisationConnector />} */}
                
                {/* Post-processing effects - using modern R3F selective bloom */}
                <SelectiveBloom enabled={enableBloom || enableGlow} />
                
                {/* Performance stats */}
                {showStats && <Stats />}
            </Canvas>
        </div>
    );
};

export default GraphCanvas;