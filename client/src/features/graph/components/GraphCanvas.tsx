import React, { useRef, useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stats } from '@react-three/drei';
import * as THREE from 'three';

// GraphManager for rendering the actual graph
import GraphManager from './GraphManager';
// Post-processing effects
import { PostProcessingEffects } from './PostProcessingEffects';
// Bots visualization for agent graph
import { BotsVisualization } from '../../bots/components';
// SpacePilot Integration
import { SpacePilotSimpleIntegration } from '../../visualisation/components/SpacePilotSimpleIntegration';
// XR Support - causes graph to disappear
// import XRController from '../../xr/components/XRController';
// import XRVisualisationConnector from '../../xr/components/XRVisualisationConnector';

// Store and utils
import { useSettingsStore } from '../../../store/settingsStore';
import { graphDataManager, type GraphData } from '../managers/graphDataManager';
import { createLogger } from '../../../utils/logger';

const logger = createLogger('GraphCanvas');

// Main GraphCanvas component
const GraphCanvas: React.FC = () => {
    console.log('[GraphCanvas] Component rendering');
    
    const containerRef = useRef<HTMLDivElement>(null);
    const { settings } = useSettingsStore();
    const showStats = settings?.system?.debug?.enablePerformanceDebug ?? false;
    const xrEnabled = settings?.xr?.enabled !== false;
    const enableBloom = settings?.visualisation?.bloom?.enabled ?? false;
    
    // Graph data state
    const [graphData, setGraphData] = useState<GraphData>({ nodes: [], edges: [] });
    const [canvasReady, setCanvasReady] = useState(false);

    // Subscribe to graph data updates
    useEffect(() => {
        let mounted = true;
        
        console.log('[GraphCanvas] Setting up graph data subscription');
        
        const handleGraphData = (data: GraphData) => {
            if (mounted) {
                console.log('[GraphCanvas] Received graph data update:', {
                    nodeCount: data.nodes.length,
                    edgeCount: data.edges.length
                });
                setGraphData(data);
            }
        };

        const unsubscribe = graphDataManager.onGraphDataChange(handleGraphData);
        
        // Get initial data
        graphDataManager.getGraphData().then((data) => {
            if (mounted) {
                console.log('[GraphCanvas] Initial graph data loaded:', {
                    nodeCount: data.nodes.length,
                    edgeCount: data.edges.length
                });
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
            
            <Canvas
                camera={{
                    fov: 75,
                    near: 0.1,
                    far: 2000,
                    position: [20, 15, 20]
                }}
                onCreated={({ gl, camera, scene }) => {
                    console.log('[GraphCanvas] Canvas created successfully');
                    gl.setClearColor(0x000033, 1);
                    setCanvasReady(true);
                    
                    // Log for debugging
                    console.log('[GraphCanvas] WebGL context established:', {
                        renderer: gl,
                        camera: camera.position.toArray(),
                        sceneChildren: scene.children.length
                    });
                }}
            >
                {/* Basic lighting */}
                <ambientLight intensity={0.5} />
                <directionalLight position={[10, 10, 10]} intensity={0.8} />
                
                {/* Graph Manager - only render when we have data and canvas is ready */}
                {canvasReady && graphData.nodes.length > 0 && (
                    <GraphManager graphData={graphData} />
                )}
                
                {/* Fallback cube if no graph data */}
                {canvasReady && graphData.nodes.length === 0 && (
                    <mesh position={[0, 0, 0]}>
                        <boxGeometry args={[2, 2, 2]} />
                        <meshStandardMaterial color="hotpink" />
                    </mesh>
                )}
                
                {/* BotsVisualization for agent graph */}
                <BotsVisualization />
                
                {/* Camera controls with SpacePilot integration */}
                <SpacePilotSimpleIntegration>
                    <OrbitControls
                        enablePan={true}
                        enableZoom={true}
                        enableRotate={true}
                        zoomSpeed={0.8}
                        panSpeed={0.8}
                        rotateSpeed={0.8}
                    />
                </SpacePilotSimpleIntegration>
                
                {/* XR Support - causes graph to disappear */}
                {/* {xrEnabled && <XRController />} */}
                {/* {xrEnabled && <XRVisualisationConnector />} */}
                
                {/* Post-processing effects */}
                {enableBloom && <PostProcessingEffects />}
                
                {/* Performance stats */}
                {showStats && <Stats />}
            </Canvas>
        </div>
    );
};

export default GraphCanvas;