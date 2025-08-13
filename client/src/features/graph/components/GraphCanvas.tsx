import { useRef, useState, useEffect } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, Stats } from '@react-three/drei';
import * as THREE from 'three';

// Components
import GraphManager from './GraphManager';
import GraphFeatures from './GraphFeatures';
import { PostProcessingEffects } from './PostProcessingEffects';
import XRController from '../../xr/components/XRController';
import XRVisualisationConnector from '../../xr/components/XRVisualisationConnector';
import { BotsVisualization } from '../../bots/components/BotsVisualization';
// import { DualVisualizationControls } from './DualVisualizationControls'; // Removed - both graphs now at origin

// SpacePilot Integration
import { SpacePilotSimpleIntegration } from '../../visualisation/components/SpacePilotSimpleIntegration';

// Innovation Manager
import { innovationManager } from '../innovations/index';
import { graphDataManager, type GraphData } from '../managers/graphDataManager';

// Store and utils
import { useSettingsStore } from '../../../store/settingsStore';
import { createLogger } from '../../../utils/logger';
import { clientDebugState as debugState } from '../../../utils/clientDebugState';

const logger = createLogger('GraphCanvas');

// Scene setup with lighting and background
const SceneSetup = () => {
    const { scene } = useThree();
    const settings = useSettingsStore(state => state.settings?.visualisation);

    // Render lights using JSX
    return (
        <>
            <color attach="background" args={[0, 0, 0.8]} /> {/* Medium blue background */}
            <ambientLight intensity={0.6} />
            <directionalLight
                intensity={0.8}
                position={[1, 1, 1]}
            />
        </>
    );
};

// Main GraphCanvas component
const GraphCanvas = () => {
    // console.log('[GRAPH CANVAS] Component rendering...');
    // console.log('[GRAPH CANVAS] About to render BotsVisualization');
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const cameraRef = useRef<THREE.Camera>(null);
    const { settings } = useSettingsStore();
    const showStats = settings?.system?.debug?.enablePerformanceDebug ?? false; // Use performance debug flag
    const xrEnabled = settings?.xr?.enabled !== false;
    const antialias = settings?.visualisation?.rendering?.enableAntialiasing !== false; // Correct property name

    // Graph data state for features
    const [graphData, setGraphData] = useState<GraphData>({ nodes: [], edges: [] });
    const [isInitialized, setIsInitialized] = useState(false);

    // Initialize innovation manager and features
    useEffect(() => {
        let mounted = true;
        
        const initializeFeatures = async () => {
            try {
                logger.info('Initializing world-class graph features...');
                
                // Initialize with essential features for production use
                await innovationManager.initialize({
                    enableSync: true,
                    enableComparison: true,
                    enableAnimations: true,
                    enableAI: settings?.ai?.enabled ?? false,
                    enableAdvancedInteractions: settings?.xr?.enabled ?? false,
                    performanceMode: settings?.system?.performanceMode as any || 'balanced'
                });
                
                if (mounted) {
                    setIsInitialized(true);
                    logger.info('Graph features initialized successfully');
                }
            } catch (error) {
                logger.error('Failed to initialize graph features:', error);
            }
        };

        // Subscribe to graph data updates
        const handleGraphData = (data: GraphData) => {
            if (mounted) {
                setGraphData(data);
            }
        };

        const unsubscribeGraphData = graphDataManager.onGraphDataChange(handleGraphData);
        initializeFeatures();

        return () => {
            mounted = false;
            unsubscribeGraphData();
        };
    }, [settings?.ai?.enabled, settings?.xr?.enabled, settings?.system?.performanceMode]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            innovationManager.dispose();
        };
    }, []);

    // Both visualizations now positioned at origin (0, 0, 0) for unified view

    // Wrapper to ensure proper canvas sizing
    return (
        <div style={{ width: '100%', height: '100%', backgroundColor: '#000033' }}>
            <Canvas
                ref={canvasRef}
                gl={{
                    antialias,
                    alpha: true,
                    powerPreference: 'high-performance',
                    failIfMajorPerformanceCaveat: false
                }}
                camera={{
                    fov: 75,
                    near: 0.1,
                    far: 2000,
                    position: [40, 30, 40] // Better angle to see both visualization systems
                }}
                onCreated={({ gl, camera }) => {
                    cameraRef.current = camera;
                    if (debugState.isEnabled()) {
                        logger.debug('Canvas created with dimensions:', {
                            width: gl.domElement.width,
                            height: gl.domElement.height
                        });
                    }
                }}
            >
                <SceneSetup />

                {/* Logseq Graph Visualization - positioned at origin */}
                <group position={[0, 0, 0]}>
                    <GraphManager />
                </group>

                {/* VisionFlow Bots Visualization - also positioned at origin for unified view */}
                <group position={[0, 0, 0]}>
                    <BotsVisualization />
                </group>

                {/* World-Class Innovative Graph Features */}
                {isInitialized && cameraRef.current && (
                    <GraphFeatures
                        graphId="main-graph"
                        graphData={graphData}
                        isVisible={true}
                        camera={cameraRef.current}
                        onFeatureUpdate={(feature, data) => {
                            if (debugState.isEnabled()) {
                                logger.debug(`Feature update: ${feature}`, data);
                            }
                        }}
                    />
                )}

                {/* Camera Controls with SpacePilot Integration */}
                <OrbitControls
                    enablePan={true}
                    enableZoom={true}
                    enableRotate={true}
                    zoomSpeed={0.8}
                    panSpeed={0.8}
                    rotateSpeed={0.8}
                />

                {xrEnabled && <XRController />}
                {xrEnabled && <XRVisualisationConnector />}
                {showStats && <Stats />}
                {settings?.visualisation?.bloom?.enabled && <PostProcessingEffects />}
            </Canvas>
        </div>
    );
};

export default GraphCanvas;
