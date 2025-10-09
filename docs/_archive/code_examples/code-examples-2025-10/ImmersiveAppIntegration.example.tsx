/**
 * ImmersiveAppIntegration - Example integration of Vircadia multi-user services
 *
 * This example shows how to integrate the Vircadia multi-user system
 * into the existing ImmersiveApp component for Quest 3 XR.
 */

import React, { useEffect, useRef } from 'react';
import * as BABYLON from '@babylonjs/core';
import { useVircadiaXR } from '../../contexts/VircadiaContext';
import { VircadiaSceneBridge } from '../babylon/VircadiaSceneBridge';
import { AvatarManager } from '../../services/vircadia/AvatarManager';
import { SpatialAudioManager } from '../../services/vircadia/SpatialAudioManager';
import { CollaborativeGraphSync } from '../../services/vircadia/CollaborativeGraphSync';

export const ImmersiveAppIntegration: React.FC = () => {
    const { client, isConnected, xrReady, agentId } = useVircadiaXR();

    const sceneRef = useRef<BABYLON.Scene | null>(null);
    const cameraRef = useRef<BABYLON.Camera | null>(null);
    const xrHelperRef = useRef<BABYLON.WebXRDefaultExperience | null>(null);

    const sceneBridgeRef = useRef<VircadiaSceneBridge | null>(null);
    const avatarManagerRef = useRef<AvatarManager | null>(null);
    const audioManagerRef = useRef<SpatialAudioManager | null>(null);
    const collabSyncRef = useRef<CollaborativeGraphSync | null>(null);

    /**
     * Initialize Babylon.js scene with XR support
     */
    useEffect(() => {
        const canvas = document.getElementById('renderCanvas') as HTMLCanvasElement;
        if (!canvas) return;

        const engine = new BABYLON.Engine(canvas, true);
        const scene = new BABYLON.Scene(engine);
        sceneRef.current = scene;

        // Create XR camera
        const camera = new BABYLON.UniversalCamera('xr-camera', new BABYLON.Vector3(0, 1.6, -3), scene);
        camera.attachControl(canvas, true);
        cameraRef.current = camera;

        // XR setup for Quest 3
        scene.createDefaultXRExperienceAsync({
            floorMeshes: [],
            optionalFeatures: true
        }).then((xrHelper) => {
            xrHelperRef.current = xrHelper;

            // Enable AR passthrough for Quest 3
            xrHelper.baseExperience.featuresManager.enableFeature(
                BABYLON.WebXRFeatureName.HAND_TRACKING,
                'latest',
                { xrInput: xrHelper.input }
            );
        });

        // Render loop
        engine.runRenderLoop(() => {
            scene.render();
        });

        // Cleanup
        return () => {
            scene.dispose();
            engine.dispose();
        };
    }, []);

    /**
     * Initialize Vircadia multi-user services when connected
     */
    useEffect(() => {
        if (!isConnected || !client || !sceneRef.current || !cameraRef.current || !xrReady) {
            return;
        }

        const scene = sceneRef.current;
        const camera = cameraRef.current;

        // 1. Initialize Scene Bridge for entity synchronization
        const sceneBridge = new VircadiaSceneBridge(scene, client, {
            enableRealTimeSync: true,
            instancedRendering: true,
            enableLOD: true,
            maxRenderDistance: 50
        });
        sceneBridgeRef.current = sceneBridge;

        // Load graph from Vircadia
        sceneBridge.loadGraphFromVircadia().catch((error) => {
            console.error('Failed to load graph from Vircadia:', error);
        });

        // 2. Initialize Avatar Manager for multi-user presence
        const avatarManager = new AvatarManager(scene, client, camera, {
            modelUrl: '/assets/avatars/default-avatar.glb',
            scale: 1.0,
            showNameplate: true,
            nameplateDistance: 10.0,
            enableAnimations: true
        });
        avatarManagerRef.current = avatarManager;

        // Create local avatar
        if (agentId) {
            avatarManager.createLocalAvatar(`User_${agentId.substring(0, 8)}`).catch((error) => {
                console.error('Failed to create local avatar:', error);
            });
        }

        // 3. Initialize Spatial Audio Manager for 3D voice chat
        const audioManager = new SpatialAudioManager(client, scene, {
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' },
                { urls: 'stun:stun1.l.google.com:19302' }
            ],
            audioConstraints: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: 48000
            },
            maxDistance: 20,
            rolloffFactor: 1,
            refDistance: 1
        });
        audioManagerRef.current = audioManager;

        // Initialize audio
        audioManager.initialize().catch((error) => {
            console.error('Failed to initialize spatial audio:', error);
        });

        // 4. Initialize Collaborative Graph Sync for shared interactions
        const collabSync = new CollaborativeGraphSync(scene, client, {
            highlightColor: new BABYLON.Color3(0.2, 0.8, 0.3),
            annotationColor: new BABYLON.Color3(1.0, 0.8, 0.2),
            selectionTimeout: 30000,
            enableAnnotations: true,
            enableFiltering: true
        });
        collabSyncRef.current = collabSync;

        // Initialize collaborative sync
        collabSync.initialize().catch((error) => {
            console.error('Failed to initialize collaborative sync:', error);
        });

        // 5. Set up render loop updates
        scene.onBeforeRenderObservable.add(() => {
            // Update listener position for spatial audio
            if (camera && audioManager) {
                const forward = camera.getForwardRay().direction;
                const up = camera.upVector;
                audioManager.updateListenerPosition(camera.position, forward, up);
            }

            // Update audio positions for remote avatars
            if (audioManager && avatarManager) {
                const avatars = avatarManager.getAvatars();
                avatars.forEach((avatar) => {
                    if (avatar.agentId !== agentId) {
                        audioManager.updatePeerPosition(avatar.agentId, avatar.position);
                    }
                });
            }
        });

        // Cleanup
        return () => {
            sceneBridge?.dispose();
            avatarManager?.dispose();
            audioManager?.dispose();
            collabSync?.dispose();
        };
    }, [isConnected, client, xrReady, agentId]);

    /**
     * Example: Handle node selection for collaborative highlighting
     */
    const handleNodeClick = async (nodeId: string) => {
        if (!collabSyncRef.current) return;

        // Select node locally and broadcast to other users
        await collabSyncRef.current.selectNodes([nodeId]);

        console.log(`Node selected: ${nodeId}`);
    };

    /**
     * Example: Create annotation on a node
     */
    const handleCreateAnnotation = async (nodeId: string, text: string, position: BABYLON.Vector3) => {
        if (!collabSyncRef.current) return;

        await collabSyncRef.current.createAnnotation(nodeId, text, position);

        console.log(`Annotation created: "${text}" on node ${nodeId}`);
    };

    /**
     * Example: Toggle microphone mute
     */
    const handleToggleMute = () => {
        if (!audioManagerRef.current) return;

        const isMuted = audioManagerRef.current.toggleMute();
        console.log(`Microphone ${isMuted ? 'muted' : 'unmuted'}`);
    };

    /**
     * Example: Connect to peer for voice chat
     */
    const handleConnectToPeer = async (peerAgentId: string, peerUsername: string) => {
        if (!audioManagerRef.current) return;

        await audioManagerRef.current.connectToPeer(peerAgentId, peerUsername);
        console.log(`Connected to peer: ${peerUsername}`);
    };

    /**
     * Example: Get collaboration statistics
     */
    const getCollaborationStats = () => {
        const stats = {
            connectedUsers: avatarManagerRef.current?.getAvatarCount() || 0,
            activeSelections: collabSyncRef.current?.getActiveSelections().length || 0,
            annotations: collabSyncRef.current?.getAnnotations().length || 0,
            syncStats: sceneBridgeRef.current?.getStats() || {}
        };

        console.log('Collaboration Stats:', stats);
        return stats;
    };

    return (
        <div style={{ width: '100%', height: '100%', position: 'relative' }}>
            <canvas
                id="renderCanvas"
                style={{ width: '100%', height: '100%', display: 'block' }}
            />

            {/* Example UI Controls */}
            {isConnected && xrReady && (
                <div style={{
                    position: 'absolute',
                    top: '10px',
                    left: '10px',
                    backgroundColor: 'rgba(0, 0, 0, 0.7)',
                    color: 'white',
                    padding: '10px',
                    borderRadius: '5px'
                }}>
                    <div>Connected Users: {avatarManagerRef.current?.getAvatarCount() || 0}</div>
                    <div>Agent ID: {agentId?.substring(0, 8)}...</div>
                    <button onClick={handleToggleMute}>Toggle Mic</button>
                    <button onClick={getCollaborationStats}>Get Stats</button>
                </div>
            )}

            {!isConnected && (
                <div style={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    color: 'white',
                    padding: '20px',
                    borderRadius: '10px'
                }}>
                    Connecting to Vircadia...
                </div>
            )}
        </div>
    );
};

/**
 * Usage in App.tsx:
 *
 * import { VircadiaProvider } from './contexts/VircadiaContext';
 * import { ImmersiveAppIntegration } from './immersive/components/ImmersiveAppIntegration.example';
 *
 * function App() {
 *   return (
 *     <VircadiaProvider autoConnect={true}>
 *       <ImmersiveAppIntegration />
 *     </VircadiaProvider>
 *   );
 * }
 */
