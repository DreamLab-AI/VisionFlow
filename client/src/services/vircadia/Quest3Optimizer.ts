// TODO: Migrated from Babylon.js to Three.js - requires WebXR integration refactoring

import * as THREE from 'three';
import { ClientCore } from './VircadiaClientCore';
import { createLogger } from '../../utils/loggerConfig';

const logger = createLogger('Quest3Optimizer');

export interface Quest3Config {
    targetFrameRate: 90 | 120;
    enableHandTracking: boolean;
    enableControllers: boolean;
    foveatedRenderingLevel: 0 | 1 | 2 | 3;
    dynamicResolutionScale: boolean;
    minResolutionScale: number;
    maxResolutionScale: number;
}

export interface HandJoint {
    name: string;
    position: THREE.Vector3;
    orientation: THREE.Quaternion;
}

export interface HandTrackingData {
    agentId: string;
    hand: 'left' | 'right';
    joints: HandJoint[];
    timestamp: number;
}

export interface ControllerState {
    agentId: string;
    controllerId: 'left' | 'right';
    position: THREE.Vector3;
    orientation: THREE.Quaternion;
    buttons: Record<string, boolean>;
    axes: Record<string, number>;
    timestamp: number;
}

export class Quest3Optimizer {
    private xrSession: XRSession | null = null;
    private localAgentId: string | null = null;
    private handUpdateInterval: ReturnType<typeof setInterval> | null = null;
    private controllerUpdateInterval: ReturnType<typeof setInterval> | null = null;
    private remoteHands = new Map<string, THREE.Mesh[]>();
    private remoteControllers = new Map<string, THREE.Mesh>();
    private inputSourcesChangeHandler: ((event: Event) => void) | null = null;
    private currentFPS = 0;
    private lastFrameTime = 0;
    private frameCount = 0;

    private defaultConfig: Quest3Config = {
        targetFrameRate: 90,
        enableHandTracking: true,
        enableControllers: true,
        foveatedRenderingLevel: 2,
        dynamicResolutionScale: true,
        minResolutionScale: 0.5,
        maxResolutionScale: 1.0
    };

    constructor(
        private scene: THREE.Scene,
        private renderer: THREE.WebGLRenderer,
        private client: ClientCore,
        config?: Partial<Quest3Config>
    ) {
        this.defaultConfig = { ...this.defaultConfig, ...config };
    }


    async initialize(xrSession: XRSession): Promise<void> {
        logger.info('Initializing Quest 3 optimizations...');

        this.xrSession = xrSession;


        const info = this.client.Utilities.Connection.getConnectionInfo();
        if (info.agentId) {
            this.localAgentId = info.agentId;
        }


        this.setupPerformanceMonitoring();


        this.setupFoveatedRendering();


        if (this.defaultConfig.dynamicResolutionScale) {
            this.setupDynamicResolution();
        }

        this.setupAnimationLoop();


        if (this.defaultConfig.enableHandTracking) {
            await this.setupHandTracking();
        }


        if (this.defaultConfig.enableControllers) {
            this.setupControllers();
        }

        logger.info('Quest 3 optimizations initialized');
    }


    private setupPerformanceMonitoring(): void {
        this.lastFrameTime = performance.now();
        logger.info('Performance monitoring enabled');
    }


    private setupFoveatedRendering(): void {
        if (!this.xrSession) return;

        try {

            const glLayer = this.xrSession.renderState.baseLayer as any;
            if (glLayer && 'fixedFoveation' in glLayer) {
                glLayer.fixedFoveation = this.defaultConfig.foveatedRenderingLevel;
                logger.info(`Foveated rendering enabled: level ${this.defaultConfig.foveatedRenderingLevel}`);
            }
        } catch (error) {
            logger.warn('Failed to enable foveated rendering:', error);
        }
    }


    private setupDynamicResolution(): void {
        logger.info('Dynamic resolution scaling enabled');
    }

    private setupAnimationLoop(): void {
        const { targetFrameRate, minResolutionScale, maxResolutionScale } = this.defaultConfig;

        this.renderer.setAnimationLoop((_timestamp: number) => {
            // Performance monitoring: update FPS metrics
            const currentTime = performance.now();
            this.frameCount++;

            if (currentTime >= this.lastFrameTime + 1000) {
                this.currentFPS = Math.round((this.frameCount * 1000) / (currentTime - this.lastFrameTime));
                this.frameCount = 0;
                this.lastFrameTime = currentTime;
            }

            // Dynamic resolution: adjust based on current FPS
            if (this.defaultConfig.dynamicResolutionScale) {
                const fps = this.currentFPS;
                const currentPixelRatio = this.renderer.getPixelRatio();

                if (fps < targetFrameRate - 10) {
                    const newPixelRatio = Math.max(currentPixelRatio * 0.9, minResolutionScale);
                    this.renderer.setPixelRatio(newPixelRatio);
                    logger.debug(`Resolution scaled down: ${newPixelRatio.toFixed(2)}`);
                } else if (fps > targetFrameRate + 5) {
                    const newPixelRatio = Math.min(currentPixelRatio * 1.05, maxResolutionScale);
                    this.renderer.setPixelRatio(newPixelRatio);
                    logger.debug(`Resolution scaled up: ${newPixelRatio.toFixed(2)}`);
                }
            }
        });

        logger.info('Animation loop started (performance monitoring + dynamic resolution)');
    }


    private async setupHandTracking(): Promise<void> {
        if (!this.xrSession) return;

        try {

            this.startHandTracking();

            logger.info('Hand tracking enabled');

        } catch (error) {
            logger.error('Failed to enable hand tracking:', error);
        }
    }


    private startHandTracking(): void {
        if (this.handUpdateInterval) {
            return;
        }

        this.handUpdateInterval = setInterval(async () => {
            if (!this.xrSession || !this.localAgentId) {
                return;
            }


            const inputSources = this.xrSession.inputSources;

            for (const source of inputSources) {
                if (source.hand) {
                    const hand = source.handedness as 'left' | 'right';
                    await this.broadcastHandData(hand, source.hand);
                }
            }

        }, 50);

        logger.info('Hand tracking broadcast started');
    }


    private async broadcastHandData(hand: 'left' | 'right', handData: XRHand): Promise<void> {
        if (!this.localAgentId) return;

        try {
            const joints: HandJoint[] = [];


            for (const joint of handData.values()) {
                if (joint) {
                    const position = new THREE.Vector3();
                    const orientation = new THREE.Quaternion();

                    joints.push({
                        name: joint.jointName || '',
                        position: position.clone(),
                        orientation: orientation.clone()
                    });
                }
            }

            const jsonPath = `{handTracking,${hand}}`;
            const query = `
                UPDATE entity.entities
                SET meta__data = jsonb_set(
                    meta__data,
                    $1::text[],
                    $2::jsonb
                )
                WHERE general__entity_name = $3
            `;

            await this.client.Utilities.Connection.query({
                query,
                parameters: [
                    jsonPath,
                    JSON.stringify({
                        joints,
                        timestamp: Date.now()
                    }),
                    `avatar_${this.localAgentId}`
                ],
                timeoutMs: 1000
            });

        } catch (error) {
            logger.debug('Failed to broadcast hand data:', error);
        }
    }


    updateRemoteHandTracking(agentId: string, hand: 'left' | 'right', joints: HandJoint[]): void {
        const handKey = `${agentId}_${hand}`;
        let handMeshes = this.remoteHands.get(handKey);

        if (!handMeshes) {

            handMeshes = joints.map((joint, index) => {
                const geometry = new THREE.SphereGeometry(0.0075, 8, 8);
                const material = new THREE.MeshBasicMaterial({
                    color: hand === 'left' ? 0x3380ff : 0xff8033
                });
                const sphere = new THREE.Mesh(geometry, material);
                sphere.name = `hand_${handKey}_${index}`;

                this.scene.add(sphere);
                return sphere;
            });

            this.remoteHands.set(handKey, handMeshes);
        }


        joints.forEach((joint, index) => {
            if (handMeshes![index]) {
                handMeshes![index].position.copy(joint.position);
                handMeshes![index].quaternion.copy(joint.orientation);
            }
        });
    }


    private setupControllers(): void {
        if (!this.xrSession) return;

        this.inputSourcesChangeHandler = (_event: Event) => {
            logger.info('Input sources changed');
            this.startControllerBroadcast();
        };
        this.xrSession.addEventListener('inputsourceschange', this.inputSourcesChangeHandler);

        logger.info('Controller support enabled');
    }


    private startControllerBroadcast(): void {
        if (this.controllerUpdateInterval) {
            return;
        }

        this.controllerUpdateInterval = setInterval(async () => {
            if (!this.xrSession || !this.localAgentId) {
                return;
            }

            const inputSources = this.xrSession.inputSources;

            for (const source of inputSources) {
                if (source.gamepad && source.gripSpace) {
                    const controllerState: ControllerState = {
                        agentId: this.localAgentId!,
                        controllerId: source.handedness as 'left' | 'right',
                        position: new THREE.Vector3(),
                        orientation: new THREE.Quaternion(),
                        buttons: {},
                        axes: {},
                        timestamp: Date.now()
                    };


                    source.gamepad.buttons.forEach((button, index) => {
                        controllerState.buttons[`button_${index}`] = button.pressed;
                    });

                    source.gamepad.axes.forEach((axis, index) => {
                        controllerState.axes[`axis_${index}`] = axis;
                    });

                    await this.broadcastControllerState(controllerState);
                }
            }

        }, 50);

        logger.info('Controller broadcast started');
    }


    private async broadcastControllerState(state: ControllerState): Promise<void> {
        try {
            const jsonPath = `{controller,${state.controllerId}}`;
            const query = `
                UPDATE entity.entities
                SET meta__data = jsonb_set(
                    meta__data,
                    $1::text[],
                    $2::jsonb
                )
                WHERE general__entity_name = $3
            `;

            await this.client.Utilities.Connection.query({
                query,
                parameters: [
                    jsonPath,
                    JSON.stringify({
                        position: {
                            x: state.position.x,
                            y: state.position.y,
                            z: state.position.z
                        },
                        orientation: {
                            x: state.orientation.x,
                            y: state.orientation.y,
                            z: state.orientation.z,
                            w: state.orientation.w
                        },
                        buttons: state.buttons,
                        axes: state.axes,
                        timestamp: state.timestamp
                    }),
                    `avatar_${this.localAgentId}`
                ],
                timeoutMs: 1000
            });

        } catch (error) {
            logger.debug('Failed to broadcast controller state:', error);
        }
    }


    updateRemoteController(agentId: string, state: ControllerState): void {
        const controllerKey = `${agentId}_${state.controllerId}`;
        let controllerMesh = this.remoteControllers.get(controllerKey);

        if (!controllerMesh) {

            const geometry = new THREE.CylinderGeometry(0.015, 0.015, 0.15, 8);
            const material = new THREE.MeshBasicMaterial({
                color: state.controllerId === 'left' ? 0x4d99ff : 0xff994d
            });
            controllerMesh = new THREE.Mesh(geometry, material);
            controllerMesh.name = `controller_${controllerKey}`;

            this.scene.add(controllerMesh);
            this.remoteControllers.set(controllerKey, controllerMesh);
        }


        controllerMesh.position.copy(state.position);
        controllerMesh.quaternion.copy(state.orientation);
    }


    getPerformanceMetrics() {
        return {
            fps: this.currentFPS,
            targetFPS: this.defaultConfig.targetFrameRate,
            pixelRatio: this.renderer.getPixelRatio(),
            foveationLevel: this.defaultConfig.foveatedRenderingLevel,
            handTrackingActive: this.handUpdateInterval !== null,
            controllersActive: this.xrSession?.inputSources.length || 0
        };
    }


    private stopHandTracking(): void {
        if (this.handUpdateInterval) {
            clearInterval(this.handUpdateInterval);
            this.handUpdateInterval = null;
            logger.info('Hand tracking broadcast stopped');
        }
    }


    private stopControllerBroadcast(): void {
        if (this.controllerUpdateInterval) {
            clearInterval(this.controllerUpdateInterval);
            this.controllerUpdateInterval = null;
            logger.info('Controller broadcast stopped');
        }
    }


    dispose(): void {
        logger.info('Disposing Quest3Optimizer');

        this.stopHandTracking();
        this.stopControllerBroadcast();

        if (this.xrSession && this.inputSourcesChangeHandler) {
            this.xrSession.removeEventListener('inputsourceschange', this.inputSourcesChangeHandler);
            this.inputSourcesChangeHandler = null;
        }

        this.renderer.setAnimationLoop(null);


        this.remoteHands.forEach(meshes => {
            meshes.forEach(mesh => {
                mesh.geometry.dispose();
                (mesh.material as THREE.Material).dispose();
                this.scene.remove(mesh);
            });
        });
        this.remoteHands.clear();


        this.remoteControllers.forEach(mesh => {
            mesh.geometry.dispose();
            (mesh.material as THREE.Material).dispose();
            this.scene.remove(mesh);
        });
        this.remoteControllers.clear();
    }
}
