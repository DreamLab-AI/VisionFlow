

import * as BABYLON from '@babylonjs/core';
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
    position: BABYLON.Vector3;
    orientation: BABYLON.Quaternion;
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
    position: BABYLON.Vector3;
    orientation: BABYLON.Quaternion;
    buttons: Record<string, boolean>;
    axes: Record<string, number>;
    timestamp: number;
}

export class Quest3Optimizer {
    private xrHelper: BABYLON.WebXRDefaultExperience | null = null;
    private handFeature: any | null = null;
    private localAgentId: string | null = null;
    private handUpdateInterval: ReturnType<typeof setInterval> | null = null;
    private controllerUpdateInterval: ReturnType<typeof setInterval> | null = null;
    private remoteHands = new Map<string, BABYLON.Mesh[]>();
    private remoteControllers = new Map<string, BABYLON.Mesh>();
    private performanceMonitor: BABYLON.PerformanceMonitor | null = null;
    private currentFPS = 0;

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
        private scene: BABYLON.Scene,
        private client: ClientCore,
        config?: Partial<Quest3Config>
    ) {
        this.defaultConfig = { ...this.defaultConfig, ...config };
    }

    
    async initialize(xrHelper: BABYLON.WebXRDefaultExperience): Promise<void> {
        logger.info('Initializing Quest 3 optimizations...');

        this.xrHelper = xrHelper;

        
        const info = this.client.Utilities.Connection.getConnectionInfo();
        if (info.agentId) {
            this.localAgentId = info.agentId;
        }

        
        this.setupPerformanceMonitoring();

        
        this.setupFoveatedRendering();

        
        if (this.defaultConfig.dynamicResolutionScale) {
            this.setupDynamicResolution();
        }

        
        if (this.defaultConfig.enableHandTracking) {
            await this.setupHandTracking();
        }

        
        if (this.defaultConfig.enableControllers) {
            this.setupControllers();
        }

        logger.info('Quest 3 optimizations initialized');
    }

    
    private setupPerformanceMonitoring(): void {
        this.performanceMonitor = new BABYLON.PerformanceMonitor();

        this.scene.onBeforeRenderObservable.add(() => {
            if (this.performanceMonitor) {
                this.performanceMonitor.sampleFrame();
                this.currentFPS = this.performanceMonitor.averageFPS;
            }
        });

        logger.info('Performance monitoring enabled');
    }

    
    private setupFoveatedRendering(): void {
        if (!this.xrHelper) return;

        const session = this.xrHelper.baseExperience.sessionManager.session;
        if (!session) return;

        try {
            
            const glLayer = session.renderState.baseLayer as any;
            if (glLayer && 'fixedFoveation' in glLayer) {
                glLayer.fixedFoveation = this.defaultConfig.foveatedRenderingLevel;
                logger.info(`Foveated rendering enabled: level ${this.defaultConfig.foveatedRenderingLevel}`);
            }
        } catch (error) {
            logger.warn('Failed to enable foveated rendering:', error);
        }
    }

    
    private setupDynamicResolution(): void {
        const engine = this.scene.getEngine();
        const { targetFrameRate, minResolutionScale, maxResolutionScale } = this.defaultConfig;

        this.scene.onBeforeRenderObservable.add(() => {
            const fps = this.currentFPS;

            if (fps < targetFrameRate - 10) {
                
                const currentScale = engine.getHardwareScalingLevel();
                const newScale = Math.min(currentScale + 0.1, 1 / minResolutionScale);
                engine.setHardwareScalingLevel(newScale);
                logger.debug(`Resolution scaled down: ${newScale.toFixed(2)}`);

            } else if (fps > targetFrameRate + 5) {
                
                const currentScale = engine.getHardwareScalingLevel();
                const newScale = Math.max(currentScale - 0.05, 1 / maxResolutionScale);
                engine.setHardwareScalingLevel(newScale);
                logger.debug(`Resolution scaled up: ${newScale.toFixed(2)}`);
            }
        });

        logger.info('Dynamic resolution scaling enabled');
    }

    
    private async setupHandTracking(): Promise<void> {
        if (!this.xrHelper) return;

        try {
            this.handFeature = this.xrHelper.baseExperience.featuresManager.enableFeature(
                BABYLON.WebXRFeatureName.HAND_TRACKING,
                'latest',
                { xrInput: this.xrHelper.input }
            );

            
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
            if (!this.handFeature || !this.localAgentId) {
                return;
            }

            const hands = this.handFeature.getHandByControllerId('left') || this.handFeature.getHandByControllerId('right');
            if (!hands) {
                return;
            }

            
            const leftHand = this.handFeature.getHandByControllerId('left');
            if (leftHand) {
                await this.broadcastHandData('left', leftHand);
            }

            
            const rightHand = this.handFeature.getHandByControllerId('right');
            if (rightHand) {
                await this.broadcastHandData('right', rightHand);
            }

        }, 50); 

        logger.info('Hand tracking broadcast started');
    }

    
    private async broadcastHandData(hand: 'left' | 'right', handData: any): Promise<void> {
        if (!this.localAgentId) return;

        try {
            const joints: HandJoint[] = [];

            
            Object.keys(handData.trackedMeshes || {}).forEach((jointName: string) => {
                const mesh = handData.trackedMeshes[jointName];
                if (mesh) {
                    joints.push({
                        name: jointName,
                        position: mesh.position.clone(),
                        orientation: mesh.rotationQuaternion?.clone() || BABYLON.Quaternion.Identity()
                    });
                }
            });

            const query = `
                UPDATE entity.entities
                SET meta__data = jsonb_set(
                    meta__data,
                    '{handTracking,${hand}}',
                    '${JSON.stringify({
                        joints,
                        timestamp: Date.now()
                    })}'::jsonb
                )
                WHERE general__entity_name = 'avatar_${this.localAgentId}'
            `;

            await this.client.Utilities.Connection.query({ query, timeoutMs: 1000 });

        } catch (error) {
            logger.debug('Failed to broadcast hand data:', error);
        }
    }

    
    updateRemoteHandTracking(agentId: string, hand: 'left' | 'right', joints: HandJoint[]): void {
        const handKey = `${agentId}_${hand}`;
        let handMeshes = this.remoteHands.get(handKey);

        if (!handMeshes) {
            
            handMeshes = joints.map((joint, index) => {
                const sphere = BABYLON.MeshBuilder.CreateSphere(
                    `hand_${handKey}_${index}`,
                    { diameter: 0.015 },
                    this.scene
                );

                const material = new BABYLON.StandardMaterial(`hand_mat_${handKey}_${index}`, this.scene);
                material.emissiveColor = hand === 'left' ? new BABYLON.Color3(0.2, 0.5, 1.0) : new BABYLON.Color3(1.0, 0.5, 0.2);
                sphere.material = material;

                return sphere;
            });

            this.remoteHands.set(handKey, handMeshes);
        }

        
        joints.forEach((joint, index) => {
            if (handMeshes![index]) {
                handMeshes![index].position = joint.position;
                handMeshes![index].rotationQuaternion = joint.orientation;
            }
        });
    }

    
    private setupControllers(): void {
        if (!this.xrHelper) return;

        this.xrHelper.input.onControllerAddedObservable.add((controller) => {
            logger.info(`Controller added: ${controller.uniqueId}`);

            
            this.startControllerBroadcast();
        });

        logger.info('Controller support enabled');
    }

    
    private startControllerBroadcast(): void {
        if (this.controllerUpdateInterval) {
            return;
        }

        this.controllerUpdateInterval = setInterval(async () => {
            if (!this.xrHelper || !this.localAgentId) {
                return;
            }

            this.xrHelper.input.controllers.forEach(async (controller) => {
                const controllerState: ControllerState = {
                    agentId: this.localAgentId!,
                    controllerId: controller.inputSource.handedness as 'left' | 'right',
                    position: controller.pointer.position.clone(),
                    orientation: controller.pointer.rotationQuaternion?.clone() || BABYLON.Quaternion.Identity(),
                    buttons: {},
                    axes: {},
                    timestamp: Date.now()
                };

                
                controller.motionController?.components.forEach((component, name) => {
                    if (component.type === BABYLON.WebXRControllerComponent.BUTTON_TYPE) {
                        controllerState.buttons[name] = component.pressed;
                    } else if (component.type === BABYLON.WebXRControllerComponent.THUMBSTICK_TYPE) {
                        controllerState.axes[name] = component.axes.x;
                        controllerState.axes[`${name}_y`] = component.axes.y;
                    }
                });

                await this.broadcastControllerState(controllerState);
            });

        }, 50); 

        logger.info('Controller broadcast started');
    }

    
    private async broadcastControllerState(state: ControllerState): Promise<void> {
        try {
            const query = `
                UPDATE entity.entities
                SET meta__data = jsonb_set(
                    meta__data,
                    '{controller,${state.controllerId}}',
                    '${JSON.stringify({
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
                    })}'::jsonb
                )
                WHERE general__entity_name = 'avatar_${this.localAgentId}'
            `;

            await this.client.Utilities.Connection.query({ query, timeoutMs: 1000 });

        } catch (error) {
            logger.debug('Failed to broadcast controller state:', error);
        }
    }

    
    updateRemoteController(agentId: string, state: ControllerState): void {
        const controllerKey = `${agentId}_${state.controllerId}`;
        let controllerMesh = this.remoteControllers.get(controllerKey);

        if (!controllerMesh) {
            
            controllerMesh = BABYLON.MeshBuilder.CreateCylinder(
                `controller_${controllerKey}`,
                { height: 0.15, diameter: 0.03 },
                this.scene
            );

            const material = new BABYLON.StandardMaterial(`controller_mat_${controllerKey}`, this.scene);
            material.emissiveColor = state.controllerId === 'left' ? new BABYLON.Color3(0.3, 0.6, 1.0) : new BABYLON.Color3(1.0, 0.6, 0.3);
            controllerMesh.material = material;

            this.remoteControllers.set(controllerKey, controllerMesh);
        }

        
        controllerMesh.position = state.position;
        controllerMesh.rotationQuaternion = state.orientation;
    }

    
    getPerformanceMetrics() {
        return {
            fps: this.currentFPS,
            targetFPS: this.defaultConfig.targetFrameRate,
            hardwareScaling: this.scene.getEngine().getHardwareScalingLevel(),
            foveationLevel: this.defaultConfig.foveatedRenderingLevel,
            handTrackingActive: this.handFeature !== null,
            controllersActive: this.xrHelper?.input.controllers.length || 0
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

        
        this.remoteHands.forEach(meshes => {
            meshes.forEach(mesh => mesh.dispose());
        });
        this.remoteHands.clear();

        
        this.remoteControllers.forEach(mesh => mesh.dispose());
        this.remoteControllers.clear();

        this.performanceMonitor = null;
    }
}
