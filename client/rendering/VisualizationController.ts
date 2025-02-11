import { Scene, Camera } from 'three';
import { createLogger } from '../core/logger';
import { Settings } from '../types/settings/base';
import { defaultSettings } from '../state/defaultSettings';
import { XRHandWithHaptics } from '../types/xr';
import { EdgeManager } from './EdgeManager';
import { EnhancedNodeManager } from './EnhancedNodeManager';
import { graphDataManager } from '../state/graphData';
import { TextRenderer } from './textRenderer';
import { GraphData } from '../core/types';

const logger = createLogger('VisualizationController');

type VisualizationCategory = 'visualization' | 'physics' | 'rendering';
type PendingUpdate = { category: VisualizationCategory; value: any };

export class VisualizationController {
    private static instance: VisualizationController | null = null;
    private currentSettings: Settings;
    private edgeManager: EdgeManager | null = null;
    private nodeManager: EnhancedNodeManager | null = null;
    private textRenderer: TextRenderer | null = null;
    private isInitialized: boolean = false;
    private pendingUpdates: Map<string, PendingUpdate> = new Map();
    private pendingEdgeUpdates: GraphData['edges'] | null = null;
    private pendingNodeUpdates: GraphData['nodes'] | null = null;

    private constructor() {
        // Initialize with complete default settings
        this.currentSettings = { ...defaultSettings };
        
        // Subscribe to graph data updates
        graphDataManager.subscribe((data: GraphData) => {
            if (this.isInitialized) {
                if (this.nodeManager) {
                    this.nodeManager.updateNodes(data.nodes);
                }
                if (this.edgeManager) {
                    this.edgeManager.updateEdges(data.edges);
                }
            } else {
                // Queue updates until initialized
                this.pendingNodeUpdates = data.nodes;
                this.pendingEdgeUpdates = data.edges;
                logger.debug('Queuing updates until initialization');
            }
        });
    }

    public initializeScene(scene: Scene, camera: Camera): void {
        logger.info('Initializing visualization scene');
        
        // Initialize managers with scene and settings
        this.edgeManager = new EdgeManager(scene, this.currentSettings);
        this.nodeManager = new EnhancedNodeManager(scene, this.currentSettings);
        this.textRenderer = new TextRenderer(camera, scene);
        this.isInitialized = true;
        
        // Apply any pending updates
        this.applyPendingUpdates();
        
        // Initialize with current graph data
        const currentData = graphDataManager.getGraphData();
        
        // Handle pending node updates
        if (this.pendingNodeUpdates && this.nodeManager) {
            this.nodeManager.updateNodes(this.pendingNodeUpdates);
            this.pendingNodeUpdates = null;
        } else if (currentData.nodes.length > 0 && this.nodeManager) {
            this.nodeManager.updateNodes(currentData.nodes);
        }
        
        // Handle pending edge updates
        if (this.pendingEdgeUpdates && this.edgeManager) {
            this.edgeManager.updateEdges(this.pendingEdgeUpdates);
            this.pendingEdgeUpdates = null;
        } else if (currentData.edges.length > 0 && this.edgeManager) {
            this.edgeManager.updateEdges(currentData.edges);
        }
        
        logger.info('Visualization scene initialized');
    }

    private applyPendingUpdates(): void {
        if (!this.isInitialized) {
            logger.debug('Cannot apply pending updates - not initialized');
            return;
        }

        logger.debug(`Applying ${this.pendingUpdates.size} pending updates`);
        this.pendingUpdates.forEach((update, path) => {
            let current = this.currentSettings as any;
            const parts = path.split('.');
            
            // Update the settings object
            for (let i = 0; i < parts.length - 1; i++) {
                const part = parts[i];
                if (!(part in current)) {
                    current[part] = {};
                }
                current = current[part];
            }
            current[parts[parts.length - 1]] = update.value;
            
            // Apply the update
            this.applySettingUpdate(update.category);
        });
        
        this.pendingUpdates.clear();
    }

    public static getInstance(): VisualizationController {
        if (!VisualizationController.instance) {
            VisualizationController.instance = new VisualizationController();
        }
        return VisualizationController.instance;
    }

    public updateSetting(path: string, value: any): void {
        const parts = path.split('.');
        const category = parts[0] as VisualizationCategory;
        
        if (!['visualization', 'physics', 'rendering'].includes(category)) {
            return;
        }

        if (!this.isInitialized) {
            logger.debug(`Queuing setting update for ${path}`);
            this.pendingUpdates.set(path, { category, value });
            return;
        }

        let current = this.currentSettings as any;
        for (let i = 0; i < parts.length - 1; i++) {
            const part = parts[i];
            if (!(part in current)) {
                current[part] = {};
            }
            current = current[part];
        }

        current[parts[parts.length - 1]] = value;
        this.applySettingUpdate(category);
    }

    public updateSettings(category: VisualizationCategory, settings: Partial<Settings>): void {
        if (!this.isInitialized) {
            logger.debug(`Queuing bulk settings update for ${category}`);
            this.pendingUpdates.set(category, { category, value: settings });
            return;
        }

        switch (category) {
            case 'visualization':
                if (settings.visualization) {
                    this.currentSettings.visualization = {
                        ...this.currentSettings.visualization,
                        ...settings.visualization
                    };
                    this.applyVisualizationUpdates();
                }
                break;
            case 'physics':
                if (settings.visualization?.physics) {
                    this.currentSettings.visualization.physics = {
                        ...this.currentSettings.visualization.physics,
                        ...settings.visualization.physics
                    };
                    this.updatePhysicsSimulation();
                }
                break;
            case 'rendering':
                if (settings.visualization?.rendering) {
                    this.currentSettings.visualization.rendering = {
                        ...this.currentSettings.visualization.rendering,
                        ...settings.visualization.rendering
                    };
                    this.updateRenderingQuality();
                }
                break;
        }
    }

    public getSettings(category: VisualizationCategory): Partial<Settings> {
        const baseVisualization = {
            nodes: { ...this.currentSettings.visualization.nodes },
            edges: { ...this.currentSettings.visualization.edges },
            physics: { ...this.currentSettings.visualization.physics },
            rendering: { ...this.currentSettings.visualization.rendering },
            animations: { ...this.currentSettings.visualization.animations },
            labels: { ...this.currentSettings.visualization.labels },
            bloom: { ...this.currentSettings.visualization.bloom },
            hologram: { ...this.currentSettings.visualization.hologram }
        };

        switch (category) {
            case 'visualization':
                return {
                    visualization: { ...this.currentSettings.visualization }
                };
            case 'physics':
                return {
                    visualization: {
                        ...baseVisualization,
                        physics: { ...this.currentSettings.visualization.physics }
                    }
                };
            case 'rendering':
                return {
                    visualization: {
                        ...baseVisualization,
                        rendering: { ...this.currentSettings.visualization.rendering }
                    }
                };
            default:
                return {
                    visualization: baseVisualization
                };
        }
    }

    public handleHandInput(hand: XRHandWithHaptics): void {
        if (!this.isInitialized || !hand) return;

        const pinchStrength = hand.pinchStrength || 0;
        const gripStrength = hand.gripStrength || 0;

        if (pinchStrength > (this.currentSettings.xr.pinchThreshold || 0.5)) {
            logger.debug('Pinch gesture detected', { strength: pinchStrength });
        }

        if (gripStrength > (this.currentSettings.xr.dragThreshold || 0.5)) {
            logger.debug('Grip gesture detected', { strength: gripStrength });
        }

        if (hand.hand?.joints) {
            logger.debug('Processing hand joints');
        }
    }

    private applySettingUpdate(category: VisualizationCategory): void {
        if (!this.isInitialized) {
            logger.debug(`Queuing category update for ${category}`);
            return;
        }

        logger.debug(`Updating ${category} settings`);
        
        switch (category) {
            case 'visualization':
                this.applyVisualizationUpdates();
                break;
            case 'physics':
                this.updatePhysicsSimulation();
                break;
            case 'rendering':
                this.updateRenderingQuality();
                break;
        }
    }

    private applyVisualizationUpdates(): void {
        if (!this.isInitialized) return;
        this.updateNodeAppearance();
        this.updateEdgeAppearance();
        // Update text labels
        if (this.textRenderer) {
            this.textRenderer.update();
        }
    }

    private updateNodeAppearance(): void {
        if (!this.isInitialized) return;
        logger.debug('Updating node appearance');
        if (this.nodeManager) {
            this.nodeManager.handleSettingsUpdate(this.currentSettings);
        }
    }

    private updateEdgeAppearance(): void {
        if (!this.isInitialized) {
            logger.debug('Queuing edge appearance update');
            return;
        }

        if (this.edgeManager) {
            this.edgeManager.handleSettingsUpdate(this.currentSettings);
            logger.debug('Edge appearance updated');
        } else {
            logger.warn('EdgeManager not initialized');
        }
    }

    private updatePhysicsSimulation(): void {
        if (!this.isInitialized) return;
        logger.debug('Updating physics simulation');
    }

    private updateRenderingQuality(): void {
        if (!this.isInitialized) return;
        logger.debug('Updating rendering quality');
    }

    public updateNodePositions(nodes: any[]): void {
        if (this.nodeManager) {
            this.nodeManager.updateNodePositions(nodes);
        }
    }

    public update(): void {
        if (this.isInitialized) {
            // Update node animations and state
            if (this.nodeManager) {
                this.nodeManager.update(1/60); // Standard 60fps delta time
            }
            
            // Update text labels
            if (this.textRenderer) {
                this.textRenderer.update();
            }
        }
    }

    public dispose(): void {
        if (this.nodeManager) {
            this.nodeManager.dispose();
            this.nodeManager = null;
        }
        if (this.edgeManager) {
            this.edgeManager.dispose();
            this.edgeManager = null;
        }
        this.textRenderer?.dispose();
        this.currentSettings = { ...defaultSettings };
        this.isInitialized = false;
        this.pendingUpdates.clear();
        this.pendingEdgeUpdates = null;
        this.pendingNodeUpdates = null;
        VisualizationController.instance = null;
    }
}
