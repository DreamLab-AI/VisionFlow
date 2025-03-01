import {
  Scene,
  PerspectiveCamera,
  // WebGLRenderer,
  Vector3,
  // Quaternion,
  // Matrix4,
  // Group,
  // Mesh,
  // MeshBasicMaterial,
  // Color,
  // Raycaster,
  // Object3D,
  // Euler,
  // MathUtils,
} from 'three';
import { createLogger, createErrorMetadata, createDataMetadata } from '../core/logger';
import { Settings } from '../types/settings/base';
import { defaultSettings } from '../state/defaultSettings';
import { XRHandWithHaptics } from '../types/xr';
import { EdgeManager } from './EdgeManager';
import { NodeManagerFacade } from './node/NodeManagerFacade';
import { graphDataManager } from '../state/graphData';
import { MetadataVisualizer } from './MetadataVisualizer';
import { GraphData } from '../core/types';
import { WebSocketService } from '../websocket/websocketService';
import { NodeMetadata } from '../types/metadata';
import { MaterialFactory } from './factories/MaterialFactory';
// import { SettingsStore } from '../state/SettingsStore'; // Commented out as it's unused
// import { debugState } from '../core/debugState'; // Commented out as it's unused
import { GraphDataManager } from '../state/graphData';
import { Node } from '../core/types';

const logger = createLogger('VisualizationController');

type VisualizationCategory = 'visualization' | 'physics' | 'rendering';
type PendingUpdate = { category: VisualizationCategory; value: any };

export class VisualizationController {
    private static instance: VisualizationController | null = null;
    private currentSettings: Settings;
    private edgeManager: EdgeManager | null = null;
    private nodeManager: NodeManagerFacade | null = null;
    private metadataVisualizer: MetadataVisualizer | null = null;
    private isInitialized: boolean = false;
    private pendingUpdates: Map<string, PendingUpdate> = new Map();
    private lastUpdateTime: number = performance.now();
    private websocketService: WebSocketService;
    private metadataVisualizationInitialized: boolean = false; 
    private lastMetadataUpdateTime: number = 0;
    private isRandomizationInProgress: boolean = false;
    private hasReceivedBinaryUpdate: boolean = false;
    private randomizationStartTime: number = 0;
    private randomizationAcknowledged: boolean = false;
    private randomizedNodeIds: Set<string> = new Set();
    private loadingIndicator: HTMLElement | null = null;

    private constructor() {
        // Initialize with complete default settings
        this.currentSettings = defaultSettings;
        this.websocketService = WebSocketService.getInstance();
        
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
                if (import.meta.env.DEV) logger.debug('Queuing updates until initialization');
            }
        });

        // Subscribe to websocket binary updates
        this.websocketService.onBinaryMessage((nodes) => {            
            if (this.nodeManager && this.isInitialized) {
                // Convert binary node data to the format expected by updateNodePositions
                this.hasReceivedBinaryUpdate = true;
                
                let updates = nodes.map(node => ({
                    id: node.id.toString(),
                    data: {
                        position: node.position,
                        velocity: node.velocity
                    }
                }));
                
                // Filter out nodes that we've just randomized if randomization is in progress
                // This prevents the server from sending back old positions during randomization
                if (this.isRandomizationInProgress) {
                    const now = performance.now();
                    // Only apply this filter for a few seconds after randomization starts
                    if (now - this.randomizationStartTime < 5000) {
                        updates = updates.filter(update => !this.randomizedNodeIds.has(update.id));
                        
                        // If we're filtering out updates, it means the server has acknowledged our data
                        if (updates.length < nodes.length) {
                            this.randomizationAcknowledged = true;
                            logger.info('Randomization acknowledged by server - filtered out ' + 
                                        (nodes.length - updates.length) + ' outdated positions');
                        }
                    } else {
                        // Time's up - end randomization mode
                        this.isRandomizationInProgress = false;
                        this.randomizedNodeIds.clear();
                        logger.info('Randomization sync period ended');
                    }
                }
                
                // Check if we haven't created metadata labels yet, and have received binary data
                // This ensures we have proper positions before creating labels
                if (this.hasReceivedBinaryUpdate && !this.metadataVisualizationInitialized && this.metadataVisualizer) {
                    logger.info('Received binary position updates. Initializing metadata visualization.');
                    this.initializeMetadataVisualization();
                }
                
                this.nodeManager.updateNodePositions(updates);
            }
        });

        // Handle loading status from WebSocket
        this.websocketService.onLoadingStatusChange((isLoading, message) => {
            if (isLoading) {
                this.showLoadingIndicator(message);
            } else {
                this.hideLoadingIndicator();
            }
        });
    }

    public initializeScene(scene: Scene, camera: PerspectiveCamera): void {
        logger.info('Initializing visualization scene');
        
        // Ensure camera can see nodes
        camera.layers.enable(0);
        logger.debug('Camera layers configured');
        
        // Enable WebSocket debugging
        this.currentSettings.system.debug.enabled = true;
        this.currentSettings.system.debug.enableWebsocketDebug = true;
        
        // Connect to websocket first
        this.websocketService.connect().then(() => {
            logger.info('WebSocket connected, enabling binary updates');
            this.metadataVisualizationInitialized = false; // Reset flag when reconnecting
            graphDataManager.enableBinaryUpdates();
            
            // Send initial request for data
            this.websocketService.sendMessage({ 
                type: 'requestInitialData',
                timestamp: Date.now()
            });

            // Show loading indicator until data arrives
            this.showLoadingIndicator("Waiting for initial graph data...");
            
            // Initially disable randomization until data loading is finished
            this.websocketService.sendMessage({
                type: 'enableRandomization',
                enabled: false
            });
        }).catch(error => {
            logger.error('Failed to connect WebSocket:', createErrorMetadata(error));
        });
        
        const materialFactory = MaterialFactory.getInstance();
        // Add debug logging
        logger.debug('Creating NodeManagerFacade');
        this.nodeManager = NodeManagerFacade.getInstance(
            scene,
            camera,
            materialFactory.getNodeMaterial(this.currentSettings)
        );
        this.edgeManager = new EdgeManager(scene, this.currentSettings, this.nodeManager.getNodeInstanceManager());
        this.metadataVisualizer = new MetadataVisualizer(camera, scene, this.currentSettings);
        this.isInitialized = true;
        
        if (import.meta.env.DEV) logger.debug('Scene managers initialized');

        // Initialize with current graph data (if any)
        const currentData = graphDataManager.getGraphData();
        if (currentData.nodes.length > 0 && this.nodeManager) {
            this.nodeManager.updateNodes(currentData.nodes);
        }

        // Start animation loop
        this.animate();

        // We'll initialize metadata visualization once we receive binary position updates
        logger.info('Scene initialization complete');
    }

    public static getInstance(): VisualizationController {
        if (!VisualizationController.instance) {
            VisualizationController.instance = new VisualizationController();
        }
        return VisualizationController.instance;
    }

    public updateSetting(path: string, value: any): void {
        // Hide loading indicator when any settings are changed
        this.hideLoadingIndicator();

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

    /**
     * Show a loading indicator while waiting for data
     * @param message Optional message to display
     */
    private showLoadingIndicator(message?: string): void {
        // Hide any existing indicator first
        this.hideLoadingIndicator();
        
        // Create loading indicator if it doesn't exist
        this.loadingIndicator = document.createElement('div');
        this.loadingIndicator.id = 'graph-loading-indicator';
        this.loadingIndicator.style.position = 'fixed';
        this.loadingIndicator.style.top = '50%';
        this.loadingIndicator.style.left = '50%';
        this.loadingIndicator.style.transform = 'translate(-50%, -50%)';
        this.loadingIndicator.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
        this.loadingIndicator.style.color = 'white';
        this.loadingIndicator.style.padding = '20px';
        this.loadingIndicator.style.borderRadius = '10px';
        this.loadingIndicator.style.zIndex = '1000';
        this.loadingIndicator.style.textAlign = 'center';
        this.loadingIndicator.style.fontFamily = 'Arial, sans-serif';
        this.loadingIndicator.style.boxShadow = '0 0 20px rgba(0, 0, 255, 0.5)';
        this.loadingIndicator.style.border = '1px solid #3a3a3a';
        
        // Add loading spinner
        const spinner = document.createElement('div');
        spinner.style.display = 'inline-block';
        spinner.style.width = '30px';
        spinner.style.height = '30px';
        spinner.style.marginBottom = '10px';
        spinner.style.border = '4px solid rgba(255, 255, 255, 0.3)';
        spinner.style.borderRadius = '50%';
        spinner.style.borderTop = '4px solid #ffffff';
        spinner.style.animation = 'spin 1s linear infinite';
        
        // Add animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        `;
        document.head.appendChild(style);
        
        this.loadingIndicator.appendChild(spinner);
        
        // Create a line break
        this.loadingIndicator.appendChild(document.createElement('br'));
        
        // Add message text
        const messageText = document.createElement('div');
        messageText.textContent = message || 'Loading graph data...';
        this.loadingIndicator.appendChild(messageText);
        
        // Add to DOM
        document.body.appendChild(this.loadingIndicator);
    }
    
    /**
     * Hide the loading indicator
     */
    private hideLoadingIndicator(): void {
        if (this.loadingIndicator && document.body.contains(this.loadingIndicator)) {
            document.body.removeChild(this.loadingIndicator);
        }
        this.loadingIndicator = null;
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
        // Update metadata visualization
        if (this.metadataVisualizer && this.metadataVisualizationInitialized) {
            // Only update positions of existing metadata, don't recreate everything
            this.updateMetadataPositions();
        }
        // Intentionally not initializing metadata visualization here anymore
        // We'll wait for binary data with positions first
        
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

    /**
     * Randomly distributes all nodes in 3D space and triggers WebSocket updates
     * @param radius The radius of the sphere within which to distribute nodes
     */
    public async randomizeNodePositions(radius: number = 7): Promise<void> {
        // First, ensure WebSocket is connected
        const connectionState = this.websocketService.getConnectionStatus();
        if (connectionState !== 'connected') {
            logger.warn(`WebSocket not connected (state: ${connectionState}), attempting to reconnect...`);
            
            try {
                // Try to reconnect
                await this.websocketService.connect();
                
                // Wait a bit to ensure connection is stable
                await new Promise(resolve => setTimeout(resolve, 1000));
                
                // Check again
                const newState = this.websocketService.getConnectionStatus();
                if (newState !== 'connected') {
                    logger.error(`Failed to establish WebSocket connection (state: ${newState})`);
                    // Proceed with local randomization only
                    logger.warn('Will perform randomization locally only without server sync');
                }
            } catch (error) {
                logger.error('Error reconnecting WebSocket:', createErrorMetadata(error));
                // Proceed with local randomization only
                logger.warn('Will perform randomization locally only without server sync');
            }
        }

        // Enforce maximum radius to prevent explosion
        radius = Math.min(radius, 5); // Reduced maximum radius to prevent explosion

        if (!this.nodeManager || !this.isInitialized) {
            logger.warn('Cannot randomize node positions - Node manager not initialized');
            return;
        }

        // Don't start another randomization if one is in progress
        if (this.isRandomizationInProgress) {
            logger.warn('Randomization already in progress, please wait...');
            return;
        }
        
        // Start tracking randomization
        this.isRandomizationInProgress = true;
        this.randomizationStartTime = performance.now();
        
        // Get node data from the graph manager
        const graphData = GraphDataManager.getInstance().getGraphData();
        if (!graphData || !graphData.nodes || graphData.nodes.length === 0) {
            logger.warn('No nodes found to randomize');
            return;
        }

        logger.info('Randomizing node positions with radius:', createDataMetadata({ 
            radius,
            nodeCount: graphData.nodes.length
        }));
        this.randomizedNodeIds.clear();
        
        // Create node updates with random positions - using original node IDs
        // WebSocketService will handle converting metadata names to numeric IDs
        const updates = graphData.nodes.map((node: Node) => {
            const nodeId = node.id; // Keep original ID - WebSocketService handles mapping to numeric

            // Generate random position within a sphere
            const theta = Math.random() * Math.PI * 2; // Random angle around Y axis
            const phi = Math.acos((Math.random() * 2) - 1); // Random angle from Y axis
            // Use square root for more central clustering
            const r = radius * Math.sqrt(Math.random()); // Random distance from center
            
            // Convert spherical to Cartesian coordinates
            const x = r * Math.sin(phi) * Math.cos(theta);
            const y = r * Math.sin(phi) * Math.sin(theta);
            const z = r * Math.cos(phi);

            // Use near-zero initial velocities to prevent explosion
            // Set zero velocity for all axes - completely remove any initial bias
            const vx = 0.0;
            const vy = 0.0;
            const vz = 0.0; // Removed bias that could cause z-axis drift
            

            // Track which nodes we've randomized
            this.randomizedNodeIds.add(nodeId);
            
            return {
                id: nodeId,
                data: {
                    position: new Vector3(x, y, z),
                    velocity: new Vector3(vx, vy, vz)
                }
            };
        });
        
        // Update local node positions
        this.nodeManager.updateNodePositions(updates);
        
        // Send updates to server via WebSocket
        if (this.websocketService) {
            // First, ensure we signal server to stop any current physics simulation
            this.websocketService.sendMessage({ 
                type: 'pauseSimulation',
                enabled: true
            });
            
            // Wait for a short delay to ensure the server has processed the pause
            await new Promise(resolve => setTimeout(resolve, 300));
            
            this.randomizationAcknowledged = false;
            
            logger.info('Sending node positions to server in batch...');
            
            // Send positions in batches to avoid overwhelming the server
            const batchSize = 5;
            const connectionState = this.websocketService.getConnectionStatus();
            
            // Only send updates if websocket is connected
            if (connectionState === 'connected') {
            for (let i = 0; i < updates.length; i += batchSize) {
                const batch = updates.slice(i, i + batchSize);
                const wsUpdates = batch.map(update => ({ 
                    id: update.id,
                    position: update.data.position,
                    velocity: update.data.velocity
                }));
                
                this.websocketService.sendNodeUpdates(wsUpdates);
            }
            }
            
            // Old approach that sent one at a time - keeping as fallback
            /*
            updates.forEach((update) => {
                // Send updates to server one at a time with minimal velocity
                this.websocketService.sendNodeUpdates([{ 
                    id: update.id,
                    position: update.data.position,
                    velocity: update.data.velocity
                }]);
            });
            */
            
            // Signal server to resume physics simulation with new positions
            setTimeout(async () => {
                logger.info('Resuming physics simulation after position updates');
                
                // Only send message if connected
                if (this.websocketService.getConnectionStatus() === 'connected') {
                this.websocketService.sendMessage({ 
                    type: 'pauseSimulation',
                    enabled: false
                });

                // Also explicitly request force calculation to begin
                this.websocketService.sendMessage({
                    type: 'applyForces',
                    timestamp: Date.now()
                });
                }
                
                // Wait up to 5 seconds for acknowledgment
                const waitForAcknowledgment = async () => {
                    for (let i = 0; i < 10; i++) { // 10 * 500ms = 5 seconds
                        if (this.randomizationAcknowledged) return true;
                        await new Promise(resolve => setTimeout(resolve, 500));
                    }
                    return false;
                };
                
                const acknowledged = await waitForAcknowledgment();
                logger.info(`Randomization ${acknowledged ? 'confirmed' : 'timed out waiting for confirmation'}`);
                
                // Mark randomization as complete even if no acknowledgment received
                this.isRandomizationInProgress = false;
                
            }, 500); // Short delay to ensure all positions are processed
        } else {
            // No WebSocket service or not connected
            this.isRandomizationInProgress = false;
            this.randomizedNodeIds.clear();
            logger.warn('WebSocket service not available, node positions only updated locally');
        }
    }

    private animate = (): void => {
        if (!this.isInitialized) return;

        requestAnimationFrame(this.animate);
        const currentTime = performance.now();
        const deltaTime = (currentTime - this.lastUpdateTime) / 1000;
        this.update(deltaTime);
    }

    public update(deltaTime: number): void {
        if (this.isInitialized) {
            const currentTime = performance.now();
            if (deltaTime === 0) {
                deltaTime = (currentTime - this.lastUpdateTime) / 1000;
            }
            this.lastUpdateTime = currentTime;
            if (this.nodeManager) {
                this.nodeManager.update(deltaTime);
            }

            // Update edge animations
            if (this.edgeManager) {
                this.edgeManager.update(deltaTime);
            }
            
            // Update ONLY metadata positions - do not recreate labels every frame
            if (this.metadataVisualizer && this.nodeManager && this.metadataVisualizationInitialized) {
                this.updateMetadataPositions(); 
            }
        }
    }

    public dispose(): void {
        // Dispose of managers and cleanup websocket
        if (this.metadataVisualizer) {
            this.metadataVisualizer?.dispose();
            this.metadataVisualizer = null;
        }
        this.nodeManager?.dispose();
        this.hideLoadingIndicator();
        this.edgeManager?.dispose();
        this.websocketService.dispose();
        this.isInitialized = false;
        VisualizationController.instance = null;
    }

    /**
     * Update only the positions of metadata labels without recreating them
     * This is called every frame for efficiency
     */
    private updateMetadataPositions(): void {
        if (!this.isInitialized || !this.metadataVisualizer || !this.nodeManager) return;
        
        const currentData = graphDataManager.getGraphData();
        // Only occasionally log updates
        if (Math.random() < 0.001) {
            logger.debug('Updating metadata positions for nodes:', createDataMetadata({
                nodeCount: currentData.nodes.length
            }));
        }
        
        // Update positions for existing metadata labels
        currentData.nodes.forEach(node => {
            const position = this.nodeManager?.getNodeInstanceManager().getNodePosition(node.id);
            if (position) {
                this.metadataVisualizer?.updateMetadataPosition(node.id, position);
            }
        });
    }

    /**
     * Initialize the metadata visualization once - called only after binary updates
     * are available so positions are correct
     */
    private initializeMetadataVisualization(): void {
        if (!this.isInitialized || !this.metadataVisualizer || !this.nodeManager) return;
        if (this.metadataVisualizationInitialized) {
            logger.debug('Metadata visualization already initialized, skipping initialization');
            return;
        }
        
        // Set flag first to prevent repeated initialization
        this.metadataVisualizationInitialized = true;
        this.lastMetadataUpdateTime = performance.now();
        
        logger.info('Initializing metadata visualization for the first time');
        
        // Perform the full visualization update (create labels)
        this.updateMetadataVisualization(true);
    }

    /**
     * Update metadata visualization, optionally clearing existing labels
     * @param clearExisting Whether to clear existing labels
     */
    private updateMetadataVisualization(clearExisting: boolean = false): void {
        if (!this.isInitialized || !this.metadataVisualizer || !this.nodeManager) return;
        
        // Debounce frequently repeated calls
        const now = performance.now();
        if (now - this.lastMetadataUpdateTime < 1000) return; // Prevent updates more than once per second
        this.lastMetadataUpdateTime = now;
        
        // Only clear existing labels if specified
        if (clearExisting) {
            logger.debug('Clearing existing metadata labels before creating new ones');
            this.metadataVisualizer.clearAllLabels();
        }
        
        
        // Store information on the nodes we'll process for logging
        const currentData = graphDataManager.getGraphData();
        logger.info('Updating metadata visualization for nodes:', createDataMetadata({
            nodeCount: currentData.nodes.length
        }));
        
        // Keep track of processed node ids to avoid duplicates
        const processedNodeIds = new Set<string>();
        
        // Create metadata for all nodes
        currentData.nodes.forEach((node, index) => {
            // Skip if already processed or no metadata available
            if (processedNodeIds.has(node.id)) {
                return;
            }
            
            // Mark as processed
            processedNodeIds.add(node.id);
            
            // Create metadata even if node.data.metadata is missing
            // Use node ID as a fallback for the name
            const nodeMetadata = node.data?.metadata || {};
            
            // Log the actual data we're working with for debugging
            if (index < 5) {
                logger.info(`Processing node #${index}: ${node.id}`, createDataMetadata({
                    name: nodeMetadata.name || node.id,
                    fileSize: nodeMetadata.fileSize,
                    position: node.data.position
                })); 
            }
                
            const metadata: NodeMetadata = {
                id: node.id,
                // Use explicit || chaining to handle all possible undefined cases
                name: (nodeMetadata.name || node.id || `Node ${index}`).toString(),
                commitAge: Math.floor((Date.now() - (nodeMetadata.lastModified || Date.now())) / (1000 * 60 * 60 * 24)),
                hyperlinkCount: nodeMetadata.hyperlinkCount || 0,
                fileSize: nodeMetadata.fileSize || 1024,
                nodeSize: Math.min(50, Math.max(1, Math.log10((nodeMetadata.fileSize || 1024) / 1024) * 10)), // Scale based on file size (1-50)
                importance: 1.0, // Default importance
                position: {
                    x: node.data.position.x || 0,
                    y: node.data.position.y || 0,
                    z: node.data.position.z || 0
                }
            };
                
            // Create the label with proper unique metadata
            this.metadataVisualizer?.createMetadataLabel(metadata, node.id);
                
            // Update position immediately to avoid the "dropping in" effect
            const position = this.nodeManager?.getNodeInstanceManager().getNodePosition(node.id);
            if (position) {
                this.metadataVisualizer?.updateMetadataPosition(node.id, position);
            }
        });
        
        logger.info(`Metadata visualization updated: ${processedNodeIds.size} nodes with unique labels. Sample labels: ${
            currentData.nodes.slice(0, 3).map(n => n.data?.metadata?.name || n.id).join(", ")
        }`);
    }
}
