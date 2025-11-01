

import { createLogger } from '../../utils/loggerConfig';

const logger = createLogger('FeatureFlags');

export interface FeatureFlagConfig {
    
    vircadiaEnabled: boolean;
    multiUserEnabled: boolean;
    spatialAudioEnabled: boolean;

    
    handTrackingEnabled: boolean;
    collaborativeGraphEnabled: boolean;
    annotationsEnabled: boolean;

    
    deltaCompressionEnabled: boolean;
    instancedRenderingEnabled: boolean;
    dynamicResolutionEnabled: boolean;
    foveatedRenderingEnabled: boolean;

    
    rolloutPercentage: number; 
    allowedUserIds?: string[];
    allowedAgentIds?: string[];
}

export class FeatureFlags {
    private static instance: FeatureFlags;
    private config: FeatureFlagConfig;
    private userHash: number = 0;

    private defaultConfig: FeatureFlagConfig = {
        
        vircadiaEnabled: import.meta.env.VITE_VIRCADIA_ENABLED === 'true',
        multiUserEnabled: import.meta.env.VITE_VIRCADIA_ENABLE_MULTI_USER === 'true',
        spatialAudioEnabled: import.meta.env.VITE_VIRCADIA_ENABLE_SPATIAL_AUDIO === 'true',

        
        handTrackingEnabled: import.meta.env.VITE_QUEST3_ENABLE_HAND_TRACKING === 'true',
        collaborativeGraphEnabled: true,
        annotationsEnabled: true,

        
        deltaCompressionEnabled: true,
        instancedRenderingEnabled: import.meta.env.VITE_BABYLON_INSTANCED_RENDERING === 'true',
        dynamicResolutionEnabled: true,
        foveatedRenderingEnabled: true,

        
        rolloutPercentage: 100, 
        allowedUserIds: [],
        allowedAgentIds: []
    };

    private constructor() {
        this.config = { ...this.defaultConfig };
        this.loadFromStorage();
        logger.info('Feature flags initialized', this.config);
    }

    
    static getInstance(): FeatureFlags {
        if (!FeatureFlags.instance) {
            FeatureFlags.instance = new FeatureFlags();
        }
        return FeatureFlags.instance;
    }

    
    private loadFromStorage(): void {
        try {
            const stored = localStorage.getItem('vircadia_feature_flags');
            if (stored) {
                const parsed = JSON.parse(stored);
                this.config = { ...this.config, ...parsed };
                logger.info('Feature flags loaded from storage');
            }
        } catch (error) {
            logger.warn('Failed to load feature flags from storage:', error);
        }
    }

    
    private saveToStorage(): void {
        try {
            localStorage.setItem('vircadia_feature_flags', JSON.stringify(this.config));
            logger.info('Feature flags saved to storage');
        } catch (error) {
            logger.warn('Failed to save feature flags to storage:', error);
        }
    }

    
    updateConfig(updates: Partial<FeatureFlagConfig>): void {
        this.config = { ...this.config, ...updates };
        this.saveToStorage();
        logger.info('Feature flags updated', updates);
    }

    
    getConfig(): FeatureFlagConfig {
        return { ...this.config };
    }

    
    isVircadiaEnabled(userId?: string, agentId?: string): boolean {
        if (!this.config.vircadiaEnabled) {
            return false;
        }

        
        if (userId && this.config.allowedUserIds && this.config.allowedUserIds.length > 0) {
            if (!this.config.allowedUserIds.includes(userId)) {
                return false;
            }
        }

        if (agentId && this.config.allowedAgentIds && this.config.allowedAgentIds.length > 0) {
            if (!this.config.allowedAgentIds.includes(agentId)) {
                return false;
            }
        }

        
        if (this.config.rolloutPercentage < 100) {
            const hash = this.hashString(userId || agentId || '');
            const userPercentage = hash % 100;
            return userPercentage < this.config.rolloutPercentage;
        }

        return true;
    }

    
    isMultiUserEnabled(): boolean {
        return this.config.vircadiaEnabled && this.config.multiUserEnabled;
    }

    
    isSpatialAudioEnabled(): boolean {
        return this.config.vircadiaEnabled && this.config.spatialAudioEnabled;
    }

    
    isHandTrackingEnabled(): boolean {
        return this.config.vircadiaEnabled && this.config.handTrackingEnabled;
    }

    
    isCollaborativeGraphEnabled(): boolean {
        return this.config.vircadiaEnabled && this.config.collaborativeGraphEnabled;
    }

    
    isAnnotationsEnabled(): boolean {
        return this.config.vircadiaEnabled && this.config.annotationsEnabled;
    }

    
    isDeltaCompressionEnabled(): boolean {
        return this.config.vircadiaEnabled && this.config.deltaCompressionEnabled;
    }

    
    isInstancedRenderingEnabled(): boolean {
        return this.config.vircadiaEnabled && this.config.instancedRenderingEnabled;
    }

    
    isDynamicResolutionEnabled(): boolean {
        return this.config.vircadiaEnabled && this.config.dynamicResolutionEnabled;
    }

    
    isFoveatedRenderingEnabled(): boolean {
        return this.config.vircadiaEnabled && this.config.foveatedRenderingEnabled;
    }

    
    private hashString(str: string): number {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; 
        }
        return Math.abs(hash);
    }

    
    enableForUsers(userIds: string[]): void {
        this.config.allowedUserIds = userIds;
        this.saveToStorage();
        logger.info(`Vircadia enabled for ${userIds.length} users`);
    }

    
    enableForAgents(agentIds: string[]): void {
        this.config.allowedAgentIds = agentIds;
        this.saveToStorage();
        logger.info(`Vircadia enabled for ${agentIds.length} agents`);
    }

    
    setRolloutPercentage(percentage: number): void {
        this.config.rolloutPercentage = Math.max(0, Math.min(100, percentage));
        this.saveToStorage();
        logger.info(`Rollout percentage set to ${this.config.rolloutPercentage}%`);
    }

    
    enableAll(): void {
        this.config = {
            vircadiaEnabled: true,
            multiUserEnabled: true,
            spatialAudioEnabled: true,
            handTrackingEnabled: true,
            collaborativeGraphEnabled: true,
            annotationsEnabled: true,
            deltaCompressionEnabled: true,
            instancedRenderingEnabled: true,
            dynamicResolutionEnabled: true,
            foveatedRenderingEnabled: true,
            rolloutPercentage: 100,
            allowedUserIds: [],
            allowedAgentIds: []
        };
        this.saveToStorage();
        logger.info('All features enabled');
    }

    
    disableAll(): void {
        this.config = {
            vircadiaEnabled: false,
            multiUserEnabled: false,
            spatialAudioEnabled: false,
            handTrackingEnabled: false,
            collaborativeGraphEnabled: false,
            annotationsEnabled: false,
            deltaCompressionEnabled: false,
            instancedRenderingEnabled: false,
            dynamicResolutionEnabled: false,
            foveatedRenderingEnabled: false,
            rolloutPercentage: 0,
            allowedUserIds: [],
            allowedAgentIds: []
        };
        this.saveToStorage();
        logger.info('All features disabled');
    }

    
    reset(): void {
        this.config = { ...this.defaultConfig };
        this.saveToStorage();
        logger.info('Feature flags reset to defaults');
    }

    
    getStatusReport(): string {
        const lines = [
            '=== Vircadia Feature Flags ===',
            `Vircadia Enabled: ${this.config.vircadiaEnabled}`,
            `Multi-User: ${this.config.multiUserEnabled}`,
            `Spatial Audio: ${this.config.spatialAudioEnabled}`,
            `Hand Tracking: ${this.config.handTrackingEnabled}`,
            `Collaborative Graph: ${this.config.collaborativeGraphEnabled}`,
            `Annotations: ${this.config.annotationsEnabled}`,
            `Delta Compression: ${this.config.deltaCompressionEnabled}`,
            `Instanced Rendering: ${this.config.instancedRenderingEnabled}`,
            `Dynamic Resolution: ${this.config.dynamicResolutionEnabled}`,
            `Foveated Rendering: ${this.config.foveatedRenderingEnabled}`,
            `Rollout Percentage: ${this.config.rolloutPercentage}%`,
            `Allowed Users: ${this.config.allowedUserIds?.length || 0}`,
            `Allowed Agents: ${this.config.allowedAgentIds?.length || 0}`,
            '============================='
        ];

        return lines.join('\n');
    }
}

// Export singleton instance
export const featureFlags = FeatureFlags.getInstance();
