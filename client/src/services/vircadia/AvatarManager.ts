/**
 * AvatarManager - Multi-user avatar system for Vircadia XR
 *
 * Manages local and remote user avatars in Quest 3 AR environment,
 * including avatar models, nameplates, and position synchronization.
 */

import * as BABYLON from '@babylonjs/core';
import '@babylonjs/loaders/glTF';
import { ClientCore } from './VircadiaClientCore';
import { createLogger } from '../../utils/loggerConfig';

const logger = createLogger('AvatarManager');

export interface AvatarConfig {
    modelUrl: string;
    scale: number;
    showNameplate: boolean;
    nameplateDistance: number;
    enableAnimations: boolean;
}

export interface UserAvatar {
    agentId: string;
    username: string;
    position: BABYLON.Vector3;
    rotation: BABYLON.Quaternion;
    mesh?: BABYLON.AbstractMesh;
    nameplate?: BABYLON.Mesh;
    animationGroups?: BABYLON.AnimationGroup[];
}

export class AvatarManager {
    private avatars = new Map<string, UserAvatar>();
    private localAgentId: string | null = null;
    private updateInterval: ReturnType<typeof setInterval> | null = null;

    private defaultConfig: AvatarConfig = {
        modelUrl: '/assets/avatars/default-avatar.glb',
        scale: 1.0,
        showNameplate: true,
        nameplateDistance: 10.0,
        enableAnimations: true
    };

    constructor(
        private scene: BABYLON.Scene,
        private client: ClientCore,
        private camera: BABYLON.Camera,
        config?: Partial<AvatarConfig>
    ) {
        this.defaultConfig = { ...this.defaultConfig, ...config };
        this.setupConnectionListeners();
    }

    /**
     * Set up connection event listeners
     */
    private setupConnectionListeners(): void {
        // Listen for connection to get our agentId
        this.client.Utilities.Connection.addEventListener('statusChange', () => {
            const info = this.client.Utilities.Connection.getConnectionInfo();
            if (info.isConnected && info.agentId) {
                this.localAgentId = info.agentId;
                logger.info(`Local agent ID: ${this.localAgentId}`);
            }
        });

        // Listen for remote avatar updates
        this.client.Utilities.Connection.addEventListener('syncUpdate', async () => {
            await this.fetchRemoteAvatars();
        });
    }

    /**
     * Create local user avatar
     */
    async createLocalAvatar(username: string): Promise<void> {
        if (!this.localAgentId) {
            logger.warn('Cannot create local avatar: no agent ID');
            return;
        }

        logger.info(`Creating local avatar for ${username}`);

        const avatar: UserAvatar = {
            agentId: this.localAgentId,
            username,
            position: this.camera.position.clone(),
            rotation: BABYLON.Quaternion.RotationYawPitchRoll(0, 0, 0)
        };

        // Note: Local avatar mesh is typically not shown in first-person VR
        // But we track it for other users to see

        this.avatars.set(this.localAgentId, avatar);

        // Start broadcasting position
        this.startPositionBroadcast();

        // Create avatar entity in Vircadia
        await this.syncAvatarToVircadia(avatar);
    }

    /**
     * Load remote user avatar
     */
    async loadRemoteAvatar(agentId: string, username: string): Promise<void> {
        if (this.avatars.has(agentId)) {
            return; // Already loaded
        }

        logger.info(`Loading remote avatar for ${username} (${agentId})`);

        try {
            // Load avatar model
            const result = await BABYLON.SceneLoader.ImportMeshAsync(
                '',
                '',
                this.defaultConfig.modelUrl,
                this.scene
            );

            const rootMesh = result.meshes[0];
            rootMesh.name = `avatar_${agentId}`;
            rootMesh.scaling = new BABYLON.Vector3(
                this.defaultConfig.scale,
                this.defaultConfig.scale,
                this.defaultConfig.scale
            );

            // Create nameplate
            let nameplate: BABYLON.Mesh | undefined;
            if (this.defaultConfig.showNameplate) {
                nameplate = this.createNameplate(rootMesh, username);
            }

            const avatar: UserAvatar = {
                agentId,
                username,
                position: BABYLON.Vector3.Zero(),
                rotation: BABYLON.Quaternion.Identity(),
                mesh: rootMesh,
                nameplate,
                animationGroups: result.animationGroups
            };

            this.avatars.set(agentId, avatar);
            logger.info(`Remote avatar loaded: ${username}`);

            // Play idle animation if available
            if (this.defaultConfig.enableAnimations && result.animationGroups.length > 0) {
                const idleAnim = result.animationGroups.find(a => a.name.toLowerCase().includes('idle'));
                if (idleAnim) {
                    idleAnim.start(true);
                }
            }

        } catch (error) {
            logger.error(`Failed to load avatar for ${username}:`, error);
        }
    }

    /**
     * Create 3D nameplate above avatar
     */
    private createNameplate(parentMesh: BABYLON.AbstractMesh, username: string): BABYLON.Mesh {
        const plane = BABYLON.MeshBuilder.CreatePlane(
            `${parentMesh.name}_nameplate`,
            { width: 1, height: 0.3 },
            this.scene
        );

        plane.parent = parentMesh;
        plane.position.y = 2.2; // Above avatar head
        plane.billboardMode = BABYLON.Mesh.BILLBOARDMODE_ALL;

        // Create texture with username
        const dynamicTexture = new BABYLON.DynamicTexture(
            `${parentMesh.name}_nameplate_texture`,
            { width: 512, height: 128 },
            this.scene
        );

        const ctx = dynamicTexture.getContext();
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(0, 0, 512, 128);
        ctx.fillStyle = 'white';
        ctx.font = 'bold 56px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(username, 256, 80);
        dynamicTexture.update();

        const material = new BABYLON.StandardMaterial(`${parentMesh.name}_nameplate_mat`, this.scene);
        material.diffuseTexture = dynamicTexture;
        material.emissiveTexture = dynamicTexture;
        material.opacityTexture = dynamicTexture;
        material.backFaceCulling = false;

        plane.material = material;

        return plane;
    }

    /**
     * Update remote avatar position
     */
    updateAvatarPosition(agentId: string, position: BABYLON.Vector3, rotation?: BABYLON.Quaternion): void {
        const avatar = this.avatars.get(agentId);
        if (!avatar) {
            logger.warn(`Cannot update avatar: ${agentId} not found`);
            return;
        }

        avatar.position = position;
        if (rotation) {
            avatar.rotation = rotation;
        }

        // Update mesh if it exists
        if (avatar.mesh) {
            avatar.mesh.position = position;
            if (rotation) {
                avatar.mesh.rotationQuaternion = rotation;
            }
        }

        // Update nameplate visibility based on distance
        if (avatar.nameplate && this.camera) {
            const distance = BABYLON.Vector3.Distance(this.camera.position, position);
            avatar.nameplate.isVisible = distance <= this.defaultConfig.nameplateDistance;
        }
    }

    /**
     * Start broadcasting local avatar position
     */
    private startPositionBroadcast(): void {
        if (this.updateInterval) {
            return;
        }

        logger.info('Starting avatar position broadcast');

        this.updateInterval = setInterval(async () => {
            if (!this.localAgentId || !this.camera) {
                return;
            }

            const localAvatar = this.avatars.get(this.localAgentId);
            if (!localAvatar) {
                return;
            }

            // Update local avatar position from camera
            localAvatar.position = this.camera.position.clone();

            // Get camera rotation as quaternion
            const cameraRotation = this.camera.absoluteRotation;

            // Broadcast to Vircadia
            await this.broadcastAvatarUpdate(localAvatar, cameraRotation);

        }, 100); // 10 Hz update rate
    }

    /**
     * Broadcast avatar update to Vircadia
     */
    private async broadcastAvatarUpdate(avatar: UserAvatar, rotation: BABYLON.Quaternion): Promise<void> {
        try {
            const query = `
                UPDATE entity.entities
                SET meta__data = jsonb_set(
                    jsonb_set(
                        jsonb_set(
                            jsonb_set(
                                meta__data,
                                '{position,x}', '${avatar.position.x}'::text::jsonb
                            ),
                            '{position,y}', '${avatar.position.y}'::text::jsonb
                        ),
                        '{position,z}', '${avatar.position.z}'::text::jsonb
                    ),
                    '{rotation}', '${JSON.stringify({
                        x: rotation.x,
                        y: rotation.y,
                        z: rotation.z,
                        w: rotation.w
                    })}'::jsonb
                )
                WHERE general__entity_name = 'avatar_${avatar.agentId}'
            `;

            await this.client.Utilities.Connection.query({
                query,
                timeoutMs: 1000
            });

        } catch (error) {
            // Silent fail for position updates to avoid spam
            logger.debug('Failed to broadcast avatar update:', error);
        }
    }

    /**
     * Fetch remote avatars from Vircadia
     */
    private async fetchRemoteAvatars(): Promise<void> {
        try {
            const query = `
                SELECT * FROM entity.entities
                WHERE general__entity_name LIKE 'avatar_%'
                AND general__entity_name != 'avatar_${this.localAgentId}'
            `;

            const result = await this.client.Utilities.Connection.query<{ result: any[] }>({
                query,
                timeoutMs: 5000
            });

            if (!result?.result) {
                return;
            }

            const avatarEntities = result.result as any[];

            for (const entity of avatarEntities) {
                const agentId = entity.general__entity_name.replace('avatar_', '');
                const metadata = entity.meta__data;

                if (!metadata?.username) {
                    continue;
                }

                // Load avatar if not already loaded
                if (!this.avatars.has(agentId)) {
                    await this.loadRemoteAvatar(agentId, metadata.username);
                }

                // Update position
                if (metadata.position) {
                    const position = new BABYLON.Vector3(
                        metadata.position.x,
                        metadata.position.y,
                        metadata.position.z
                    );

                    let rotation: BABYLON.Quaternion | undefined;
                    if (metadata.rotation) {
                        rotation = new BABYLON.Quaternion(
                            metadata.rotation.x,
                            metadata.rotation.y,
                            metadata.rotation.z,
                            metadata.rotation.w
                        );
                    }

                    this.updateAvatarPosition(agentId, position, rotation);
                }
            }

        } catch (error) {
            logger.error('Failed to fetch remote avatars:', error);
        }
    }

    /**
     * Sync avatar entity to Vircadia database
     */
    private async syncAvatarToVircadia(avatar: UserAvatar): Promise<void> {
        try {
            const query = `
                INSERT INTO entity.entities (
                    general__entity_name,
                    general__semantic_version,
                    general__created_by,
                    group__sync,
                    group__load_priority,
                    meta__data
                ) VALUES (
                    'avatar_${avatar.agentId}',
                    '1.0.0',
                    'visionflow-xr',
                    'public.NORMAL',
                    100,
                    '${JSON.stringify({
                        entityType: 'avatar',
                        username: avatar.username,
                        position: {
                            x: avatar.position.x,
                            y: avatar.position.y,
                            z: avatar.position.z
                        },
                        rotation: {
                            x: avatar.rotation.x,
                            y: avatar.rotation.y,
                            z: avatar.rotation.z,
                            w: avatar.rotation.w
                        }
                    })}'::jsonb
                )
                ON CONFLICT (general__entity_name)
                DO UPDATE SET meta__data = EXCLUDED.meta__data
            `;

            await this.client.Utilities.Connection.query({
                query,
                timeoutMs: 5000
            });

            logger.info(`Avatar synced to Vircadia: ${avatar.username}`);

        } catch (error) {
            logger.error('Failed to sync avatar to Vircadia:', error);
        }
    }

    /**
     * Remove avatar
     */
    removeAvatar(agentId: string): void {
        const avatar = this.avatars.get(agentId);
        if (!avatar) {
            return;
        }

        logger.info(`Removing avatar: ${avatar.username}`);

        // Dispose mesh
        if (avatar.mesh) {
            avatar.mesh.dispose();
        }

        // Dispose nameplate
        if (avatar.nameplate) {
            avatar.nameplate.dispose();
        }

        this.avatars.delete(agentId);
    }

    /**
     * Get all active avatars
     */
    getAvatars(): UserAvatar[] {
        return Array.from(this.avatars.values());
    }

    /**
     * Get avatar count
     */
    getAvatarCount(): number {
        return this.avatars.size;
    }

    /**
     * Stop position broadcast
     */
    private stopPositionBroadcast(): void {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
            logger.info('Stopped avatar position broadcast');
        }
    }

    /**
     * Dispose avatar manager
     */
    dispose(): void {
        logger.info('Disposing AvatarManager');

        this.stopPositionBroadcast();

        // Remove all avatars
        this.avatars.forEach((_, agentId) => {
            this.removeAvatar(agentId);
        });

        this.avatars.clear();
    }
}
