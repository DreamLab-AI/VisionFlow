/**
 * AvatarManager - Multi-user avatar synchronization via Vircadia
 *
 * TODO: This service was migrated from Babylon.js to Three.js.
 * Avatar loading and mesh creation need full Three.js implementation.
 * Current implementation provides type stubs and position sync only.
 */

import * as THREE from 'three';
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
    position: THREE.Vector3;
    rotation: THREE.Quaternion;
    mesh?: THREE.Object3D;
    nameplate?: THREE.Mesh;
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
        private scene: THREE.Scene,
        private client: ClientCore,
        private camera: THREE.Camera,
        config?: Partial<AvatarConfig>
    ) {
        this.defaultConfig = { ...this.defaultConfig, ...config };
        this.setupConnectionListeners();
        logger.info('AvatarManager initialized (Three.js mode)');
    }

    private setupConnectionListeners(): void {
        this.client.Utilities.Connection.addEventListener('statusChange', () => {
            const info = this.client.Utilities.Connection.getConnectionInfo();
            if (info.isConnected && info.agentId) {
                this.localAgentId = info.agentId;
                logger.info(`Local agent ID: ${this.localAgentId}`);
            }
        });

        this.client.Utilities.Connection.addEventListener('syncUpdate', async () => {
            await this.fetchRemoteAvatars();
        });
    }

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
            rotation: new THREE.Quaternion()
        };

        this.avatars.set(this.localAgentId, avatar);
        this.startPositionBroadcast();
        await this.syncAvatarToVircadia(avatar);
    }

    async loadRemoteAvatar(agentId: string, username: string): Promise<void> {
        if (this.avatars.has(agentId)) {
            return;
        }

        logger.info(`Loading remote avatar for ${username} (${agentId})`);

        // TODO: Implement GLTFLoader for Three.js avatar loading
        // For now, create a simple placeholder
        const geometry = new THREE.SphereGeometry(0.3, 16, 16);
        const material = new THREE.MeshBasicMaterial({ color: 0x4488ff });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.name = `avatar_${agentId}`;
        this.scene.add(mesh);

        const avatar: UserAvatar = {
            agentId,
            username,
            position: new THREE.Vector3(),
            rotation: new THREE.Quaternion(),
            // @ts-ignore - THREE.js type mismatch between Mesh and Object3D
            mesh
        };

        this.avatars.set(agentId, avatar);
        logger.info(`Remote avatar loaded: ${username} (placeholder mesh)`);
    }

    updateAvatarPosition(agentId: string, position: THREE.Vector3, rotation?: THREE.Quaternion): void {
        const avatar = this.avatars.get(agentId);
        if (!avatar) {
            logger.warn(`Cannot update avatar: ${agentId} not found`);
            return;
        }

        avatar.position.copy(position);
        if (rotation) {
            avatar.rotation.copy(rotation);
        }

        if (avatar.mesh) {
            avatar.mesh.position.copy(position);
            if (rotation) {
                avatar.mesh.quaternion.copy(rotation);
            }
        }
    }

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

            localAvatar.position.copy(this.camera.position);
            const cameraRotation = this.camera.quaternion.clone();
            await this.broadcastAvatarUpdate(localAvatar, cameraRotation);
        }, 100);
    }

    private async broadcastAvatarUpdate(avatar: UserAvatar, rotation: THREE.Quaternion): Promise<void> {
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
            logger.debug('Failed to broadcast avatar update:', error);
        }
    }

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

            const avatarEntities = (result as any).result as any[];

            for (const entity of avatarEntities) {
                const agentId = entity.general__entity_name.replace('avatar_', '');
                const metadata = entity.meta__data;

                if (!metadata?.username) {
                    continue;
                }

                if (!this.avatars.has(agentId)) {
                    await this.loadRemoteAvatar(agentId, metadata.username);
                }

                if (metadata.position) {
                    const position = new THREE.Vector3(
                        metadata.position.x,
                        metadata.position.y,
                        metadata.position.z
                    );

                    let rotation: THREE.Quaternion | undefined;
                    if (metadata.rotation) {
                        rotation = new THREE.Quaternion(
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

    removeAvatar(agentId: string): void {
        const avatar = this.avatars.get(agentId);
        if (!avatar) {
            return;
        }

        logger.info(`Removing avatar: ${avatar.username}`);

        if (avatar.mesh) {
            // @ts-ignore - THREE.js type mismatch between Object3D variants
            this.scene.remove(avatar.mesh);
            if (avatar.mesh instanceof THREE.Mesh) {
                avatar.mesh.geometry.dispose();
                if (avatar.mesh.material instanceof THREE.Material) {
                    avatar.mesh.material.dispose();
                }
            }
        }

        if (avatar.nameplate) {
            this.scene.remove(avatar.nameplate);
            avatar.nameplate.geometry.dispose();
            if (avatar.nameplate.material instanceof THREE.Material) {
                avatar.nameplate.material.dispose();
            }
        }

        this.avatars.delete(agentId);
    }

    getAvatars(): UserAvatar[] {
        return Array.from(this.avatars.values());
    }

    getAvatarCount(): number {
        return this.avatars.size;
    }

    private stopPositionBroadcast(): void {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
            logger.info('Stopped avatar position broadcast');
        }
    }

    dispose(): void {
        logger.info('Disposing AvatarManager');
        this.stopPositionBroadcast();
        this.avatars.forEach((_, agentId) => {
            this.removeAvatar(agentId);
        });
        this.avatars.clear();
    }
}
