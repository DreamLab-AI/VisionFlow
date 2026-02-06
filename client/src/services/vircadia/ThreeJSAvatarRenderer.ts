import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { ClientCore } from './VircadiaClientCore';
import { createLogger } from '../../utils/loggerConfig';

const logger = createLogger('ThreeJSAvatarRenderer');

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
  nameplate?: THREE.Sprite;
  mixer?: THREE.AnimationMixer;
  animations?: THREE.AnimationClip[];
}

export class ThreeJSAvatarRenderer {
  private avatars = new Map<string, UserAvatar>();
  private localAgentId: string | null = null;
  private updateInterval: ReturnType<typeof setInterval> | null = null;
  private gltfLoader = new GLTFLoader();

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
  }

  // Fix 2: Arrow function class properties for proper listener removal
  private handleStatusChange = (): void => {
    const info = this.client.Utilities.Connection.getConnectionInfo();
    if (info.isConnected && info.agentId) {
      this.localAgentId = info.agentId;
      logger.info(`Local agent ID: ${this.localAgentId}`);
    }
  };

  private handleSyncUpdate = async (): Promise<void> => {
    await this.fetchRemoteAvatars();
  };

  private setupConnectionListeners(): void {
    this.client.Utilities.Connection.addEventListener('statusChange', this.handleStatusChange);
    this.client.Utilities.Connection.addEventListener('syncUpdate', this.handleSyncUpdate);
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

    try {
      const gltf = await this.gltfLoader.loadAsync(this.defaultConfig.modelUrl);
      const model = gltf.scene;

      model.name = `avatar_${agentId}`;
      model.scale.setScalar(this.defaultConfig.scale);
      this.scene.add(model);

      // Setup animations
      let mixer: THREE.AnimationMixer | undefined;
      if (this.defaultConfig.enableAnimations && gltf.animations.length > 0) {
        mixer = new THREE.AnimationMixer(model);
        const idleClip = gltf.animations.find(clip =>
          clip.name.toLowerCase().includes('idle')
        );
        if (idleClip) {
          mixer.clipAction(idleClip).play();
        }
      }

      // Create nameplate
      let nameplate: THREE.Sprite | undefined;
      if (this.defaultConfig.showNameplate) {
        nameplate = this.createNameplate(username);
        nameplate.position.y = 2.2;
        model.add(nameplate);
      }

      // Fix 4: Proper type assertion instead of @ts-ignore
      const avatar: UserAvatar = {
        agentId,
        username,
        position: new THREE.Vector3(),
        rotation: new THREE.Quaternion(),
        mesh: model as unknown as THREE.Object3D,
        nameplate,
        mixer,
        animations: gltf.animations
      };

      this.avatars.set(agentId, avatar);
      logger.info(`Remote avatar loaded: ${username}`);

    } catch (error) {
      logger.error(`Failed to load avatar for ${username}:`, error);
    }
  }

  private createNameplate(username: string): THREE.Sprite {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 128;

    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(0, 0, 512, 128);
    ctx.fillStyle = 'white';
    ctx.font = 'bold 56px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(username, 256, 80);

    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      opacity: 0.9
    });

    const sprite = new THREE.Sprite(material);
    sprite.scale.set(2, 0.5, 1);

    return sprite;
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

    // Update nameplate visibility based on distance
    if (avatar.nameplate && this.camera) {
      const distance = this.camera.position.distanceTo(position);
      avatar.nameplate.visible = distance <= this.defaultConfig.nameplateDistance;
    }
  }

  private startPositionBroadcast(): void {
    if (this.updateInterval) return;

    logger.info('Starting avatar position broadcast');

    this.updateInterval = setInterval(async () => {
      if (!this.localAgentId || !this.camera) return;

      const localAvatar = this.avatars.get(this.localAgentId);
      if (!localAvatar) return;

      localAvatar.position.copy(this.camera.position);
      this.camera.getWorldQuaternion(localAvatar.rotation);

      await this.broadcastAvatarUpdate(localAvatar);
    }, 100);
  }

  // Fix 1: Parameterized SQL query
  private async broadcastAvatarUpdate(avatar: UserAvatar): Promise<void> {
    try {
      const query = `
        UPDATE entity.entities
        SET meta__data = jsonb_set(
          jsonb_set(
            jsonb_set(meta__data, '{position}', $1::jsonb),
            '{rotation}', $2::jsonb
          ),
          '{timestamp}', $3::text::jsonb
        )
        WHERE general__entity_name = $4
      `;

      await this.client.Utilities.Connection.query({
        query,
        parameters: [
          JSON.stringify({ x: avatar.position.x, y: avatar.position.y, z: avatar.position.z }),
          JSON.stringify({ x: avatar.rotation.x, y: avatar.rotation.y, z: avatar.rotation.z, w: avatar.rotation.w }),
          String(Date.now()),
          `avatar_${avatar.agentId}`
        ],
        timeoutMs: 1000
      });
    } catch (error) {
      logger.debug('Failed to broadcast avatar update:', error);
    }
  }

  // Fix 1: Parameterized SQL query
  private async fetchRemoteAvatars(): Promise<void> {
    try {
      const query = `
        SELECT * FROM entity.entities
        WHERE general__entity_name LIKE $1
        AND general__entity_name != $2
      `;

      const result = await this.client.Utilities.Connection.query<{ result: any[] }>({
        query,
        parameters: [
          'avatar_%',
          `avatar_${this.localAgentId}`
        ],
        timeoutMs: 5000
      });

      if (!result?.result) return;

      for (const entity of (result as any).result) {
        const agentId = entity.general__entity_name.replace('avatar_', '');
        const metadata = entity.meta__data;

        if (!metadata?.username) continue;

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

  // Fix 1: Parameterized SQL query
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
          $1, $2, $3, $4, $5, $6::jsonb
        )
        ON CONFLICT (general__entity_name)
        DO UPDATE SET meta__data = EXCLUDED.meta__data
      `;

      await this.client.Utilities.Connection.query({
        query,
        parameters: [
          `avatar_${avatar.agentId}`,
          '1.0.0',
          'visionflow-xr',
          'public.NORMAL',
          100,
          JSON.stringify({
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
          })
        ],
        timeoutMs: 5000
      });

      logger.info(`Avatar synced to Vircadia: ${avatar.username}`);
    } catch (error) {
      logger.error('Failed to sync avatar to Vircadia:', error);
    }
  }

  update(deltaTime: number): void {
    // Update animation mixers
    this.avatars.forEach(avatar => {
      if (avatar.mixer) {
        avatar.mixer.update(deltaTime);
      }
    });
  }

  removeAvatar(agentId: string): void {
    const avatar = this.avatars.get(agentId);
    if (!avatar) return;

    logger.info(`Removing avatar: ${avatar.username}`);

    if (avatar.mesh) {
      // Fix 4: Proper type assertion instead of @ts-ignore
      // eslint-disable-next-line @typescript-eslint/no-explicit-any -- @types/three dual-path resolution (build/ vs src/) creates incompatible Object3D types
      this.scene.remove(avatar.mesh as any);
      avatar.mesh.traverse(child => {
        if ((child as THREE.Mesh).geometry) {
          (child as THREE.Mesh).geometry.dispose();
        }
        if ((child as THREE.Mesh).material) {
          const material = (child as THREE.Mesh).material;
          if (Array.isArray(material)) {
            material.forEach(m => m.dispose());
          } else {
            material.dispose();
          }
        }
      });
    }

    // Fix 3: Dispose nameplate sprite textures to prevent texture leak
    if (avatar.nameplate) {
      avatar.nameplate.material.map?.dispose();
      avatar.nameplate.material.dispose();
    }

    this.avatars.delete(agentId);
  }

  getAvatars(): UserAvatar[] {
    return Array.from(this.avatars.values());
  }

  getAvatarCount(): number {
    return this.avatars.size;
  }

  dispose(): void {
    logger.info('Disposing ThreeJSAvatarRenderer');

    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }

    // Fix 2: Remove event listeners to prevent leaks
    this.client.Utilities.Connection.removeEventListener('statusChange', this.handleStatusChange);
    this.client.Utilities.Connection.removeEventListener('syncUpdate', this.handleSyncUpdate);

    this.avatars.forEach((_, agentId) => {
      this.removeAvatar(agentId);
    });

    this.avatars.clear();
  }
}
