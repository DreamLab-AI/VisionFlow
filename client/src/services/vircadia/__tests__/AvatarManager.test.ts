/**
 * AvatarManager Tests - Phase 5: Unit Testing
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as BABYLON from '@babylonjs/core';
import { AvatarManager } from '../AvatarManager';
import { ClientCore } from '../VircadiaClientCore';

describe('AvatarManager', () => {
    let scene: BABYLON.Scene;
    let engine: BABYLON.Engine;
    let camera: BABYLON.Camera;
    let mockClient: ClientCore;
    let avatarManager: AvatarManager;

    beforeEach(() => {
        // Create Babylon scene
        const canvas = document.createElement('canvas');
        engine = new BABYLON.Engine(canvas, false);
        scene = new BABYLON.Scene(engine);
        camera = new BABYLON.UniversalCamera('camera', new BABYLON.Vector3(0, 1.6, -3), scene);

        // Mock Vircadia client
        mockClient = {
            Utilities: {
                Connection: {
                    getConnectionInfo: vi.fn().mockReturnValue({
                        isConnected: true,
                        agentId: 'test-agent-123'
                    }),
                    addEventListener: vi.fn().mockReturnValue(() => {}),
                    query: vi.fn().mockResolvedValue({ result: [] })
                }
            },
            dispose: vi.fn()
        } as any;

        avatarManager = new AvatarManager(scene, mockClient, camera);
    });

    afterEach(() => {
        avatarManager?.dispose();
        scene?.dispose();
        engine?.dispose();
        vi.clearAllMocks();
    });

    describe('Initialization', () => {
        it('should create avatar manager', () => {
            expect(avatarManager).toBeDefined();
            expect(avatarManager.getAvatarCount()).toBe(0);
        });

        it('should set up connection listeners', () => {
            expect(mockClient.Utilities.Connection.addEventListener).toHaveBeenCalledWith(
                'statusChange',
                expect.any(Function)
            );
            expect(mockClient.Utilities.Connection.addEventListener).toHaveBeenCalledWith(
                'syncUpdate',
                expect.any(Function)
            );
        });
    });

    describe('Local Avatar', () => {
        it('should create local avatar', async () => {
            await avatarManager.createLocalAvatar('TestUser');

            expect(avatarManager.getAvatarCount()).toBe(1);
            const avatars = avatarManager.getAvatars();
            expect(avatars[0].username).toBe('TestUser');
            expect(avatars[0].agentId).toBe('test-agent-123');
        });

        it('should sync local avatar to Vircadia', async () => {
            await avatarManager.createLocalAvatar('TestUser');

            expect(mockClient.Utilities.Connection.query).toHaveBeenCalledWith(
                expect.objectContaining({
                    query: expect.stringContaining('INSERT INTO entity.entities'),
                    timeoutMs: 5000
                })
            );
        });

        it('should not create local avatar without agent ID', async () => {
            mockClient.Utilities.Connection.getConnectionInfo = vi.fn().mockReturnValue({
                isConnected: true,
                agentId: null
            });

            await avatarManager.createLocalAvatar('TestUser');

            expect(avatarManager.getAvatarCount()).toBe(0);
        });
    });

    describe('Remote Avatars', () => {
        it('should load remote avatar', async () => {
            // Mock SceneLoader
            const mockMesh = new BABYLON.Mesh('avatar', scene);
            vi.spyOn(BABYLON.SceneLoader, 'ImportMeshAsync').mockResolvedValue({
                meshes: [mockMesh],
                particleSystems: [],
                skeletons: [],
                animationGroups: [],
                transformNodes: [],
                geometries: [],
                lights: []
            });

            await avatarManager.loadRemoteAvatar('remote-agent-456', 'RemoteUser');

            expect(avatarManager.getAvatarCount()).toBe(1);
            const avatars = avatarManager.getAvatars();
            expect(avatars[0].agentId).toBe('remote-agent-456');
            expect(avatars[0].username).toBe('RemoteUser');
            expect(avatars[0].mesh).toBeDefined();
        });

        it('should not reload existing avatar', async () => {
            const mockMesh = new BABYLON.Mesh('avatar', scene);
            const importSpy = vi.spyOn(BABYLON.SceneLoader, 'ImportMeshAsync').mockResolvedValue({
                meshes: [mockMesh],
                particleSystems: [],
                skeletons: [],
                animationGroups: [],
                transformNodes: [],
                geometries: [],
                lights: []
            });

            await avatarManager.loadRemoteAvatar('remote-agent-456', 'RemoteUser');
            await avatarManager.loadRemoteAvatar('remote-agent-456', 'RemoteUser');

            expect(importSpy).toHaveBeenCalledTimes(1);
            expect(avatarManager.getAvatarCount()).toBe(1);
        });
    });

    describe('Avatar Position Updates', () => {
        it('should update avatar position', async () => {
            const mockMesh = new BABYLON.Mesh('avatar', scene);
            vi.spyOn(BABYLON.SceneLoader, 'ImportMeshAsync').mockResolvedValue({
                meshes: [mockMesh],
                particleSystems: [],
                skeletons: [],
                animationGroups: [],
                transformNodes: [],
                geometries: [],
                lights: []
            });

            await avatarManager.loadRemoteAvatar('remote-agent-456', 'RemoteUser');

            const newPosition = new BABYLON.Vector3(5, 2, 3);
            const newRotation = BABYLON.Quaternion.RotationYawPitchRoll(1, 0, 0);

            avatarManager.updateAvatarPosition('remote-agent-456', newPosition, newRotation);

            const avatars = avatarManager.getAvatars();
            expect(avatars[0].position.equals(newPosition)).toBe(true);
            expect(avatars[0].rotation.equals(newRotation)).toBe(true);
        });

        it('should update nameplate visibility based on distance', async () => {
            const mockMesh = new BABYLON.Mesh('avatar', scene);
            const mockNameplate = new BABYLON.Mesh('nameplate', scene);

            vi.spyOn(BABYLON.SceneLoader, 'ImportMeshAsync').mockResolvedValue({
                meshes: [mockMesh],
                particleSystems: [],
                skeletons: [],
                animationGroups: [],
                transformNodes: [],
                geometries: [],
                lights: []
            });

            await avatarManager.loadRemoteAvatar('remote-agent-456', 'RemoteUser');

            const avatars = avatarManager.getAvatars();
            avatars[0].nameplate = mockNameplate;

            // Far away - nameplate hidden
            avatarManager.updateAvatarPosition('remote-agent-456', new BABYLON.Vector3(100, 0, 0));
            expect(mockNameplate.isVisible).toBe(false);

            // Close - nameplate visible
            avatarManager.updateAvatarPosition('remote-agent-456', new BABYLON.Vector3(5, 0, 0));
            expect(mockNameplate.isVisible).toBe(true);
        });
    });

    describe('Avatar Removal', () => {
        it('should remove avatar', async () => {
            const mockMesh = new BABYLON.Mesh('avatar', scene);
            vi.spyOn(BABYLON.SceneLoader, 'ImportMeshAsync').mockResolvedValue({
                meshes: [mockMesh],
                particleSystems: [],
                skeletons: [],
                animationGroups: [],
                transformNodes: [],
                geometries: [],
                lights: []
            });

            await avatarManager.loadRemoteAvatar('remote-agent-456', 'RemoteUser');
            expect(avatarManager.getAvatarCount()).toBe(1);

            avatarManager.removeAvatar('remote-agent-456');
            expect(avatarManager.getAvatarCount()).toBe(0);
        });

        it('should dispose avatar meshes on removal', async () => {
            const mockMesh = new BABYLON.Mesh('avatar', scene);
            const disposeSpy = vi.spyOn(mockMesh, 'dispose');

            vi.spyOn(BABYLON.SceneLoader, 'ImportMeshAsync').mockResolvedValue({
                meshes: [mockMesh],
                particleSystems: [],
                skeletons: [],
                animationGroups: [],
                transformNodes: [],
                geometries: [],
                lights: []
            });

            await avatarManager.loadRemoteAvatar('remote-agent-456', 'RemoteUser');
            avatarManager.removeAvatar('remote-agent-456');

            expect(disposeSpy).toHaveBeenCalled();
        });
    });

    describe('Disposal', () => {
        it('should dispose all avatars', async () => {
            const mockMesh1 = new BABYLON.Mesh('avatar1', scene);
            const mockMesh2 = new BABYLON.Mesh('avatar2', scene);

            vi.spyOn(BABYLON.SceneLoader, 'ImportMeshAsync')
                .mockResolvedValueOnce({
                    meshes: [mockMesh1],
                    particleSystems: [],
                    skeletons: [],
                    animationGroups: [],
                    transformNodes: [],
                    geometries: [],
                    lights: []
                })
                .mockResolvedValueOnce({
                    meshes: [mockMesh2],
                    particleSystems: [],
                    skeletons: [],
                    animationGroups: [],
                    transformNodes: [],
                    geometries: [],
                    lights: []
                });

            await avatarManager.loadRemoteAvatar('agent-1', 'User1');
            await avatarManager.loadRemoteAvatar('agent-2', 'User2');

            avatarManager.dispose();

            expect(avatarManager.getAvatarCount()).toBe(0);
        });
    });
});
