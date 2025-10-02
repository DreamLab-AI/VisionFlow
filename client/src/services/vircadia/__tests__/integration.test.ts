/**
 * Integration Tests - Phase 5: Multi-user scenario testing
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as BABYLON from '@babylonjs/core';
import { ClientCore } from '../VircadiaClientCore';
import { VircadiaSceneBridge } from '../../immersive/babylon/VircadiaSceneBridge';
import { AvatarManager } from '../AvatarManager';
import { CollaborativeGraphSync } from '../CollaborativeGraphSync';

describe('Multi-User Integration Tests', () => {
    let scene: BABYLON.Scene;
    let engine: BABYLON.Engine;
    let camera: BABYLON.Camera;
    let mockClient1: ClientCore;
    let mockClient2: ClientCore;

    beforeEach(() => {
        // Create Babylon scene
        const canvas = document.createElement('canvas');
        engine = new BABYLON.Engine(canvas, false);
        scene = new BABYLON.Scene(engine);
        camera = new BABYLON.UniversalCamera('camera', new BABYLON.Vector3(0, 1.6, -3), scene);

        // Mock two users
        mockClient1 = {
            Utilities: {
                Connection: {
                    getConnectionInfo: vi.fn().mockReturnValue({
                        isConnected: true,
                        agentId: 'user-1'
                    }),
                    addEventListener: vi.fn().mockReturnValue(() => {}),
                    query: vi.fn().mockResolvedValue({ result: [] })
                }
            },
            dispose: vi.fn()
        } as any;

        mockClient2 = {
            Utilities: {
                Connection: {
                    getConnectionInfo: vi.fn().mockReturnValue({
                        isConnected: true,
                        agentId: 'user-2'
                    }),
                    addEventListener: vi.fn().mockReturnValue(() => {}),
                    query: vi.fn().mockResolvedValue({ result: [] })
                }
            },
            dispose: vi.fn()
        } as any;
    });

    afterEach(() => {
        scene?.dispose();
        engine?.dispose();
        vi.clearAllMocks();
    });

    describe('Avatar Synchronization', () => {
        it('should sync avatars between two users', async () => {
            const avatarManager1 = new AvatarManager(scene, mockClient1, camera);
            const avatarManager2 = new AvatarManager(scene, mockClient2, camera);

            // User 1 creates their avatar
            await avatarManager1.createLocalAvatar('Alice');
            expect(avatarManager1.getAvatarCount()).toBe(1);

            // User 2 creates their avatar
            await avatarManager2.createLocalAvatar('Bob');
            expect(avatarManager2.getAvatarCount()).toBe(1);

            // Simulate receiving remote avatar data
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

            // User 1 sees User 2's avatar
            await avatarManager1.loadRemoteAvatar('user-2', 'Bob');
            expect(avatarManager1.getAvatarCount()).toBe(2);

            // User 2 sees User 1's avatar
            await avatarManager2.loadRemoteAvatar('user-1', 'Alice');
            expect(avatarManager2.getAvatarCount()).toBe(2);

            avatarManager1.dispose();
            avatarManager2.dispose();
        });

        it('should sync avatar position updates', async () => {
            const avatarManager = new AvatarManager(scene, mockClient1, camera);

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

            await avatarManager.loadRemoteAvatar('user-2', 'Bob');

            const newPosition = new BABYLON.Vector3(5, 2, 3);
            avatarManager.updateAvatarPosition('user-2', newPosition);

            const avatars = avatarManager.getAvatars();
            const bobAvatar = avatars.find(a => a.agentId === 'user-2');
            expect(bobAvatar?.position.equals(newPosition)).toBe(true);

            avatarManager.dispose();
        });
    });

    describe('Collaborative Graph Interaction', () => {
        it('should sync node selections between users', async () => {
            const collabSync1 = new CollaborativeGraphSync(scene, mockClient1);
            const collabSync2 = new CollaborativeGraphSync(scene, mockClient2);

            await collabSync1.initialize();
            await collabSync2.initialize();

            // User 1 selects nodes
            await collabSync1.selectNodes(['node-1', 'node-2']);

            // Verify query was sent
            expect(mockClient1.Utilities.Connection.query).toHaveBeenCalledWith(
                expect.objectContaining({
                    query: expect.stringContaining('selection_user-1')
                })
            );

            // Simulate User 2 receiving the selection update
            mockClient2.Utilities.Connection.query = vi.fn().mockResolvedValue({
                result: [{
                    general__entity_name: 'selection_user-1',
                    meta__data: {
                        type: 'selection',
                        agentId: 'user-1',
                        nodeIds: ['node-1', 'node-2'],
                        timestamp: Date.now()
                    }
                }]
            });

            // User 2 fetches selections
            const selections = collabSync2.getActiveSelections();
            expect(selections).toBeDefined();

            collabSync1.dispose();
            collabSync2.dispose();
        });

        it('should sync annotations between users', async () => {
            const collabSync = new CollaborativeGraphSync(scene, mockClient1, {
                enableAnnotations: true
            });

            await collabSync.initialize();

            const position = new BABYLON.Vector3(1, 1, 1);
            await collabSync.createAnnotation('node-1', 'Important finding', position);

            expect(mockClient1.Utilities.Connection.query).toHaveBeenCalledWith(
                expect.objectContaining({
                    query: expect.stringContaining('annotation_')
                })
            );

            const annotations = collabSync.getNodeAnnotations('node-1');
            expect(annotations.length).toBeGreaterThan(0);

            collabSync.dispose();
        });
    });

    describe('Scene Bridge Graph Sync', () => {
        it('should load graph from Vircadia', async () => {
            mockClient1.Utilities.Connection.query = vi.fn().mockResolvedValue({
                result: [
                    {
                        general__entity_name: 'node_node-1',
                        meta__data: {
                            entityType: 'node',
                            graphId: 'node-1',
                            position: { x: 1, y: 1, z: 1 },
                            label: 'Test Node'
                        }
                    }
                ]
            });

            const sceneBridge = new VircadiaSceneBridge(scene, mockClient1);
            await sceneBridge.loadGraphFromVircadia();

            expect(mockClient1.Utilities.Connection.query).toHaveBeenCalled();

            sceneBridge.dispose();
        });

        it('should push graph to Vircadia', async () => {
            const sceneBridge = new VircadiaSceneBridge(scene, mockClient1);

            const graphData = {
                nodes: [
                    { id: 'node-1', label: 'Node 1', x: 1, y: 1, z: 1 }
                ],
                edges: [
                    { id: 'edge-1', source: 'node-1', target: 'node-2' }
                ]
            };

            await sceneBridge.pushGraphToVircadia(graphData);

            expect(mockClient1.Utilities.Connection.query).toHaveBeenCalled();

            sceneBridge.dispose();
        });
    });

    describe('Performance Under Load', () => {
        it('should handle multiple concurrent users', async () => {
            const avatarManager = new AvatarManager(scene, mockClient1, camera);

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

            // Simulate 5 concurrent users
            const userPromises = [];
            for (let i = 0; i < 5; i++) {
                userPromises.push(
                    avatarManager.loadRemoteAvatar(`user-${i}`, `User${i}`)
                );
            }

            await Promise.all(userPromises);

            expect(avatarManager.getAvatarCount()).toBe(5);

            avatarManager.dispose();
        });

        it('should handle rapid position updates', async () => {
            const avatarManager = new AvatarManager(scene, mockClient1, camera);

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

            await avatarManager.loadRemoteAvatar('user-2', 'Bob');

            // Simulate 100 rapid position updates
            for (let i = 0; i < 100; i++) {
                avatarManager.updateAvatarPosition(
                    'user-2',
                    new BABYLON.Vector3(i * 0.1, 1, 1)
                );
            }

            const avatars = avatarManager.getAvatars();
            const bobAvatar = avatars.find(a => a.agentId === 'user-2');
            expect(bobAvatar?.position.x).toBeCloseTo(9.9, 1);

            avatarManager.dispose();
        });
    });

    describe('Error Handling', () => {
        it('should handle failed avatar loading gracefully', async () => {
            const avatarManager = new AvatarManager(scene, mockClient1, camera);

            vi.spyOn(BABYLON.SceneLoader, 'ImportMeshAsync').mockRejectedValue(
                new Error('Failed to load model')
            );

            await avatarManager.loadRemoteAvatar('user-2', 'Bob');

            // Should not crash, avatar count should remain 0
            expect(avatarManager.getAvatarCount()).toBe(0);

            avatarManager.dispose();
        });

        it('should handle Vircadia connection failures', async () => {
            mockClient1.Utilities.Connection.query = vi.fn().mockRejectedValue(
                new Error('Connection failed')
            );

            const collabSync = new CollaborativeGraphSync(scene, mockClient1);
            await collabSync.initialize();

            // Should not crash when query fails
            await collabSync.selectNodes(['node-1']);

            collabSync.dispose();
        });
    });
});
