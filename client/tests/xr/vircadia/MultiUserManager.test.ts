import * as BABYLON from '@babylonjs/core'
import { MultiUserManager } from '../../../src/services/vircadia/MultiUserManager'

// Mock dependencies
jest.mock('@babylonjs/core')
jest.mock('../../../src/utils/logger', () => ({
  createLogger: () => ({
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn()
  })
}))

describe('MultiUserManager', () => {
  let mockScene: any
  let multiUserManager: MultiUserManager
  let mockMeshes: Map<string, any>
  
  beforeEach(() => {
    jest.clearAllMocks()
    mockMeshes = new Map()
    
    // Mock BABYLON scene
    mockScene = {
      xr: null,
      activeCamera: {
        position: { x: 0, y: 1.6, z: -3 },
        rotation: { x: 0, y: 0, z: 0 }
      },
      onBeforeRenderObservable: {
        add: jest.fn()
      }
    }
    
    // Mock BABYLON constructors
    ;(BABYLON.TransformNode as jest.Mock).mockImplementation(() => ({}))
    ;(BABYLON.Vector3 as jest.Mock).mockImplementation((x, y, z) => ({ x, y, z, clone: () => ({ x, y, z }) }))
    ;(BABYLON.Vector3 as any).Lerp = jest.fn((a, b, t) => ({
      x: a.x + (b.x - a.x) * t,
      y: a.y + (b.y - a.y) * t,
      z: a.z + (b.z - a.z) * t
    }))
    ;(BABYLON.Color3 as any).FromHexString = jest.fn((hex) => ({ scale: () => ({}) }))
    
    // Mock mesh creation
    ;(BABYLON.MeshBuilder as any) = {
      CreateSphere: jest.fn((name) => {
        const mesh = {
          name,
          position: { x: 0, y: 0, z: 0 },
          rotation: { x: 0, y: 0, z: 0 },
          material: null,
          parent: null,
          dispose: jest.fn()
        }
        mockMeshes.set(name, mesh)
        return mesh
      }),
      CreateCylinder: jest.fn((name) => {
        const mesh = {
          name,
          position: { x: 0, y: 0, z: 0 },
          rotation: { x: 0, y: 0, z: 0 },
          material: null,
          dispose: jest.fn()
        }
        mockMeshes.set(name, mesh)
        return mesh
      }),
      CreatePlane: jest.fn((name) => {
        const mesh = {
          name,
          position: { x: 0, y: 0, z: 0 },
          parent: null,
          billboardMode: null,
          material: null,
          dispose: jest.fn()
        }
        mockMeshes.set(name, mesh)
        return mesh
      })
    }
    
    // Mock materials
    ;(BABYLON.StandardMaterial as jest.Mock).mockImplementation((name) => ({
      name,
      diffuseColor: null,
      emissiveColor: null,
      specularPower: 0,
      alpha: 1,
      diffuseTexture: null,
      emissiveTexture: null,
      backFaceCulling: true,
      disableLighting: false
    }))
    
    // Mock dynamic texture
    ;(BABYLON.DynamicTexture as jest.Mock).mockImplementation(() => ({
      getContext: jest.fn().mockReturnValue({
        font: '',
        fillStyle: '',
        textAlign: '',
        textBaseline: '',
        fillText: jest.fn()
      }),
      update: jest.fn()
    }))
    
    // Mock billboard mode
    ;(BABYLON.Mesh as any).BILLBOARDMODE_Y = 4
    
    multiUserManager = new MultiUserManager(mockScene)
  })
  
  describe('Initialization', () => {
    it('should initialize successfully', async () => {
      await multiUserManager.initialize()
      
      expect(BABYLON.TransformNode).toHaveBeenCalledWith('avatarContainer', mockScene)
    })
  })
  
  describe('Connection', () => {
    const mockMultiUserState = {
      users: {
        'user1': {
          id: 'user1',
          position: [1, 1.6, 0] as [number, number, number],
          rotation: [0, 0, 0] as [number, number, number],
          isSelecting: false
        }
      },
      localUserId: 'localUser',
      sendUpdate: jest.fn()
    }
    
    it('should connect to multi-user session', async () => {
      await multiUserManager.initialize()
      await multiUserManager.connect(mockMultiUserState)
      
      expect(multiUserManager.isConnected()).toBe(true)
    })
    
    it('should process initial users on connect', async () => {
      await multiUserManager.initialize()
      await multiUserManager.connect(mockMultiUserState)
      
      // Should create avatar for user1 (not local user)
      expect(BABYLON.MeshBuilder.CreateSphere).toHaveBeenCalledWith(
        'avatar_user1',
        expect.any(Object),
        mockScene
      )
    })
    
    it('should start local tracking after connect', async () => {
      await multiUserManager.initialize()
      await multiUserManager.connect(mockMultiUserState)
      
      expect(mockScene.onBeforeRenderObservable.add).toHaveBeenCalled()
    })
  })
  
  describe('User Management', () => {
    const mockState = {
      users: {
        'user1': {
          id: 'user1',
          position: [1, 1.6, 0] as [number, number, number],
          rotation: [0, 0, 0] as [number, number, number],
          isSelecting: false,
          color: '#FF6B6B'
        },
        'user2': {
          id: 'user2',
          position: [2, 1.6, 0] as [number, number, number],
          rotation: [0, Math.PI/2, 0] as [number, number, number],
          isSelecting: true
        }
      },
      localUserId: 'localUser',
      sendUpdate: jest.fn()
    }
    
    beforeEach(async () => {
      await multiUserManager.initialize()
      await multiUserManager.connect(mockState)
    })
    
    it('should add new users with avatars', () => {
      expect(BABYLON.MeshBuilder.CreateSphere).toHaveBeenCalledTimes(2)
      expect(mockMeshes.has('avatar_user1')).toBe(true)
      expect(mockMeshes.has('avatar_user2')).toBe(true)
    })
    
    it('should assign colors to users', () => {
      const avatar1Material = mockMeshes.get('avatar_user1')?.material
      const avatar2Material = mockMeshes.get('avatar_user2')?.material
      
      expect(BABYLON.Color3.FromHexString).toHaveBeenCalledWith('#FF6B6B')
      expect(avatar1Material).toBeDefined()
      expect(avatar2Material).toBeDefined()
    })
    
    it('should position avatars correctly', () => {
      const avatar1 = mockMeshes.get('avatar_user1')
      const avatar2 = mockMeshes.get('avatar_user2')
      
      expect(avatar1?.position).toEqual({ x: 1, y: 1.6, z: 0 })
      expect(avatar2?.position).toEqual({ x: 2, y: 1.6, z: 0 })
    })
    
    it('should create selection pointer for selecting users', () => {
      // user2 is selecting
      expect(BABYLON.MeshBuilder.CreateCylinder).toHaveBeenCalledWith(
        'pointer_user2',
        expect.any(Object),
        mockScene
      )
    })
    
    it('should create name labels for users', () => {
      expect(BABYLON.MeshBuilder.CreatePlane).toHaveBeenCalledWith(
        'label_user1',
        expect.any(Object),
        mockScene
      )
      expect(BABYLON.MeshBuilder.CreatePlane).toHaveBeenCalledWith(
        'label_user2',
        expect.any(Object),
        mockScene
      )
    })
    
    it('should not create avatar for local user', () => {
      const avatarNames = Array.from(mockMeshes.keys()).filter(name => name.startsWith('avatar_'))
      expect(avatarNames).not.toContain('avatar_localUser')
    })
  })
  
  describe('User Updates', () => {
    const initialState = {
      users: {
        'user1': {
          id: 'user1',
          position: [0, 1.6, 0] as [number, number, number],
          rotation: [0, 0, 0] as [number, number, number],
          isSelecting: false
        }
      },
      localUserId: 'localUser',
      sendUpdate: jest.fn()
    }
    
    beforeEach(async () => {
      await multiUserManager.initialize()
      await multiUserManager.connect(initialState)
    })
    
    it('should update user position with interpolation', () => {
      const newState = {
        ...initialState,
        users: {
          'user1': {
            ...initialState.users.user1,
            position: [2, 1.6, 0] as [number, number, number]
          }
        }
      }
      
      multiUserManager.updateState(newState)
      
      expect(BABYLON.Vector3.Lerp).toHaveBeenCalled()
    })
    
    it('should update user rotation', () => {
      const avatar = mockMeshes.get('avatar_user1')
      
      const newState = {
        ...initialState,
        users: {
          'user1': {
            ...initialState.users.user1,
            rotation: [0, Math.PI, 0] as [number, number, number]
          }
        }
      }
      
      multiUserManager.updateState(newState)
      
      expect(avatar?.rotation).toEqual({ x: 0, y: Math.PI, z: 0 })
    })
    
    it('should add selection pointer when user starts selecting', () => {
      const newState = {
        ...initialState,
        users: {
          'user1': {
            ...initialState.users.user1,
            isSelecting: true
          }
        }
      }
      
      multiUserManager.updateState(newState)
      
      expect(BABYLON.MeshBuilder.CreateCylinder).toHaveBeenCalledWith(
        'pointer_user1',
        expect.any(Object),
        mockScene
      )
    })
    
    it('should remove selection pointer when user stops selecting', () => {
      // First add pointer
      multiUserManager.updateState({
        ...initialState,
        users: {
          'user1': {
            ...initialState.users.user1,
            isSelecting: true
          }
        }
      })
      
      const pointer = mockMeshes.get('pointer_user1')
      
      // Then remove
      multiUserManager.updateState({
        ...initialState,
        users: {
          'user1': {
            ...initialState.users.user1,
            isSelecting: false
          }
        }
      })
      
      expect(pointer?.dispose).toHaveBeenCalled()
    })
  })
  
  describe('User Removal', () => {
    const initialState = {
      users: {
        'user1': { id: 'user1', position: [0, 1.6, 0] as [number, number, number], rotation: [0, 0, 0] as [number, number, number], isSelecting: false },
        'user2': { id: 'user2', position: [1, 1.6, 0] as [number, number, number], rotation: [0, 0, 0] as [number, number, number], isSelecting: true }
      },
      localUserId: 'localUser',
      sendUpdate: jest.fn()
    }
    
    beforeEach(async () => {
      await multiUserManager.initialize()
      await multiUserManager.connect(initialState)
    })
    
    it('should remove users who left', () => {
      const avatar1 = mockMeshes.get('avatar_user1')
      const avatar2 = mockMeshes.get('avatar_user2')
      const pointer2 = mockMeshes.get('pointer_user2')
      
      // Update with only user1
      multiUserManager.updateState({
        ...initialState,
        users: {
          'user1': initialState.users.user1
        }
      })
      
      expect(avatar1?.dispose).not.toHaveBeenCalled()
      expect(avatar2?.dispose).toHaveBeenCalled()
      expect(pointer2?.dispose).toHaveBeenCalled()
    })
    
    it('should handle all users leaving', () => {
      const avatar1 = mockMeshes.get('avatar_user1')
      const avatar2 = mockMeshes.get('avatar_user2')
      
      multiUserManager.updateState({
        ...initialState,
        users: {}
      })
      
      expect(avatar1?.dispose).toHaveBeenCalled()
      expect(avatar2?.dispose).toHaveBeenCalled()
    })
  })
  
  describe('Local Tracking', () => {
    const mockState = {
      users: {},
      localUserId: 'localUser',
      sendUpdate: jest.fn()
    }
    
    beforeEach(async () => {
      await multiUserManager.initialize()
      await multiUserManager.connect(mockState)
    })
    
    it('should track camera position', () => {
      // Get the callback added to onBeforeRenderObservable
      const callback = mockScene.onBeforeRenderObservable.add.mock.calls[0][0]
      
      // Execute callback
      callback()
      
      expect(mockState.sendUpdate).toHaveBeenCalledWith({
        type: 'positionUpdate',
        userId: 'localUser',
        position: [0, 1.6, -3],
        rotation: [0, 0, 0]
      })
    })
    
    it('should track XR camera when in XR mode', () => {
      // Setup XR mode
      mockScene.xr = {
        baseExperience: {
          state: BABYLON.WebXRState.IN_XR,
          camera: {
            position: { x: 1, y: 1.8, z: -2 },
            rotation: { x: 0.1, y: 0.2, z: 0 }
          }
        }
      }
      ;(BABYLON as any).WebXRState = { IN_XR: 'IN_XR' }
      
      const callback = mockScene.onBeforeRenderObservable.add.mock.calls[0][0]
      callback()
      
      expect(mockState.sendUpdate).toHaveBeenCalledWith({
        type: 'positionUpdate',
        userId: 'localUser',
        position: [1, 1.8, -2],
        rotation: [0.1, 0.2, 0]
      })
    })
    
    it('should not send updates when not connected', () => {
      multiUserManager.dispose()
      
      const callback = mockScene.onBeforeRenderObservable.add.mock.calls[0][0]
      callback()
      
      expect(mockState.sendUpdate).not.toHaveBeenCalled()
    })
  })
  
  describe('Disposal', () => {
    const mockState = {
      users: {
        'user1': { id: 'user1', position: [0, 1.6, 0] as [number, number, number], rotation: [0, 0, 0] as [number, number, number], isSelecting: true }
      },
      localUserId: 'localUser',
      sendUpdate: jest.fn()
    }
    
    it('should dispose all resources', async () => {
      await multiUserManager.initialize()
      await multiUserManager.connect(mockState)
      
      const avatar = mockMeshes.get('avatar_user1')
      const pointer = mockMeshes.get('pointer_user1')
      
      multiUserManager.dispose()
      
      expect(avatar?.dispose).toHaveBeenCalled()
      expect(pointer?.dispose).toHaveBeenCalled()
      expect(multiUserManager.isConnected()).toBe(false)
    })
  })
  
  describe('Edge Cases', () => {
    it('should handle missing position data', async () => {
      const mockState = {
        users: {
          'user1': {
            id: 'user1',
            position: undefined as any,
            rotation: [0, 0, 0] as [number, number, number],
            isSelecting: false
          }
        },
        localUserId: 'localUser',
        sendUpdate: jest.fn()
      }
      
      await multiUserManager.initialize()
      await multiUserManager.connect(mockState)
      
      const avatar = mockMeshes.get('avatar_user1')
      // Should use default position
      expect(avatar?.position).toEqual({ x: 0, y: 1.6, z: 0 })
    })
    
    it('should handle concurrent user updates', async () => {
      const initialState = {
        users: {},
        localUserId: 'localUser',
        sendUpdate: jest.fn()
      }
      
      await multiUserManager.initialize()
      await multiUserManager.connect(initialState)
      
      // Simulate rapid updates
      for (let i = 0; i < 5; i++) {
        multiUserManager.updateState({
          ...initialState,
          users: {
            [`user${i}`]: {
              id: `user${i}`,
              position: [i, 1.6, 0] as [number, number, number],
              rotation: [0, 0, 0] as [number, number, number],
              isSelecting: false
            }
          }
        })
      }
      
      // Should have created only the last user
      expect(mockMeshes.has('avatar_user4')).toBe(true)
      expect(mockMeshes.has('avatar_user0')).toBe(false)
    })
  })
})