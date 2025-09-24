import * as BABYLON from '@babylonjs/core'
import { createLogger } from '../../utils/loggerConfig'

const logger = createLogger('MultiUserManager')

interface User {
  id: string
  position: [number, number, number]
  rotation: [number, number, number]
  isSelecting: boolean
  selectedNodes?: string[]
  color?: string
}

interface MultiUserState {
  users: Record<string, User>
  localUserId: string
  sendUpdate: (data: any) => void
}

export class MultiUserManager {
  private scene: BABYLON.Scene
  private userAvatars: Map<string, BABYLON.Mesh> = new Map()
  private userPointers: Map<string, BABYLON.Mesh> = new Map()
  private localUserId: string = ''
  private updateCallback?: (data: any) => void
  private connected = false
  private userColors: string[] = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
    '#FF8B94', '#A8E6CF', '#FFD93D', '#6BCF7F', '#C490E4'
  ]
  private userColorMap: Map<string, string> = new Map()
  
  constructor(scene: BABYLON.Scene) {
    this.scene = scene
  }
  
  async initialize(): Promise<void> {
    logger.info('Initializing MultiUserManager')
    
    // Create shared parent for avatars
    const avatarContainer = new BABYLON.TransformNode('avatarContainer', this.scene)
    
    // Setup pointer ray material
    this.createPointerMaterial()
    
    logger.info('MultiUserManager initialized')
  }
  
  async connect(multiUserState: MultiUserState): Promise<void> {
    this.localUserId = multiUserState.localUserId
    this.updateCallback = multiUserState.sendUpdate
    
    // Process initial users
    this.updateState(multiUserState)
    
    // Start sending local user updates
    this.startLocalTracking()
    
    this.connected = true
    logger.info('Connected to multi-user session')
  }
  
  updateState(multiUserState: MultiUserState): void {
    const { users, localUserId } = multiUserState
    
    // Update local user ID if changed
    if (localUserId !== this.localUserId) {
      this.localUserId = localUserId
    }
    
    // Get current user IDs
    const currentUserIds = new Set(this.userAvatars.keys())
    const newUserIds = new Set(Object.keys(users).filter(id => id !== this.localUserId))
    
    // Remove users who left
    for (const userId of currentUserIds) {
      if (!newUserIds.has(userId)) {
        this.removeUser(userId)
      }
    }
    
    // Add or update users
    for (const [userId, userData] of Object.entries(users)) {
      if (userId === this.localUserId) continue // Skip local user
      
      if (this.userAvatars.has(userId)) {
        this.updateUser(userId, userData)
      } else {
        this.addUser(userId, userData)
      }
    }
  }
  
  private addUser(userId: string, userData: User): void {
    logger.info(`Adding user: ${userId}`)
    
    // Assign color
    const colorIndex = this.userColorMap.size % this.userColors.length
    const color = userData.color || this.userColors[colorIndex]
    this.userColorMap.set(userId, color)
    
    // Create avatar (head representation)
    const avatar = BABYLON.MeshBuilder.CreateSphere(
      `avatar_${userId}`,
      { diameter: 0.2, segments: 16 },
      this.scene
    )
    
    // Avatar material
    const avatarMaterial = new BABYLON.StandardMaterial(`avatarMat_${userId}`, this.scene)
    avatarMaterial.diffuseColor = BABYLON.Color3.FromHexString(color)
    avatarMaterial.emissiveColor = BABYLON.Color3.FromHexString(color).scale(0.3)
    avatarMaterial.specularPower = 64
    avatar.material = avatarMaterial
    
    // Position avatar
    const [x, y, z] = userData.position || [0, 1.6, 0]
    avatar.position = new BABYLON.Vector3(x, y, z)
    
    // Add to map
    this.userAvatars.set(userId, avatar)
    
    // Create selection pointer/ray
    if (userData.isSelecting) {
      this.createUserPointer(userId, avatar.position, color)
    }
    
    // Add name label
    this.createNameLabel(userId, avatar)
  }
  
  private updateUser(userId: string, userData: User): void {
    const avatar = this.userAvatars.get(userId)
    if (!avatar) return
    
    // Update position with smooth interpolation
    const targetPosition = new BABYLON.Vector3(...(userData.position || [0, 1.6, 0]))
    avatar.position = BABYLON.Vector3.Lerp(avatar.position, targetPosition, 0.1)
    
    // Update rotation
    if (userData.rotation) {
      const [rx, ry, rz] = userData.rotation
      avatar.rotation = new BABYLON.Vector3(rx, ry, rz)
    }
    
    // Update selection state
    if (userData.isSelecting && !this.userPointers.has(userId)) {
      const color = this.userColorMap.get(userId) || '#4ECDC4'
      this.createUserPointer(userId, avatar.position, color)
    } else if (!userData.isSelecting && this.userPointers.has(userId)) {
      this.removeUserPointer(userId)
    }
    
    // Update pointer position if selecting
    if (userData.isSelecting) {
      this.updateUserPointer(userId, avatar.position)
    }
  }
  
  private removeUser(userId: string): void {
    logger.info(`Removing user: ${userId}`)
    
    // Remove avatar
    const avatar = this.userAvatars.get(userId)
    if (avatar) {
      avatar.dispose()
      this.userAvatars.delete(userId)
    }
    
    // Remove pointer
    this.removeUserPointer(userId)
    
    // Clean up color mapping
    this.userColorMap.delete(userId)
  }
  
  private createUserPointer(userId: string, origin: BABYLON.Vector3, color: string): void {
    // Create ray/pointer for selection
    const pointer = BABYLON.MeshBuilder.CreateCylinder(
      `pointer_${userId}`,
      { height: 2, diameterBottom: 0.01, diameterTop: 0.05 },
      this.scene
    )
    
    // Pointer material
    const pointerMat = new BABYLON.StandardMaterial(`pointerMat_${userId}`, this.scene)
    pointerMat.diffuseColor = BABYLON.Color3.FromHexString(color)
    pointerMat.emissiveColor = BABYLON.Color3.FromHexString(color).scale(0.5)
    pointerMat.alpha = 0.8
    pointer.material = pointerMat
    
    // Position pointer
    pointer.position = origin.clone()
    pointer.position.y -= 0.5
    pointer.rotation.x = Math.PI
    
    this.userPointers.set(userId, pointer)
  }
  
  private updateUserPointer(userId: string, origin: BABYLON.Vector3): void {
    const pointer = this.userPointers.get(userId)
    if (!pointer) return
    
    pointer.position = origin.clone()
    pointer.position.y -= 0.5
  }
  
  private removeUserPointer(userId: string): void {
    const pointer = this.userPointers.get(userId)
    if (pointer) {
      pointer.dispose()
      this.userPointers.delete(userId)
    }
  }
  
  private createNameLabel(userId: string, avatar: BABYLON.Mesh): void {
    // Create dynamic texture for name
    const texture = new BABYLON.DynamicTexture(
      `nameTexture_${userId}`,
      { width: 256, height: 64 },
      this.scene,
      false
    )
    
    const context = texture.getContext()
    context.font = '32px Arial'
    context.fillStyle = 'white'
    context.textAlign = 'center'
    context.textBaseline = 'middle'
    context.fillText(userId.substring(0, 10), 128, 32)
    texture.update()
    
    // Create plane for label
    const label = BABYLON.MeshBuilder.CreatePlane(
      `label_${userId}`,
      { width: 0.5, height: 0.125 },
      this.scene
    )
    
    const labelMat = new BABYLON.StandardMaterial(`labelMat_${userId}`, this.scene)
    labelMat.diffuseTexture = texture
    labelMat.emissiveTexture = texture
    labelMat.backFaceCulling = false
    labelMat.disableLighting = true
    label.material = labelMat
    
    // Position above avatar
    label.parent = avatar
    label.position.y = 0.2
    label.billboardMode = BABYLON.Mesh.BILLBOARDMODE_Y
  }
  
  private createPointerMaterial(): void {
    // Shared material for selection rays
    const pointerMat = new BABYLON.StandardMaterial('sharedPointerMat', this.scene)
    pointerMat.emissiveColor = new BABYLON.Color3(1, 1, 1)
    pointerMat.disableLighting = true
  }
  
  private startLocalTracking(): void {
    // Track local user camera/controller position
    this.scene.onBeforeRenderObservable.add(() => {
      if (!this.connected || !this.updateCallback) return
      
      const camera = this.scene.activeCamera
      if (!camera) return
      
      // Get XR camera position if in XR
      let position = camera.position
      let rotation = camera.rotation
      
      // Check if we're in XR mode
      const xr = this.scene.xr
      if (xr && xr.baseExperience && xr.baseExperience.state === BABYLON.WebXRState.IN_XR) {
        const xrCamera = xr.baseExperience.camera
        if (xrCamera) {
          position = xrCamera.position
          rotation = xrCamera.rotation
        }
      }
      
      // Send update
      this.updateCallback({
        type: 'positionUpdate',
        userId: this.localUserId,
        position: [position.x, position.y, position.z],
        rotation: [rotation.x, rotation.y, rotation.z]
      })
    })
  }
  
  isConnected(): boolean {
    return this.connected
  }
  
  dispose(): void {
    logger.info('Disposing MultiUserManager')
    
    // Remove all users
    this.userAvatars.forEach((_, userId) => this.removeUser(userId))
    
    this.userAvatars.clear()
    this.userPointers.clear()
    this.userColorMap.clear()
    this.connected = false
  }
}