import { useState, useCallback, useRef } from 'react'
import * as BABYLON from '@babylonjs/core'
import '@babylonjs/core/XR'
import { createLogger } from '../utils/logger'

const logger = createLogger('useVircadiaXR')

interface XRState {
  isSupported: boolean
  isInXR: boolean
  sessionMode: 'immersive-vr' | 'immersive-ar' | null
  error: string | null
}

export const useVircadiaXR = () => {
  const [xrState, setXRState] = useState<XRState>({
    isSupported: false,
    isInXR: false,
    sessionMode: null,
    error: null
  })
  
  const xrHelperRef = useRef<BABYLON.WebXRDefaultExperience | null>(null)
  const xrSessionRef = useRef<XRSession | null>(null)
  
  // Check if WebXR is supported
  const checkXRSupport = useCallback(async () => {
    try {
      if (!navigator.xr) {
        setXRState(prev => ({ ...prev, isSupported: false }))
        return false
      }
      
      // Check for VR support
      const vrSupported = await navigator.xr.isSessionSupported('immersive-vr')
      // Check for AR support (Quest 3 passthrough)
      const arSupported = await navigator.xr.isSessionSupported('immersive-ar')
      
      const isSupported = vrSupported || arSupported
      setXRState(prev => ({ ...prev, isSupported }))
      
      logger.info(`WebXR support - VR: ${vrSupported}, AR: ${arSupported}`)
      return isSupported
    } catch (err) {
      logger.error('Failed to check XR support:', err)
      setXRState(prev => ({ ...prev, isSupported: false, error: 'XR not supported' }))
      return false
    }
  }, [])
  
  // Setup XR for Babylon scene
  const setupXR = useCallback(async (scene: BABYLON.Scene, engine: BABYLON.Engine) => {
    try {
      // Check support first
      const supported = await checkXRSupport()
      if (!supported) {
        logger.warn('WebXR not supported on this device')
        return null
      }
      
      // Create XR helper
      const xrHelper = await scene.createDefaultXRExperienceAsync({
        uiOptions: {
          sessionMode: 'immersive-ar', // Prefer AR for Quest 3 passthrough
          referenceSpaceType: 'local-floor'
        },
        optionalFeatures: true
      })
      
      xrHelperRef.current = xrHelper
      
      // Configure for Quest 3 optimizations
      xrHelper.baseExperience.onStateChangedObservable.add((state) => {
        if (state === BABYLON.WebXRState.IN_XR) {
          // Enable foveated rendering for performance
          const xrSession = xrHelper.baseExperience.sessionManager.session
          if (xrSession) {
            xrSessionRef.current = xrSession
            
            // Quest 3 specific optimizations
            if ('updateRenderState' in xrSession) {
              xrSession.updateRenderState({
                depthNear: 0.1,
                depthFar: 1000.0
              })
            }
          }
          
          setXRState(prev => ({ 
            ...prev, 
            isInXR: true, 
            sessionMode: xrHelper.baseExperience.sessionManager.sessionMode as any 
          }))
          
          logger.info('Entered XR mode')
        } else if (state === BABYLON.WebXRState.NOT_IN_XR) {
          setXRState(prev => ({ ...prev, isInXR: false, sessionMode: null }))
          xrSessionRef.current = null
          logger.info('Exited XR mode')
        }
      })
      
      // Setup teleportation
      const teleportation = xrHelper.teleportation
      if (teleportation) {
        teleportation.parabolicRayEnabled = true
        teleportation.straightRayEnabled = false
        teleportation.rotationEnabled = true
      }
      
      // Setup pointer selection
      xrHelper.pointerSelection.displayLaserPointer = true
      xrHelper.pointerSelection.displaySelectionMesh = true
      
      // Add hand tracking support (for Quest 3)
      try {
        const handTracking = xrHelper.baseExperience.featuresManager.enableFeature(
          BABYLON.WebXRFeatureName.HAND_TRACKING,
          'latest',
          {
            xrInput: xrHelper.input,
            jointMeshes: {
              enablePhysics: true
            }
          }
        ) as BABYLON.WebXRHandTracking
        
        logger.info('Hand tracking enabled')
      } catch (err) {
        logger.warn('Hand tracking not available:', err)
      }
      
      // Add controller haptics
      xrHelper.input.onControllerAddedObservable.add((controller) => {
        controller.onMotionControllerInitObservable.add((motionController) => {
          if (motionController.handness === 'right') {
            const triggerComponent = motionController.getComponent('xr-standard-trigger')
            if (triggerComponent) {
              triggerComponent.onButtonStateChangedObservable.add(() => {
                if (triggerComponent.pressed && controller.gamepadController.hapticActuators) {
                  controller.gamepadController.hapticActuators[0]?.pulse(0.1, 50)
                }
              })
            }
          }
        })
      })
      
      logger.info('XR setup completed successfully')
      return xrHelper
      
    } catch (err) {
      logger.error('Failed to setup XR:', err)
      setXRState(prev => ({ 
        ...prev, 
        error: err instanceof Error ? err.message : 'Failed to setup XR' 
      }))
      return null
    }
  }, [checkXRSupport])
  
  // Enter XR mode
  const enterXR = useCallback(async (scene: BABYLON.Scene, engine: BABYLON.Engine) => {
    try {
      if (!xrHelperRef.current) {
        logger.warn('XR not initialized, setting up...')
        const helper = await setupXR(scene, engine)
        if (!helper) {
          throw new Error('Failed to initialize XR')
        }
      }
      
      if (xrHelperRef.current) {
        await xrHelperRef.current.baseExperience.enterXRAsync(
          'immersive-ar', 
          'local-floor'
        )
      }
    } catch (err) {
      logger.error('Failed to enter XR:', err)
      setXRState(prev => ({ 
        ...prev, 
        error: err instanceof Error ? err.message : 'Failed to enter XR' 
      }))
      throw err
    }
  }, [setupXR])
  
  // Exit XR mode
  const exitXR = useCallback(async () => {
    try {
      if (xrHelperRef.current && xrState.isInXR) {
        await xrHelperRef.current.baseExperience.exitXRAsync()
      }
    } catch (err) {
      logger.error('Failed to exit XR:', err)
      setXRState(prev => ({ 
        ...prev, 
        error: err instanceof Error ? err.message : 'Failed to exit XR' 
      }))
      throw err
    }
  }, [xrState.isInXR])
  
  // Get current XR camera position
  const getXRCameraPosition = useCallback(() => {
    if (!xrHelperRef.current || !xrState.isInXR) {
      return null
    }
    
    const camera = xrHelperRef.current.baseExperience.camera
    return camera ? camera.position.clone() : null
  }, [xrState.isInXR])
  
  // Get controller positions
  const getControllerPositions = useCallback(() => {
    if (!xrHelperRef.current || !xrState.isInXR) {
      return { left: null, right: null }
    }
    
    const controllers = xrHelperRef.current.input.controllers
    const positions: { left: BABYLON.Vector3 | null, right: BABYLON.Vector3 | null } = {
      left: null,
      right: null
    }
    
    controllers.forEach(controller => {
      if (controller.inputSource.handedness === 'left') {
        positions.left = controller.pointer.position.clone()
      } else if (controller.inputSource.handedness === 'right') {
        positions.right = controller.pointer.position.clone()
      }
    })
    
    return positions
  }, [xrState.isInXR])
  
  // Trigger haptic feedback
  const triggerHaptics = useCallback((
    hand: 'left' | 'right', 
    intensity: number = 0.5, 
    duration: number = 100
  ) => {
    if (!xrHelperRef.current || !xrState.isInXR) return
    
    const controllers = xrHelperRef.current.input.controllers
    const controller = controllers.find(c => c.inputSource.handedness === hand)
    
    if (controller?.gamepadController.hapticActuators) {
      controller.gamepadController.hapticActuators[0]?.pulse(intensity, duration)
    }
  }, [xrState.isInXR])
  
  return {
    // State
    isXRSupported: xrState.isSupported,
    isInXR: xrState.isInXR,
    xrSessionMode: xrState.sessionMode,
    xrError: xrState.error,
    
    // Actions
    checkXRSupport,
    setupXR,
    enterXR,
    exitXR,
    
    // Utilities
    getXRCameraPosition,
    getControllerPositions,
    triggerHaptics
  }
}