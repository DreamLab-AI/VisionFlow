import * as THREE from 'three';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import { createLogger } from '../../../utils/loggerConfig';

const logger = createLogger('SpacePilotController');

// Type alias for OrbitControls from drei
type OrbitControls = OrbitControlsImpl;

/**
 * Calculates a quaternion that levels the horizon while preserving look direction.
 * This removes roll from the camera orientation, making the "up" vector vertical.
 */
function calculateLeveledQuaternion(
  currentQuaternion: THREE.Quaternion,
  worldUp: THREE.Vector3 = new THREE.Vector3(0, 1, 0)
): THREE.Quaternion {
  // Get current forward direction
  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(currentQuaternion).normalize();

  // Calculate the right vector perpendicular to forward and world up
  const right = new THREE.Vector3().crossVectors(forward, worldUp);

  // Handle edge case: looking straight up or down
  if (right.lengthSq() < 0.0001) {
    // When looking straight up/down, preserve current orientation
    return currentQuaternion.clone();
  }
  right.normalize();

  // Recalculate up to be perpendicular to both forward and right
  const leveledUp = new THREE.Vector3().crossVectors(right, forward).normalize();

  // Build rotation matrix from these basis vectors
  const matrix = new THREE.Matrix4();
  matrix.makeBasis(right, leveledUp, forward.negate());

  // Extract quaternion from matrix
  const leveledQuat = new THREE.Quaternion().setFromRotationMatrix(matrix);

  return leveledQuat;
}

export interface SpacePilotConfig {
  
  translationSensitivity: {
    x: number;
    y: number;
    z: number;
  };
  rotationSensitivity: {
    x: number;
    y: number;
    z: number;
  };
  
  
  deadzone: number;
  
  
  smoothing: number;
  
  
  mode: 'camera' | 'object' | 'navigation';
  
  
  invertAxes: {
    x: boolean;
    y: boolean;
    z: boolean;
    rx: boolean;
    ry: boolean;
    rz: boolean;
  };
  
  
  enabledAxes: {
    x: boolean;
    y: boolean;
    z: boolean;
    rx: boolean;
    ry: boolean;
    rz: boolean;
  };
}


export const defaultSpacePilotConfig: SpacePilotConfig = {
  translationSensitivity: { x: 1.0, y: 1.0, z: 1.0 },
  rotationSensitivity: { x: 1.0, y: 1.0, z: 1.0 },
  deadzone: 0.1,
  smoothing: 0.8,
  mode: 'camera',
  invertAxes: {
    x: false,
    y: false,
    z: false,
    rx: false,
    ry: false,
    rz: false
  },
  enabledAxes: {
    x: true,
    y: true,
    z: true,
    rx: true,
    ry: true,
    rz: true
  }
};


class SmoothingBuffer {
  private values: Map<string, number> = new Map();
  private smoothingFactor: number;

  constructor(smoothingFactor: number = 0.8) {
    this.smoothingFactor = smoothingFactor;
  }

  update(key: string, value: number): number {
    const current = this.values.get(key) || 0;
    const smoothed = current * this.smoothingFactor + value * (1 - this.smoothingFactor);
    this.values.set(key, smoothed);
    return smoothed;
  }

  reset(): void {
    this.values.clear();
  }

  setSmoothingFactor(factor: number): void {
    this.smoothingFactor = Math.max(0, Math.min(1, factor));
  }
}


export class SpacePilotController {
  private camera: THREE.Camera;
  private controls?: OrbitControls;
  private config: SpacePilotConfig;
  private smoothedValues: SmoothingBuffer;
  private isActive: boolean = false;
  private selectedObject?: THREE.Object3D;
  private animationFrameId?: number;

  // Horizon leveling transition state
  private isLevelingHorizon: boolean = false;
  private levelingStartQuat: THREE.Quaternion = new THREE.Quaternion();
  private levelingTargetQuat: THREE.Quaternion = new THREE.Quaternion();
  private levelingProgress: number = 0;
  private levelingAnimationId?: number;
  private static readonly LEVELING_DURATION = 500; // ms

  private translation = { x: 0, y: 0, z: 0 };
  private rotation = { x: 0, y: 0, z: 0 };

  // Pre-allocated reusable objects to avoid per-frame GC pressure
  private readonly _tempVec = new THREE.Vector3();
  private readonly _tempVec2 = new THREE.Vector3();
  private readonly _tempVec3 = new THREE.Vector3();
  private readonly _tempQuat = new THREE.Quaternion();
  private readonly _tempEuler = new THREE.Euler();
  private readonly _tempSpherical = new THREE.Spherical();


  private static readonly INPUT_SCALE = 1 / 32768;
  private static readonly TRANSLATION_SPEED = 0.01;
  private static readonly ROTATION_SPEED = 0.001;

  constructor(
    camera: THREE.Camera,
    config: Partial<SpacePilotConfig> = {},
    controls?: OrbitControls
  ) {
    this.camera = camera;
    this.controls = controls;
    this.config = { ...defaultSpacePilotConfig, ...config };
    this.smoothedValues = new SmoothingBuffer(this.config.smoothing);
  }

  
  start(): void {
    if (this.isActive) return;
    logger.info('Controller starting - animation loop beginning');

    // Cancel any in-progress horizon leveling when starting
    this.cancelHorizonLeveling();

    this.isActive = true;
    this.animate();
  }

  
  /**
   * Stops the controller and initiates smooth horizon leveling transition.
   * The camera maintains its position and look direction, only the roll is corrected
   * to make labels upright (level the horizon).
   */
  stop(): void {
    this.isActive = false;
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
    }
    this.smoothedValues.reset();

    // Initiate horizon leveling when stopping from navigation mode
    if (this.config.mode === 'navigation' && this.camera) {
      this.startHorizonLeveling();
    }
  }

  /**
   * Starts the smooth horizon leveling animation.
   * Preserves camera position and look direction, only adjusts roll to make "up" vertical.
   */
  private startHorizonLeveling(): void {
    if (!this.camera || this.isLevelingHorizon) return;

    logger.info('Starting horizon leveling transition');

    // Store current quaternion
    this.levelingStartQuat.copy(this.camera.quaternion);

    // Calculate target quaternion with leveled horizon
    this.levelingTargetQuat = calculateLeveledQuaternion(this.camera.quaternion);

    this.isLevelingHorizon = true;
    this.levelingProgress = 0;

    const startTime = performance.now();

    const animateLeveling = (currentTime: number) => {
      if (!this.isLevelingHorizon) return;

      const elapsed = currentTime - startTime;
      this.levelingProgress = Math.min(elapsed / SpacePilotController.LEVELING_DURATION, 1);

      // Use smooth easing (ease-out cubic)
      const eased = 1 - Math.pow(1 - this.levelingProgress, 3);

      // Slerp between start and target quaternion
      this.camera.quaternion.slerpQuaternions(
        this.levelingStartQuat,
        this.levelingTargetQuat,
        eased
      );

      if (this.levelingProgress < 1) {
        this.levelingAnimationId = requestAnimationFrame(animateLeveling);
      } else {
        this.finishHorizonLeveling();
      }
    };

    this.levelingAnimationId = requestAnimationFrame(animateLeveling);
  }

  /**
   * Completes the horizon leveling and updates OrbitControls target.
   */
  private finishHorizonLeveling(): void {
    this.isLevelingHorizon = false;
    logger.info('Horizon leveling complete');

    // Update OrbitControls target to match where camera is now looking
    if (this.controls && this.camera) {
      this._tempVec.set(0, 0, -1).applyQuaternion(this.camera.quaternion);
      // Place target at a reasonable distance in front of camera
      const targetDistance = this.camera.position.length() || 50;
      this._tempVec2.copy(this.camera.position).add(this._tempVec.multiplyScalar(targetDistance));
      this.controls.target.copy(this._tempVec2);
      this.controls.update();
    }
  }

  /**
   * Cancels any in-progress horizon leveling animation.
   */
  private cancelHorizonLeveling(): void {
    if (this.levelingAnimationId) {
      cancelAnimationFrame(this.levelingAnimationId);
      this.levelingAnimationId = undefined;
    }
    this.isLevelingHorizon = false;
  }

  
  handleTranslation(detail: { x: number; y: number; z: number }): void {
    if (!this.isActive) {
      logger.debug('Translation ignored - controller not active');
      return;
    }

    
    if (Math.abs(detail.x) > 10 || Math.abs(detail.y) > 10 || Math.abs(detail.z) > 10) {
      logger.debug('Raw translation input:', detail);
    }

    
    const normalized = {
      x: this.applyDeadzone(detail.x * SpacePilotController.INPUT_SCALE),
      y: this.applyDeadzone(detail.y * SpacePilotController.INPUT_SCALE),
      z: this.applyDeadzone(detail.z * SpacePilotController.INPUT_SCALE)
    };

    
    this.translation = {
      x: normalized.x * this.config.translationSensitivity.x * (this.config.invertAxes.x ? -1 : 1),
      y: normalized.y * this.config.translationSensitivity.y * (this.config.invertAxes.y ? -1 : 1),
      z: normalized.z * this.config.translationSensitivity.z * (this.config.invertAxes.z ? -1 : 1)
    };

    
    if (this.config.smoothing > 0) {
      this.translation.x = this.smoothedValues.update('tx', this.translation.x);
      this.translation.y = this.smoothedValues.update('ty', this.translation.y);
      this.translation.z = this.smoothedValues.update('tz', this.translation.z);
    }
  }

  
  handleRotation(detail: { rx: number; ry: number; rz: number }): void {
    if (!this.isActive) return;

    
    const normalized = {
      x: this.applyDeadzone(detail.rx * SpacePilotController.INPUT_SCALE),
      y: this.applyDeadzone(detail.ry * SpacePilotController.INPUT_SCALE),
      z: this.applyDeadzone(detail.rz * SpacePilotController.INPUT_SCALE)
    };

    
    this.rotation = {
      x: normalized.x * this.config.rotationSensitivity.x * (this.config.invertAxes.rx ? -1 : 1),
      y: normalized.y * this.config.rotationSensitivity.y * (this.config.invertAxes.ry ? -1 : 1),
      z: normalized.z * this.config.rotationSensitivity.z * (this.config.invertAxes.rz ? -1 : 1)
    };

    
    if (this.config.smoothing > 0) {
      this.rotation.x = this.smoothedValues.update('rx', this.rotation.x);
      this.rotation.y = this.smoothedValues.update('ry', this.rotation.y);
      this.rotation.z = this.smoothedValues.update('rz', this.rotation.z);
    }
  }

  
  handleButtons(detail: { buttons: string[] }): void {
    
    
    detail.buttons.forEach(button => {
      this.handleButton(button);
    });
  }

  
  updateConfig(config: Partial<SpacePilotConfig>): void {
    this.config = { ...this.config, ...config };
    this.smoothedValues.setSmoothingFactor(this.config.smoothing);
  }

  
  /**
   * Changes the control mode. When transitioning FROM navigation mode,
   * triggers horizon leveling to smoothly correct any camera roll.
   */
  setMode(mode: 'camera' | 'object' | 'navigation'): void {
    const wasNavigation = this.config.mode === 'navigation';
    this.config.mode = mode;
    this.smoothedValues.reset();

    // If leaving navigation mode, level the horizon
    if (wasNavigation && mode !== 'navigation' && this.camera) {
      this.startHorizonLeveling();
    }

    // If entering a new mode, cancel any pending horizon leveling
    if (mode === 'navigation') {
      this.cancelHorizonLeveling();
    }
  }

  
  setSelectedObject(object?: THREE.Object3D): void {
    this.selectedObject = object;
  }

  
  private applyDeadzone(value: number): number {
    return Math.abs(value) < this.config.deadzone ? 0 : value;
  }

  
  private animate = (): void => {
    if (!this.isActive) return;

    switch (this.config.mode) {
      case 'camera':
        this.updateCamera();
        break;
      case 'object':
        this.updateObject();
        break;
      case 'navigation':
        this.updateNavigation();
        break;
    }

    this.animationFrameId = requestAnimationFrame(this.animate);
  };

  
  private updateCamera(): void {
    if (!this.camera) return;


    if (this.config.enabledAxes.x || this.config.enabledAxes.y || this.config.enabledAxes.z) {
      this._tempVec.set(
        this.config.enabledAxes.x ? this.translation.x * SpacePilotController.TRANSLATION_SPEED : 0,
        this.config.enabledAxes.y ? this.translation.y * SpacePilotController.TRANSLATION_SPEED : 0,
        this.config.enabledAxes.z ? -this.translation.z * SpacePilotController.TRANSLATION_SPEED : 0
      );


      if (this._tempVec.length() > 0.0001) {
        logger.debug('Camera translation:', this._tempVec);
      }


      this._tempVec.applyQuaternion(this.camera.quaternion);
      this.camera.position.add(this._tempVec);
    }


    if (this.controls && (this.config.enabledAxes.rx || this.config.enabledAxes.ry)) {

      this._tempSpherical.setFromVector3(this._tempVec.copy(this.camera.position).sub(this.controls.target));

      if (this.config.enabledAxes.ry) {
        this._tempSpherical.theta -= this.rotation.y * SpacePilotController.ROTATION_SPEED;
      }
      if (this.config.enabledAxes.rx) {
        this._tempSpherical.phi += this.rotation.x * SpacePilotController.ROTATION_SPEED;
        this._tempSpherical.phi = Math.max(0.01, Math.min(Math.PI - 0.01, this._tempSpherical.phi));
      }

      this.camera.position.setFromSpherical(this._tempSpherical).add(this.controls.target);
      this.camera.lookAt(this.controls.target);
    } else if (!this.controls) {

      this._tempEuler.set(
        this.config.enabledAxes.rx ? this.rotation.x * SpacePilotController.ROTATION_SPEED : 0,
        this.config.enabledAxes.ry ? this.rotation.y * SpacePilotController.ROTATION_SPEED : 0,
        this.config.enabledAxes.rz ? this.rotation.z * SpacePilotController.ROTATION_SPEED : 0,
        'YXZ'
      );
      this.camera.quaternion.multiply(this._tempQuat.setFromEuler(this._tempEuler));
    }
  }

  
  private updateObject(): void {
    if (!this.selectedObject) return;


    if (this.config.enabledAxes.x || this.config.enabledAxes.y || this.config.enabledAxes.z) {
      this._tempVec.set(
        this.config.enabledAxes.x ? this.translation.x * SpacePilotController.TRANSLATION_SPEED : 0,
        this.config.enabledAxes.y ? this.translation.y * SpacePilotController.TRANSLATION_SPEED : 0,
        this.config.enabledAxes.z ? this.translation.z * SpacePilotController.TRANSLATION_SPEED : 0
      );

      this.selectedObject.position.add(this._tempVec);
    }


    this._tempEuler.set(
      this.config.enabledAxes.rx ? this.rotation.x * SpacePilotController.ROTATION_SPEED : 0,
      this.config.enabledAxes.ry ? this.rotation.y * SpacePilotController.ROTATION_SPEED : 0,
      this.config.enabledAxes.rz ? this.rotation.z * SpacePilotController.ROTATION_SPEED : 0,
      'XYZ'
    );

    this.selectedObject.quaternion.multiply(this._tempQuat.setFromEuler(this._tempEuler));
  }

  
  private updateNavigation(): void {
    if (!this.camera) return;


    this._tempVec.set(0, 0, -1);
    this._tempVec.applyQuaternion(this.camera.quaternion);
    this._tempVec.multiplyScalar(this.translation.z * SpacePilotController.TRANSLATION_SPEED * 2);


    this._tempVec2.set(1, 0, 0);
    this._tempVec2.applyQuaternion(this.camera.quaternion);
    this._tempVec2.multiplyScalar(this.translation.x * SpacePilotController.TRANSLATION_SPEED * 2);


    this._tempVec3.set(0, 1, 0);
    this._tempVec3.multiplyScalar(this.translation.y * SpacePilotController.TRANSLATION_SPEED * 2);


    this.camera.position.add(this._tempVec);
    this.camera.position.add(this._tempVec2);
    this.camera.position.add(this._tempVec3);


    this._tempEuler.set(
      this.rotation.x * SpacePilotController.ROTATION_SPEED * 2,
      this.rotation.y * SpacePilotController.ROTATION_SPEED * 2,
      this.rotation.z * SpacePilotController.ROTATION_SPEED * 2,
      'YXZ'
    );

    this.camera.quaternion.multiply(this._tempQuat.setFromEuler(this._tempEuler));
  }

  
  private handleButton(button: string): void {
    
    switch (button) {
      case '[1]':
        
        this.resetView();
        break;
      case '[2]':
        
        const modes: Array<'camera' | 'object' | 'navigation'> = ['camera', 'object', 'navigation'];
        const currentIndex = modes.indexOf(this.config.mode);
        this.setMode(modes[(currentIndex + 1) % modes.length]);
        break;
      
    }
  }

  
  private resetView(): void {
    if (this.controls) {
      this.controls.reset();
    } else if (this.camera) {
      this.camera.position.set(0, 10, 20);
      this.camera.lookAt(0, 0, 0);
    }
  }

  /**
   * Public method to trigger horizon leveling.
   * Useful when the user wants to level the view without a full reset.
   */
  public levelHorizon(): void {
    if (this.camera) {
      this.startHorizonLeveling();
    }
  }

  /**
   * Smoothly transitions the camera to OrbitControls mode.
   * Levels the horizon and updates the OrbitControls target to match current view.
   */
  public transitionToOrbitMode(): void {
    if (!this.camera) return;

    // Stop any active 6DOF control
    this.isActive = false;
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
    }
    this.smoothedValues.reset();

    // Level the horizon with smooth transition
    this.startHorizonLeveling();
  }
}