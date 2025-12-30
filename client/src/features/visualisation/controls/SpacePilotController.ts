import * as THREE from 'three';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';

// Type alias for OrbitControls from drei
type OrbitControls = OrbitControlsImpl;


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
  
  
  private translation = { x: 0, y: 0, z: 0 };
  private rotation = { x: 0, y: 0, z: 0 };
  
  
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
    console.log('[SpacePilot] Controller starting - animation loop beginning');
    this.isActive = true;
    this.animate();
  }

  
  stop(): void {
    this.isActive = false;
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
    }
    this.smoothedValues.reset();
  }

  
  handleTranslation(detail: { x: number; y: number; z: number }): void {
    if (!this.isActive) {
      console.log('[SpacePilot] Translation ignored - controller not active');
      return;
    }

    
    if (Math.abs(detail.x) > 10 || Math.abs(detail.y) > 10 || Math.abs(detail.z) > 10) {
      console.log('[SpacePilot] Raw translation input:', detail);
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

  
  setMode(mode: 'camera' | 'object' | 'navigation'): void {
    this.config.mode = mode;
    this.smoothedValues.reset();
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
      const translationVector = new THREE.Vector3(
        this.config.enabledAxes.x ? this.translation.x * SpacePilotController.TRANSLATION_SPEED : 0,
        this.config.enabledAxes.y ? this.translation.y * SpacePilotController.TRANSLATION_SPEED : 0,
        this.config.enabledAxes.z ? -this.translation.z * SpacePilotController.TRANSLATION_SPEED : 0
      );

      
      if (translationVector.length() > 0.0001) {
        console.log('[SpacePilot] Camera translation:', translationVector);
      }

      
      translationVector.applyQuaternion(this.camera.quaternion);
      this.camera.position.add(translationVector);
    }

    
    if (this.controls && (this.config.enabledAxes.rx || this.config.enabledAxes.ry)) {
      
      const spherical = new THREE.Spherical();
      spherical.setFromVector3(this.camera.position.clone().sub(this.controls.target));
      
      if (this.config.enabledAxes.ry) {
        spherical.theta -= this.rotation.y * SpacePilotController.ROTATION_SPEED;
      }
      if (this.config.enabledAxes.rx) {
        spherical.phi += this.rotation.x * SpacePilotController.ROTATION_SPEED;
        spherical.phi = Math.max(0.01, Math.min(Math.PI - 0.01, spherical.phi));
      }
      
      this.camera.position.setFromSpherical(spherical).add(this.controls.target);
      this.camera.lookAt(this.controls.target);
    } else if (!this.controls) {
      
      const euler = new THREE.Euler(
        this.config.enabledAxes.rx ? this.rotation.x * SpacePilotController.ROTATION_SPEED : 0,
        this.config.enabledAxes.ry ? this.rotation.y * SpacePilotController.ROTATION_SPEED : 0,
        this.config.enabledAxes.rz ? this.rotation.z * SpacePilotController.ROTATION_SPEED : 0,
        'YXZ'
      );
      this.camera.quaternion.multiply(new THREE.Quaternion().setFromEuler(euler));
    }
  }

  
  private updateObject(): void {
    if (!this.selectedObject) return;

    
    if (this.config.enabledAxes.x || this.config.enabledAxes.y || this.config.enabledAxes.z) {
      const translationVector = new THREE.Vector3(
        this.config.enabledAxes.x ? this.translation.x * SpacePilotController.TRANSLATION_SPEED : 0,
        this.config.enabledAxes.y ? this.translation.y * SpacePilotController.TRANSLATION_SPEED : 0,
        this.config.enabledAxes.z ? this.translation.z * SpacePilotController.TRANSLATION_SPEED : 0
      );
      
      this.selectedObject.position.add(translationVector);
    }

    
    const euler = new THREE.Euler(
      this.config.enabledAxes.rx ? this.rotation.x * SpacePilotController.ROTATION_SPEED : 0,
      this.config.enabledAxes.ry ? this.rotation.y * SpacePilotController.ROTATION_SPEED : 0,
      this.config.enabledAxes.rz ? this.rotation.z * SpacePilotController.ROTATION_SPEED : 0,
      'XYZ'
    );
    
    this.selectedObject.quaternion.multiply(new THREE.Quaternion().setFromEuler(euler));
  }

  
  private updateNavigation(): void {
    if (!this.camera) return;

    
    const forward = new THREE.Vector3(0, 0, -1);
    forward.applyQuaternion(this.camera.quaternion);
    forward.multiplyScalar(this.translation.z * SpacePilotController.TRANSLATION_SPEED * 2);
    
    
    const right = new THREE.Vector3(1, 0, 0);
    right.applyQuaternion(this.camera.quaternion);
    right.multiplyScalar(this.translation.x * SpacePilotController.TRANSLATION_SPEED * 2);
    
    
    const up = new THREE.Vector3(0, 1, 0);
    up.multiplyScalar(this.translation.y * SpacePilotController.TRANSLATION_SPEED * 2);
    
    
    this.camera.position.add(forward);
    this.camera.position.add(right);
    this.camera.position.add(up);
    
    
    const euler = new THREE.Euler(
      this.rotation.x * SpacePilotController.ROTATION_SPEED * 2,
      this.rotation.y * SpacePilotController.ROTATION_SPEED * 2,
      this.rotation.z * SpacePilotController.ROTATION_SPEED * 2,
      'YXZ'
    );
    
    this.camera.quaternion.multiply(new THREE.Quaternion().setFromEuler(euler));
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
}