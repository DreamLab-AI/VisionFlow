import * as THREE from 'three';
import { createLogger } from '../../../utils/loggerConfig';

const logger = createLogger('InteractionManager');

export type InputType = 'mouse' | 'touch' | 'xr-controller';

export interface InteractionEvent {
  type: 'select' | 'drag' | 'release' | 'hover';
  inputType: InputType;
  nodeId?: string;
  position?: THREE.Vector3;
  screenPosition?: { x: number; y: number };
  controllerHandedness?: 'left' | 'right';
  timestamp: number;
}

export interface InteractionHandler {
  onSelect?: (event: InteractionEvent) => void;
  onDrag?: (event: InteractionEvent) => void;
  onRelease?: (event: InteractionEvent) => void;
  onHover?: (event: InteractionEvent) => void;
}

/**
 * Unified interaction manager for mouse, touch, and XR controllers
 */
export class InteractionManager {
  private handlers: InteractionHandler = {};
  private currentInteraction: InteractionEvent | null = null;
  private raycaster = new THREE.Raycaster();

  constructor(handlers?: InteractionHandler) {
    if (handlers) {
      this.handlers = handlers;
    }
  }

  setHandlers(handlers: InteractionHandler) {
    this.handlers = { ...this.handlers, ...handlers };
  }

  // Mouse events
  handleMouseDown(event: MouseEvent, camera: THREE.Camera, nodes: Array<{ id: string; position: THREE.Vector3 }>) {
    const interaction = this.createInteractionFromMouse(event, camera, nodes, 'select');
    if (interaction) {
      this.currentInteraction = interaction;
      this.handlers.onSelect?.(interaction);
    }
  }

  handleMouseMove(event: MouseEvent, camera: THREE.Camera, nodes: Array<{ id: string; position: THREE.Vector3 }>) {
    if (this.currentInteraction) {
      const interaction = this.createInteractionFromMouse(event, camera, nodes, 'drag');
      if (interaction) {
        this.handlers.onDrag?.(interaction);
      }
    } else {
      const interaction = this.createInteractionFromMouse(event, camera, nodes, 'hover');
      if (interaction) {
        this.handlers.onHover?.(interaction);
      }
    }
  }

  handleMouseUp(event: MouseEvent) {
    if (this.currentInteraction) {
      const interaction: InteractionEvent = {
        ...this.currentInteraction,
        type: 'release',
        timestamp: Date.now()
      };
      this.handlers.onRelease?.(interaction);
      this.currentInteraction = null;
    }
  }

  // Touch events
  handleTouchStart(event: TouchEvent, camera: THREE.Camera, nodes: Array<{ id: string; position: THREE.Vector3 }>) {
    const touch = event.touches[0];
    if (!touch) return;

    const interaction = this.createInteractionFromTouch(touch, camera, nodes, 'select');
    if (interaction) {
      this.currentInteraction = interaction;
      this.handlers.onSelect?.(interaction);
    }
  }

  handleTouchMove(event: TouchEvent, camera: THREE.Camera, nodes: Array<{ id: string; position: THREE.Vector3 }>) {
    const touch = event.touches[0];
    if (!touch || !this.currentInteraction) return;

    const interaction = this.createInteractionFromTouch(touch, camera, nodes, 'drag');
    if (interaction) {
      this.handlers.onDrag?.(interaction);
    }
  }

  handleTouchEnd(event: TouchEvent) {
    if (this.currentInteraction) {
      const interaction: InteractionEvent = {
        ...this.currentInteraction,
        type: 'release',
        timestamp: Date.now()
      };
      this.handlers.onRelease?.(interaction);
      this.currentInteraction = null;
    }
  }

  // XR controller events
  handleXRSelect(nodeId: string, position: THREE.Vector3, handedness: 'left' | 'right') {
    const interaction: InteractionEvent = {
      type: 'select',
      inputType: 'xr-controller',
      nodeId,
      position,
      controllerHandedness: handedness,
      timestamp: Date.now()
    };
    this.currentInteraction = interaction;
    this.handlers.onSelect?.(interaction);
  }

  handleXRDrag(nodeId: string, position: THREE.Vector3, handedness: 'left' | 'right') {
    if (!this.currentInteraction) return;

    const interaction: InteractionEvent = {
      type: 'drag',
      inputType: 'xr-controller',
      nodeId,
      position,
      controllerHandedness: handedness,
      timestamp: Date.now()
    };
    this.handlers.onDrag?.(interaction);
  }

  handleXRRelease(nodeId: string, handedness: 'left' | 'right') {
    if (!this.currentInteraction) return;

    const interaction: InteractionEvent = {
      type: 'release',
      inputType: 'xr-controller',
      nodeId,
      controllerHandedness: handedness,
      timestamp: Date.now()
    };
    this.handlers.onRelease?.(interaction);
    this.currentInteraction = null;
  }

  // Helper methods
  private createInteractionFromMouse(
    event: MouseEvent,
    camera: THREE.Camera,
    nodes: Array<{ id: string; position: THREE.Vector3 }>,
    type: 'select' | 'drag' | 'hover'
  ): InteractionEvent | null {
    const rect = (event.target as HTMLElement).getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    const intersectedNode = this.raycastNodes(camera, x, y, nodes);

    return {
      type,
      inputType: 'mouse',
      nodeId: intersectedNode?.id,
      screenPosition: { x: event.clientX, y: event.clientY },
      position: intersectedNode?.position,
      timestamp: Date.now()
    };
  }

  private createInteractionFromTouch(
    touch: Touch,
    camera: THREE.Camera,
    nodes: Array<{ id: string; position: THREE.Vector3 }>,
    type: 'select' | 'drag'
  ): InteractionEvent | null {
    const rect = (touch.target as HTMLElement).getBoundingClientRect();
    const x = ((touch.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((touch.clientY - rect.top) / rect.height) * 2 + 1;

    const intersectedNode = this.raycastNodes(camera, x, y, nodes);

    return {
      type,
      inputType: 'touch',
      nodeId: intersectedNode?.id,
      screenPosition: { x: touch.clientX, y: touch.clientY },
      position: intersectedNode?.position,
      timestamp: Date.now()
    };
  }

  private raycastNodes(
    camera: THREE.Camera,
    x: number,
    y: number,
    nodes: Array<{ id: string; position: THREE.Vector3 }>
  ): { id: string; position: THREE.Vector3 } | null {
    this.raycaster.setFromCamera(new THREE.Vector2(x, y), camera);

    let closestNode: { id: string; position: THREE.Vector3 } | null = null;
    let minDistance = Infinity;

    nodes.forEach(node => {
      const sphere = new THREE.Sphere(node.position, 0.5);
      const intersectionPoint = new THREE.Vector3();

      if (this.raycaster.ray.intersectSphere(sphere, intersectionPoint)) {
        const distance = camera.position.distanceTo(intersectionPoint);
        if (distance < minDistance) {
          minDistance = distance;
          closestNode = node;
        }
      }
    });

    return closestNode;
  }

  dispose() {
    this.handlers = {};
    this.currentInteraction = null;
  }
}
