

import { Vector3, Color, Camera, Quaternion, AnimationMixer, Clock } from 'three';
import { createLogger } from '../../../utils/loggerConfig';
import type { GraphData } from '../managers/graphDataManager';

const logger = createLogger('GraphAnimations');

export interface AnimationOptions {
  duration: number;
  easing: 'linear' | 'easeIn' | 'easeOut' | 'easeInOut' | 'bounce' | 'elastic';
  delay: number;
  repeat: number;
  autoReverse: boolean;
}

export interface TransitionAnimation {
  id: string;
  type: 'fade' | 'slide' | 'scale' | 'rotate' | 'morph';
  startState: any;
  endState: any;
  progress: number;
  options: AnimationOptions;
  onComplete?: () => void;
  onProgress?: (progress: number) => void;
}

export interface CameraFlightPath {
  id: string;
  waypoints: Array<{
    position: Vector3;
    target: Vector3;
    zoom: number;
    duration: number;
  }>;
  currentWaypoint: number;
  totalDuration: number;
  onComplete?: () => void;
}

export interface NodeAnimationState {
  nodeId: string;
  animationType: 'pulse' | 'spin' | 'bounce' | 'glow' | 'scale' | 'float';
  intensity: number;
  speed: number;
  phase: number;
}

export interface MorphingTransition {
  id: string;
  fromGraph: GraphData;
  toGraph: GraphData;
  progress: number;
  nodeMapping: Map<string, string>;
  interpolatedPositions: Map<string, Vector3>;
  interpolatedColors: Map<string, Color>;
  duration: number;
}

export class GraphAnimations {
  private static instance: GraphAnimations;
  private activeAnimations: Map<string, TransitionAnimation> = new Map();
  private cameraFlights: Map<string, CameraFlightPath> = new Map();
  private nodeAnimations: Map<string, NodeAnimationState> = new Map();
  private morphingTransitions: Map<string, MorphingTransition> = new Map();
  private animationMixer: AnimationMixer | null = null;
  private clock = new Clock();
  private isRunning = false;

  private constructor() {}

  public static getInstance(): GraphAnimations {
    if (!GraphAnimations.instance) {
      GraphAnimations.instance = new GraphAnimations();
    }
    return GraphAnimations.instance;
  }

  
  public start(): void {
    if (this.isRunning) return;
    
    this.isRunning = true;
    this.clock.start();
    this.animate();
    logger.info('Animation system started');
  }

  
  public stop(): void {
    this.isRunning = false;
    logger.info('Animation system stopped');
  }

  
  public animateGraphTransition(
    graphId: string,
    show: boolean,
    options: Partial<AnimationOptions> = {}
  ): Promise<void> {
    return new Promise((resolve) => {
      const animationId = `graph-transition-${graphId}`;
      
      const fullOptions: AnimationOptions = {
        duration: 1000,
        easing: 'easeInOut',
        delay: 0,
        repeat: 1,
        autoReverse: false,
        ...options
      };

      const animation: TransitionAnimation = {
        id: animationId,
        type: show ? 'fade' : 'fade',
        startState: { opacity: show ? 0 : 1, scale: show ? 0.5 : 1 },
        endState: { opacity: show ? 1 : 0, scale: show ? 1 : 0.5 },
        progress: 0,
        options: fullOptions,
        onComplete: () => {
          this.activeAnimations.delete(animationId);
          resolve();
        }
      };

      this.activeAnimations.set(animationId, animation);
      logger.info(`Started graph transition animation for ${graphId}`);
    });
  }

  
  public animateGraphMorph(
    morphId: string,
    fromGraph: GraphData,
    toGraph: GraphData,
    duration: number = 2000,
    nodeMapping?: Map<string, string>
  ): Promise<void> {
    return new Promise((resolve) => {
      
      const mapping = nodeMapping || this.generateNodeMapping(fromGraph, toGraph);
      
      const morph: MorphingTransition = {
        id: morphId,
        fromGraph,
        toGraph,
        progress: 0,
        nodeMapping: mapping,
        interpolatedPositions: new Map(),
        interpolatedColors: new Map(),
        duration
      };

      this.morphingTransitions.set(morphId, morph);
      
      
      setTimeout(() => {
        this.morphingTransitions.delete(morphId);
        resolve();
      }, duration);

      logger.info(`Started graph morphing animation: ${morphId}`);
    });
  }

  
  public animateCameraFlight(
    flightId: string,
    camera: Camera,
    waypoints: CameraFlightPath['waypoints'],
    onComplete?: () => void
  ): Promise<void> {
    return new Promise((resolve) => {
      const totalDuration = waypoints.reduce((sum, wp) => sum + wp.duration, 0);
      
      const flight: CameraFlightPath = {
        id: flightId,
        waypoints,
        currentWaypoint: 0,
        totalDuration,
        onComplete: () => {
          this.cameraFlights.delete(flightId);
          onComplete?.();
          resolve();
        }
      };

      this.cameraFlights.set(flightId, flight);
      logger.info(`Started camera flight: ${flightId} with ${waypoints.length} waypoints`);
    });
  }

  
  public animateNodeAppearance(
    nodeId: string,
    options: Partial<AnimationOptions> = {}
  ): Promise<void> {
    return new Promise((resolve) => {
      const animationId = `node-appear-${nodeId}`;
      
      const fullOptions: AnimationOptions = {
        duration: 800,
        easing: 'bounce',
        delay: 0,
        repeat: 1,
        autoReverse: false,
        ...options
      };

      const animation: TransitionAnimation = {
        id: animationId,
        type: 'scale',
        startState: { scale: 0, opacity: 0 },
        endState: { scale: 1, opacity: 1 },
        progress: 0,
        options: fullOptions,
        onComplete: () => {
          this.activeAnimations.delete(animationId);
          resolve();
        }
      };

      this.activeAnimations.set(animationId, animation);
    });
  }

  
  public animateNodeDisappearance(
    nodeId: string,
    options: Partial<AnimationOptions> = {}
  ): Promise<void> {
    return new Promise((resolve) => {
      const animationId = `node-disappear-${nodeId}`;
      
      const fullOptions: AnimationOptions = {
        duration: 600,
        easing: 'easeIn',
        delay: 0,
        repeat: 1,
        autoReverse: false,
        ...options
      };

      const animation: TransitionAnimation = {
        id: animationId,
        type: 'scale',
        startState: { scale: 1, opacity: 1 },
        endState: { scale: 0, opacity: 0 },
        progress: 0,
        options: fullOptions,
        onComplete: () => {
          this.activeAnimations.delete(animationId);
          resolve();
        }
      };

      this.activeAnimations.set(animationId, animation);
    });
  }

  
  public addNodeAnimation(
    nodeId: string,
    type: NodeAnimationState['animationType'],
    intensity: number = 1.0,
    speed: number = 1.0
  ): void {
    const animation: NodeAnimationState = {
      nodeId,
      animationType: type,
      intensity,
      speed,
      phase: 0
    };

    this.nodeAnimations.set(nodeId, animation);
    logger.info(`Added ${type} animation to node ${nodeId}`);
  }

  
  public removeNodeAnimation(nodeId: string): void {
    this.nodeAnimations.delete(nodeId);
    logger.info(`Removed animation from node ${nodeId}`);
  }

  
  public createGuidedTour(
    tourId: string,
    camera: Camera,
    interestingPoints: Array<{
      position: Vector3;
      target: Vector3;
      description: string;
      duration: number;
      highlightNodes?: string[];
    }>,
    onWaypointReached?: (index: number, point: any) => void
  ): Promise<void> {
    return new Promise((resolve) => {
      const waypoints = interestingPoints.map(point => ({
        position: point.position,
        target: point.target,
        zoom: 1,
        duration: point.duration
      }));

      this.animateCameraFlight(tourId, camera, waypoints, resolve);
      
      
      if (onWaypointReached) {
        interestingPoints.forEach((point, index) => {
          setTimeout(() => {
            onWaypointReached(index, point);
            
            
            if (point.highlightNodes) {
              point.highlightNodes.forEach(nodeId => {
                this.addNodeAnimation(nodeId, 'glow', 1.5, 2.0);
              });
            }
          }, waypoints.slice(0, index).reduce((sum, wp) => sum + wp.duration, 0));
        });
      }
    });
  }

  
  public animateTimeTravelMode(
    graphStates: GraphData[],
    stepDuration: number = 1000,
    onStateChange?: (stateIndex: number, graphData: GraphData) => void
  ): Promise<void> {
    return new Promise((resolve) => {
      let currentStateIndex = 0;
      
      const animateNextState = () => {
        if (currentStateIndex >= graphStates.length) {
          resolve();
          return;
        }

        const currentGraph = graphStates[currentStateIndex];
        onStateChange?.(currentStateIndex, currentGraph);

        if (currentStateIndex > 0) {
          const previousGraph = graphStates[currentStateIndex - 1];
          this.animateGraphMorph(
            `time-travel-${currentStateIndex}`,
            previousGraph,
            currentGraph,
            stepDuration
          );
        }

        currentStateIndex++;
        setTimeout(animateNextState, stepDuration);
      };

      animateNextState();
      logger.info(`Started time-travel animation through ${graphStates.length} states`);
    });
  }

  
  public getNodeAnimationValues(nodeId: string, time: number): {
    scale: number;
    rotation: number;
    position: Vector3;
    color: Color;
    opacity: number;
  } {
    const animation = this.nodeAnimations.get(nodeId);
    if (!animation) {
      return {
        scale: 1,
        rotation: 0,
        position: new Vector3(0, 0, 0),
        color: new Color(1, 1, 1),
        opacity: 1
      };
    }

    const phase = (time * animation.speed + animation.phase) % (Math.PI * 2);
    
    switch (animation.animationType) {
      case 'pulse':
        return {
          scale: 1 + Math.sin(phase) * animation.intensity * 0.3,
          rotation: 0,
          position: new Vector3(0, 0, 0),
          color: new Color(1, 1, 1),
          opacity: 1
        };

      case 'spin':
        return {
          scale: 1,
          rotation: phase,
          position: new Vector3(0, 0, 0),
          color: new Color(1, 1, 1),
          opacity: 1
        };

      case 'bounce':
        const bounceHeight = Math.abs(Math.sin(phase)) * animation.intensity * 2;
        return {
          scale: 1,
          rotation: 0,
          position: new Vector3(0, bounceHeight, 0),
          color: new Color(1, 1, 1),
          opacity: 1
        };

      case 'glow':
        const glowIntensity = (Math.sin(phase) + 1) * 0.5 * animation.intensity;
        return {
          scale: 1,
          rotation: 0,
          position: new Vector3(0, 0, 0),
          color: new Color(1 + glowIntensity, 1 + glowIntensity, 1 + glowIntensity),
          opacity: 1
        };

      case 'float':
        const floatY = Math.sin(phase) * animation.intensity * 0.5;
        const floatX = Math.cos(phase * 0.7) * animation.intensity * 0.3;
        return {
          scale: 1,
          rotation: 0,
          position: new Vector3(floatX, floatY, 0),
          color: new Color(1, 1, 1),
          opacity: 1
        };

      default:
        return {
          scale: 1,
          rotation: 0,
          position: new Vector3(0, 0, 0),
          color: new Color(1, 1, 1),
          opacity: 1
        };
    }
  }

  
  public getMorphingState(morphId: string): MorphingTransition | null {
    return this.morphingTransitions.get(morphId) || null;
  }

  
  private animate = (): void => {
    if (!this.isRunning) return;

    const deltaTime = this.clock.getDelta();
    const elapsedTime = this.clock.getElapsedTime();

    
    this.updateTransitionAnimations(deltaTime);

    
    this.updateCameraFlights(deltaTime);

    
    this.updateMorphingTransitions(deltaTime);

    
    this.updateNodeAnimations(elapsedTime);

    requestAnimationFrame(this.animate);
  };

  private updateTransitionAnimations(deltaTime: number): void {
    this.activeAnimations.forEach((animation, id) => {
      const progressDelta = deltaTime * 1000 / animation.options.duration;
      animation.progress = Math.min(animation.progress + progressDelta, 1);

      
      const easedProgress = this.applyEasing(animation.progress, animation.options.easing);
      
      
      animation.onProgress?.(easedProgress);

      
      if (animation.progress >= 1) {
        animation.onComplete?.();
      }
    });
  }

  private updateCameraFlights(deltaTime: number): void {
    this.cameraFlights.forEach((flight, id) => {
      
      
    });
  }

  private updateMorphingTransitions(deltaTime: number): void {
    this.morphingTransitions.forEach((morph, id) => {
      const progressDelta = deltaTime * 1000 / morph.duration;
      morph.progress = Math.min(morph.progress + progressDelta, 1);

      
      this.interpolateMorphingState(morph);
    });
  }

  private updateNodeAnimations(elapsedTime: number): void {
    this.nodeAnimations.forEach((animation, nodeId) => {
      animation.phase = elapsedTime * animation.speed;
    });
  }

  private interpolateMorphingState(morph: MorphingTransition): void {
    const progress = this.applyEasing(morph.progress, 'easeInOut');

    
    morph.nodeMapping.forEach((toNodeId, fromNodeId) => {
      const fromNode = morph.fromGraph.nodes.find(n => n.id === fromNodeId);
      const toNode = morph.toGraph.nodes.find(n => n.id === toNodeId);

      if (fromNode?.position && toNode?.position) {
        const fromPos = new Vector3(fromNode.position.x, fromNode.position.y, fromNode.position.z);
        const toPos = new Vector3(toNode.position.x, toNode.position.y, toNode.position.z);
        const interpolatedPos = fromPos.lerp(toPos, progress);
        
        morph.interpolatedPositions.set(fromNodeId, interpolatedPos);
      }
    });

    
    
  }

  private generateNodeMapping(fromGraph: GraphData, toGraph: GraphData): Map<string, string> {
    const mapping = new Map<string, string>();
    
    
    
    fromGraph.nodes.forEach(fromNode => {
      const matchingToNode = toGraph.nodes.find(toNode => 
        toNode.id === fromNode.id || toNode.label === fromNode.label
      );
      
      if (matchingToNode) {
        mapping.set(fromNode.id, matchingToNode.id);
      }
    });

    return mapping;
  }

  private applyEasing(progress: number, easing: AnimationOptions['easing']): number {
    switch (easing) {
      case 'linear':
        return progress;
      
      case 'easeIn':
        return progress * progress;
      
      case 'easeOut':
        return 1 - Math.pow(1 - progress, 2);
      
      case 'easeInOut':
        return progress < 0.5 
          ? 2 * progress * progress 
          : 1 - Math.pow(-2 * progress + 2, 2) / 2;
      
      case 'bounce':
        const n1 = 7.5625;
        const d1 = 2.75;
        
        if (progress < 1 / d1) {
          return n1 * progress * progress;
        } else if (progress < 2 / d1) {
          return n1 * (progress -= 1.5 / d1) * progress + 0.75;
        } else if (progress < 2.5 / d1) {
          return n1 * (progress -= 2.25 / d1) * progress + 0.9375;
        } else {
          return n1 * (progress -= 2.625 / d1) * progress + 0.984375;
        }
      
      case 'elastic':
        const c4 = (2 * Math.PI) / 3;
        return progress === 0 ? 0 : progress === 1 ? 1 : 
          -Math.pow(2, 10 * progress - 10) * Math.sin((progress * 10 - 10.75) * c4);
      
      default:
        return progress;
    }
  }

  
  public dispose(): void {
    this.stop();
    this.activeAnimations.clear();
    this.cameraFlights.clear();
    this.nodeAnimations.clear();
    this.morphingTransitions.clear();
    logger.info('Graph animations disposed');
  }
}

// Export singleton instance
export const graphAnimations = GraphAnimations.getInstance();