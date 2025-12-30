import { useEffect, useRef } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { SpaceDriver } from '../../../services/SpaceDriverService';

interface SpacePilotSimpleIntegrationProps {
  orbitControlsRef?: React.RefObject<any>;
}


export const SpacePilotSimpleIntegration: React.FC<SpacePilotSimpleIntegrationProps> = ({ orbitControlsRef }) => {
  const { camera } = useThree();

  

  
  const translation = useRef({ x: 0, y: 0, z: 0 });
  const rotation = useRef({ rx: 0, ry: 0, rz: 0 });

  
  const smoothedTranslation = useRef({ x: 0, y: 0, z: 0 });
  const smoothedRotation = useRef({ rx: 0, ry: 0, rz: 0 });

  
  const spherical = useRef(new THREE.Spherical());
  const target = useRef(new THREE.Vector3(0, 0, 0));

  
  const config = {
    translationSpeed: 5.0,   
    translationSpeedY: 2.5,  
    rotationSpeed: 0.02,     
    rotationSpeedRY: 0.02,   
    rotationSpeedRZ: 0.02,   
    deadzone: 0.02,          
    smoothing: 0.85,
    invertRX: true           
  };

  
  useEffect(() => {
    if (camera) {
      
      spherical.current.setFromVector3(camera.position.clone().sub(target.current));
    }
  }, [camera]);

  
  const isConnected = useRef(false);
  const isActivelyMoving = useRef(false);
  const hasDivergedFromOrbit = useRef(false);
  const lastActivityTime = useRef(0);

  
  useEffect(() => {
    const handleMouseInteraction = () => {
      
      const orbitControls = orbitControlsRef?.current;
      if (orbitControls && hasDivergedFromOrbit.current) {
        
        orbitControls.target.set(0, 0, 0);

        
        orbitControls.object.position.copy(camera.position);

        
        orbitControls.update();

        hasDivergedFromOrbit.current = false;
        console.log('[SpacePilot] Mouse interaction - OrbitControls snapped back to origin');
      }
    };

    window.addEventListener('mousedown', handleMouseInteraction);
    window.addEventListener('wheel', handleMouseInteraction);

    return () => {
      window.removeEventListener('mousedown', handleMouseInteraction);
      window.removeEventListener('wheel', handleMouseInteraction);
    };
  }, [camera, orbitControlsRef]);

  
  useEffect(() => {
    const handleConnect = () => {
      isConnected.current = true;
      
      console.log('[SpacePilot] Connected - Hybrid control mode active');
    };

    const handleDisconnect = () => {
      isConnected.current = false;
      isActivelyMoving.current = false;
      
      const orbitControls = orbitControlsRef?.current;
      if (orbitControls) {
        orbitControls.enabled = true;
        console.log('[SpacePilot] Disconnected - OrbitControls re-enabled');
      }
    };

    const handleTranslate = (event: CustomEvent) => {
      const { x, y, z } = event.detail;
      
      
      
      const scale = 1 / 90;
      const nx = x * scale;
      const ny = y * scale;
      const nz = z * scale;
      translation.current = {
        x: Math.abs(nx) > config.deadzone ? nx : 0,
        y: Math.abs(ny) > config.deadzone ? ny : 0,
        z: Math.abs(nz) > config.deadzone ? nz : 0
      };
      
    };

    const handleRotate = (event: CustomEvent) => {
      const { rx, ry, rz } = event.detail;
      
      const scale = 1 / 90; 
      const nrx = rx * scale;
      const nry = ry * scale;
      const nrz = rz * scale;
      rotation.current = {
        rx: Math.abs(nrx) > config.deadzone ? nrx : 0,
        ry: Math.abs(nry) > config.deadzone ? nry : 0,
        rz: Math.abs(nrz) > config.deadzone ? nrz : 0
      };
      
    };

    const handleButtons = (event: CustomEvent) => {
      const { buttons } = event.detail;
      
      if (buttons.includes('[1]')) {
        spherical.current.set(50, Math.PI / 4, 0);
        target.current.set(0, 0, 0);
      }
    };

    SpaceDriver.addEventListener('connect', handleConnect as EventListener);
    SpaceDriver.addEventListener('disconnect', handleDisconnect as EventListener);
    SpaceDriver.addEventListener('translate', handleTranslate as EventListener);
    SpaceDriver.addEventListener('rotate', handleRotate as EventListener);
    SpaceDriver.addEventListener('buttons', handleButtons as EventListener);

    return () => {
      SpaceDriver.removeEventListener('connect', handleConnect as EventListener);
      SpaceDriver.removeEventListener('disconnect', handleDisconnect as EventListener);
      SpaceDriver.removeEventListener('translate', handleTranslate as EventListener);
      SpaceDriver.removeEventListener('rotate', handleRotate as EventListener);
      SpaceDriver.removeEventListener('buttons', handleButtons as EventListener);
    };
  }, [orbitControlsRef]);

  
  useFrame(() => {
    if (!camera || !isConnected.current) return;

    
    smoothedTranslation.current.x = smoothedTranslation.current.x * config.smoothing + translation.current.x * (1 - config.smoothing);
    smoothedTranslation.current.y = smoothedTranslation.current.y * config.smoothing + translation.current.y * (1 - config.smoothing);
    smoothedTranslation.current.z = smoothedTranslation.current.z * config.smoothing + translation.current.z * (1 - config.smoothing);

    smoothedRotation.current.rx = smoothedRotation.current.rx * config.smoothing + rotation.current.rx * (1 - config.smoothing);
    smoothedRotation.current.ry = smoothedRotation.current.ry * config.smoothing + rotation.current.ry * (1 - config.smoothing);
    smoothedRotation.current.rz = smoothedRotation.current.rz * config.smoothing + rotation.current.rz * (1 - config.smoothing);

    
    const isTranslating = Math.abs(smoothedTranslation.current.x) > 0.001 ||
                          Math.abs(smoothedTranslation.current.y) > 0.001 ||
                          Math.abs(smoothedTranslation.current.z) > 0.001;
    const isRotating = Math.abs(smoothedRotation.current.rx) > 0.001 ||
                       Math.abs(smoothedRotation.current.ry) > 0.001 ||
                       Math.abs(smoothedRotation.current.rz) > 0.001;

    const wasActive = isActivelyMoving.current;
    isActivelyMoving.current = isTranslating || isRotating;

    
    const orbitControls = orbitControlsRef?.current;
    if (orbitControls) {
      if (isActivelyMoving.current) {
        
        if (orbitControls.enabled) {
          orbitControls.enabled = false;
          lastActivityTime.current = Date.now();
        }
        hasDivergedFromOrbit.current = true;
      } else if (wasActive && !isActivelyMoving.current) {
        
        const timeSinceActive = Date.now() - lastActivityTime.current;
        if (timeSinceActive > 100) { 
          orbitControls.enabled = true;
          
          
        }
      }
    }

    
    if (isActivelyMoving.current) {
      
      
      const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion); 
      const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
      const up = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion); 

      
      camera.position.add(right.multiplyScalar(smoothedTranslation.current.x * config.translationSpeed)); 
      camera.position.add(forward.multiplyScalar(-smoothedTranslation.current.y * config.translationSpeedY)); 
      camera.position.add(up.multiplyScalar(-smoothedTranslation.current.z * config.translationSpeed)); 

      
      const pitchAmount = smoothedRotation.current.rx * config.rotationSpeed * (config.invertRX ? -1 : 1);
      const yawAmount = -smoothedRotation.current.rz * config.rotationSpeedRZ;  
      const rollAmount = -smoothedRotation.current.ry * config.rotationSpeedRY;  

      const euler = new THREE.Euler(
        pitchAmount, 
        yawAmount,   
        rollAmount,  
        'YXZ'
      );

      
      const quaternion = new THREE.Quaternion().setFromEuler(euler);
      camera.quaternion.multiply(quaternion);
    }
  });

  return null;
};

export default SpacePilotSimpleIntegration;