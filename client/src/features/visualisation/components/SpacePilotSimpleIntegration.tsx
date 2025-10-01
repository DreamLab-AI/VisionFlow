import { useEffect, useRef } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { SpaceDriver } from '../../../services/SpaceDriverService';

interface SpacePilotSimpleIntegrationProps {
  orbitControlsRef?: React.RefObject<any>;
}

/**
 * Simple SpacePilot integration for orbit-style camera control
 * Works with or without OrbitControls
 */
export const SpacePilotSimpleIntegration: React.FC<SpacePilotSimpleIntegrationProps> = ({ orbitControlsRef }) => {
  const { camera } = useThree();

  // console.log('[SpacePilotSimpleIntegration] Component mounted', { hasControls: !!controls });

  // Current input values
  const translation = useRef({ x: 0, y: 0, z: 0 });
  const rotation = useRef({ rx: 0, ry: 0, rz: 0 });

  // Smoothing
  const smoothedTranslation = useRef({ x: 0, y: 0, z: 0 });
  const smoothedRotation = useRef({ rx: 0, ry: 0, rz: 0 });

  // Camera orbit parameters
  const spherical = useRef(new THREE.Spherical());
  const target = useRef(new THREE.Vector3(0, 0, 0));

  // Configuration
  const config = {
    translationSpeed: 5.0,   // 5x increase for proper sensitivity
    translationSpeedY: 2.5,  // Y axis at 2.5x (reduced by 2x from other axes)
    rotationSpeed: 0.02,     // Reduced 5x for RX (pitch) sensitivity
    rotationSpeedRY: 0.02,   // Reduced sensitivity for RY (roll)
    rotationSpeedRZ: 0.02,   // Original reduced sensitivity for RZ (yaw)
    deadzone: 0.02,          // Very low deadzone for sensitivity
    smoothing: 0.85,
    invertRX: true           // Invert pitch
  };

  // Initialize camera position
  useEffect(() => {
    if (camera) {
      // Get current camera position in spherical coordinates
      spherical.current.setFromVector3(camera.position.clone().sub(target.current));
    }
  }, [camera]);

  // Track connection state
  const isConnected = useRef(false);

  // Set up SpacePilot event listeners
  useEffect(() => {
    const handleConnect = () => {
      isConnected.current = true;
      // Completely disable OrbitControls when SpacePilot connects
      const orbitControls = orbitControlsRef?.current;
      if (orbitControls) {
        orbitControls.enabled = false;
        orbitControls.enableRotate = false;
        orbitControls.enablePan = false;
        orbitControls.enableZoom = false;
        console.log('[SpacePilot] Connected - OrbitControls completely disabled for free-flying');
      } else {
        console.log('[SpacePilot] Connected - No OrbitControls ref provided');
      }
    };

    const handleDisconnect = () => {
      isConnected.current = false;
      // Re-enable OrbitControls when SpacePilot disconnects
      const orbitControls = orbitControlsRef?.current;
      if (orbitControls) {
        orbitControls.enabled = true;
        orbitControls.enableRotate = true;
        orbitControls.enablePan = true;
        orbitControls.enableZoom = true;
        console.log('[SpacePilot] Disconnected - OrbitControls re-enabled');
      }
    };

    const handleTranslate = (event: CustomEvent) => {
      const { x, y, z } = event.detail;
      // console.log('[SpacePilot] Raw translation:', { x, y, z });
      // Normalize from Int16 range (-32768 to 32767) to -1 to 1
      // Increased scale by 5x for proper sensitivity (was 1/450, now 1/90)
      const scale = 1 / 90;
      const nx = x * scale;
      const ny = y * scale;
      const nz = z * scale;
      translation.current = {
        x: Math.abs(nx) > config.deadzone ? nx : 0,
        y: Math.abs(ny) > config.deadzone ? ny : 0,
        z: Math.abs(nz) > config.deadzone ? nz : 0
      };
      // console.log('[SpacePilot] Normalized translation:', translation.current);
    };

    const handleRotate = (event: CustomEvent) => {
      const { rx, ry, rz } = event.detail;
      // console.log('[SpacePilot] Raw rotation:', { rx, ry, rz });
      const scale = 1 / 90; // 5x increase for proper sensitivity
      const nrx = rx * scale;
      const nry = ry * scale;
      const nrz = rz * scale;
      rotation.current = {
        rx: Math.abs(nrx) > config.deadzone ? nrx : 0,
        ry: Math.abs(nry) > config.deadzone ? nry : 0,
        rz: Math.abs(nrz) > config.deadzone ? nrz : 0
      };
      // console.log('[SpacePilot] Normalized rotation:', rotation.current);
    };

    const handleButtons = (event: CustomEvent) => {
      const { buttons } = event.detail;
      // Button 1 = Reset view
      if (buttons.includes('[1]')) {
        spherical.current.set(50, Math.PI / 4, 0);
        target.current.set(0, 0, 0);
      }
    };

    SpaceDriver.addEventListener('connect', handleConnect);
    SpaceDriver.addEventListener('disconnect', handleDisconnect);
    SpaceDriver.addEventListener('translate', handleTranslate);
    SpaceDriver.addEventListener('rotate', handleRotate);
    SpaceDriver.addEventListener('buttons', handleButtons);

    return () => {
      SpaceDriver.removeEventListener('connect', handleConnect);
      SpaceDriver.removeEventListener('disconnect', handleDisconnect);
      SpaceDriver.removeEventListener('translate', handleTranslate);
      SpaceDriver.removeEventListener('rotate', handleRotate);
      SpaceDriver.removeEventListener('buttons', handleButtons);
    };
  }, [orbitControlsRef]);

  // Update camera position every frame
  useFrame(() => {
    if (!camera || !isConnected.current) return;

    // Apply smoothing
    smoothedTranslation.current.x = smoothedTranslation.current.x * config.smoothing + translation.current.x * (1 - config.smoothing);
    smoothedTranslation.current.y = smoothedTranslation.current.y * config.smoothing + translation.current.y * (1 - config.smoothing);
    smoothedTranslation.current.z = smoothedTranslation.current.z * config.smoothing + translation.current.z * (1 - config.smoothing);

    smoothedRotation.current.rx = smoothedRotation.current.rx * config.smoothing + rotation.current.rx * (1 - config.smoothing);
    smoothedRotation.current.ry = smoothedRotation.current.ry * config.smoothing + rotation.current.ry * (1 - config.smoothing);
    smoothedRotation.current.rz = smoothedRotation.current.rz * config.smoothing + rotation.current.rz * (1 - config.smoothing);

    // Free-fly camera control (completely independent from OrbitControls)
    // Get camera-relative directions
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion); // Forward is -Z in camera space
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion); // Up is Y in camera space

    // Translation: X=strafe left/right (positive X = right), Y=forward/back (negative Y = forward), Z=up/down (positive Z = down)
    camera.position.add(right.multiplyScalar(smoothedTranslation.current.x * config.translationSpeed)); // X controls strafe (positive = right)
    camera.position.add(forward.multiplyScalar(-smoothedTranslation.current.y * config.translationSpeedY)); // Y controls forward/back (reduced sensitivity)
    camera.position.add(up.multiplyScalar(-smoothedTranslation.current.z * config.translationSpeed)); // Z controls up/down (positive = down)

    // Rotation: Apply to camera directly for free-flying
    const pitchAmount = smoothedRotation.current.rx * config.rotationSpeed * (config.invertRX ? -1 : 1);
    const yawAmount = -smoothedRotation.current.rz * config.rotationSpeedRZ;  // RZ controls yaw (inverted)
    const rollAmount = -smoothedRotation.current.ry * config.rotationSpeedRY;  // RY controls roll (inverted)

    const euler = new THREE.Euler(
      pitchAmount, // Pitch (tilt up/down)
      yawAmount,   // Yaw (turn left/right) - RZ controls yaw (inverted)
      rollAmount,  // Roll (tilt sideways) - RY controls roll
      'YXZ'
    );

    // Apply rotation
    const quaternion = new THREE.Quaternion().setFromEuler(euler);
    camera.quaternion.multiply(quaternion);
  });

  return null;
};

export default SpacePilotSimpleIntegration;