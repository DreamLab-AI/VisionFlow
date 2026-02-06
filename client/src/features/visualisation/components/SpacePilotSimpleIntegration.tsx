import { useEffect, useRef, useCallback } from 'react';
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

  // Pre-allocated reusable objects to avoid per-frame GC pressure
  const _forward = useRef(new THREE.Vector3());
  const _right = useRef(new THREE.Vector3());
  const _up = useRef(new THREE.Vector3());
  const _euler = useRef(new THREE.Euler());
  const _quat = useRef(new THREE.Quaternion());
  const _targetVec = useRef(new THREE.Vector3());

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

  const isConnected = useRef(false);
  const isActivelyMoving = useRef(false);
  const hasDivergedFromOrbit = useRef(false);
  const lastActivityTime = useRef(0);

  /**
   * Syncs OrbitControls internal state to match the camera's current
   * position and orientation. Places the orbit target along the camera's
   * forward vector so re-enabling OrbitControls doesn't snap the view.
   */
  const syncOrbitControlsToCamera = useCallback(() => {
    const orbitControls = orbitControlsRef?.current;
    if (!orbitControls || !camera) return;

    // Compute forward direction from current camera quaternion
    const fwd = _forward.current.set(0, 0, -1).applyQuaternion(camera.quaternion);

    // Place target at a reasonable distance along the forward vector
    const dist = Math.max(camera.position.length(), 20);
    _targetVec.current.copy(camera.position).add(fwd.multiplyScalar(dist));

    // Update OrbitControls to accept the new camera state
    orbitControls.target.copy(_targetVec.current);
    orbitControls.update();
  }, [camera, orbitControlsRef]);

  // On mouse/wheel interaction after SpacePilot diverged, sync orbit to
  // where the camera actually IS instead of snapping to origin.
  useEffect(() => {
    const handleMouseInteraction = () => {
      const orbitControls = orbitControlsRef?.current;
      if (orbitControls && hasDivergedFromOrbit.current) {
        syncOrbitControlsToCamera();
        hasDivergedFromOrbit.current = false;
      }
    };

    window.addEventListener('mousedown', handleMouseInteraction);
    window.addEventListener('wheel', handleMouseInteraction);

    return () => {
      window.removeEventListener('mousedown', handleMouseInteraction);
      window.removeEventListener('wheel', handleMouseInteraction);
    };
  }, [camera, orbitControlsRef, syncOrbitControlsToCamera]);

  useEffect(() => {
    const handleConnect = () => {
      isConnected.current = true;
    };

    const handleDisconnect = () => {
      isConnected.current = false;
      isActivelyMoving.current = false;

      // Sync OrbitControls to current view BEFORE re-enabling
      const orbitControls = orbitControlsRef?.current;
      if (orbitControls) {
        syncOrbitControlsToCamera();
        orbitControls.enabled = true;
        hasDivergedFromOrbit.current = false;
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
        // Reset: sync orbit target to origin and move camera to default
        const orbitControls = orbitControlsRef?.current;
        if (orbitControls) {
          orbitControls.target.set(0, 0, 0);
          orbitControls.update();
        }
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
  }, [orbitControlsRef, syncOrbitControlsToCamera]);

  useFrame(() => {
    if (!camera || !isConnected.current) return;

    // Exponential smoothing
    const s = config.smoothing;
    const inv = 1 - s;
    smoothedTranslation.current.x = smoothedTranslation.current.x * s + translation.current.x * inv;
    smoothedTranslation.current.y = smoothedTranslation.current.y * s + translation.current.y * inv;
    smoothedTranslation.current.z = smoothedTranslation.current.z * s + translation.current.z * inv;

    smoothedRotation.current.rx = smoothedRotation.current.rx * s + rotation.current.rx * inv;
    smoothedRotation.current.ry = smoothedRotation.current.ry * s + rotation.current.ry * inv;
    smoothedRotation.current.rz = smoothedRotation.current.rz * s + rotation.current.rz * inv;

    // Detect active movement
    const isTranslating = Math.abs(smoothedTranslation.current.x) > 0.001 ||
                          Math.abs(smoothedTranslation.current.y) > 0.001 ||
                          Math.abs(smoothedTranslation.current.z) > 0.001;
    const isRotating = Math.abs(smoothedRotation.current.rx) > 0.001 ||
                       Math.abs(smoothedRotation.current.ry) > 0.001 ||
                       Math.abs(smoothedRotation.current.rz) > 0.001;

    const wasActive = isActivelyMoving.current;
    isActivelyMoving.current = isTranslating || isRotating;

    // OrbitControls handoff
    const orbitControls = orbitControlsRef?.current;
    if (orbitControls) {
      if (isActivelyMoving.current) {
        if (orbitControls.enabled) {
          orbitControls.enabled = false;
          lastActivityTime.current = Date.now();
        }
        hasDivergedFromOrbit.current = true;
      } else if (wasActive && !isActivelyMoving.current) {
        // SpacePilot stopped — sync OrbitControls to current camera state
        // BEFORE re-enabling so it doesn't snap back to stale position
        syncOrbitControlsToCamera();
        orbitControls.enabled = true;
        hasDivergedFromOrbit.current = false;
      }
    }

    // Apply SpacePilot input to camera
    if (isActivelyMoving.current) {
      // Reuse pre-allocated vectors
      const forward = _forward.current.set(0, 0, -1).applyQuaternion(camera.quaternion);
      const right = _right.current.set(1, 0, 0).applyQuaternion(camera.quaternion);
      const up = _up.current.set(0, 1, 0).applyQuaternion(camera.quaternion);

      camera.position.add(right.multiplyScalar(smoothedTranslation.current.x * config.translationSpeed));
      camera.position.add(forward.multiplyScalar(-smoothedTranslation.current.y * config.translationSpeedY));
      camera.position.add(up.multiplyScalar(-smoothedTranslation.current.z * config.translationSpeed));

      const pitchAmount = smoothedRotation.current.rx * config.rotationSpeed * (config.invertRX ? -1 : 1);
      const yawAmount = -smoothedRotation.current.rz * config.rotationSpeedRZ;
      const rollAmount = -smoothedRotation.current.ry * config.rotationSpeedRY;

      _euler.current.set(pitchAmount, yawAmount, rollAmount, 'YXZ');
      camera.quaternion.multiply(_quat.current.setFromEuler(_euler.current));
    }
  });

  return null;
};

export default SpacePilotSimpleIntegration;