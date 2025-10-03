import { useEffect } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useSettingsStore } from '@/store/settingsStore';
import { useHeadTracking } from '@/hooks/useHeadTracking';
import { toast } from '@/features/design-system/components/Toast';

export function HeadTrackedParallaxController() {
  const { camera, size } = useThree();
  const { isEnabled, setIsEnabled, isTracking, headPosition, error } = useHeadTracking();

  const trackingEnabled = useSettingsStore(state => state.settings?.visualisation?.interaction?.headTrackedParallax?.enabled);
  const sensitivity = useSettingsStore(state => state.settings?.visualisation?.interaction?.headTrackedParallax?.sensitivity ?? 1.0);
  const cameraMode = useSettingsStore(state => state.settings?.visualisation?.interaction?.headTrackedParallax?.cameraMode ?? 'asymmetricFrustum');

  useEffect(() => {
    setIsEnabled(!!trackingEnabled);
  }, [trackingEnabled, setIsEnabled]);

  useEffect(() => {
    if (error) {
      toast({
        title: 'Head Tracking Error',
        description: error,
        variant: 'destructive',
      });
    }
  }, [error]);

  useFrame(() => {
    if (isTracking && headPosition && camera instanceof THREE.PerspectiveCamera) {
      if (cameraMode === 'asymmetricFrustum') {
        // Asymmetric frustum approach - more realistic parallax
        const virtualScreenScale = 1.0 + sensitivity * 0.5;
        const fullWidth = size.width * virtualScreenScale;
        const fullHeight = size.height * virtualScreenScale;

        const x_offset = -headPosition.x * (fullWidth - size.width) / 2;
        const y_offset = headPosition.y * (fullHeight - size.height) / 2;

        camera.setViewOffset(
          fullWidth,
          fullHeight,
          x_offset,
          y_offset,
          size.width,
          size.height
        );
        camera.updateProjectionMatrix();
      } else {
        // Offset mode - simpler parallax (may conflict with OrbitControls)
        const offsetX = headPosition.x * sensitivity * -0.5;
        const offsetY = headPosition.y * sensitivity * 0.5;

        const offsetVector = new THREE.Vector3(offsetX, offsetY, 0);
        const nudgeMatrix = new THREE.Matrix4().makeTranslation(offsetVector.x, offsetVector.y, 0);
        camera.projectionMatrix.multiply(nudgeMatrix);
      }
    } else {
      // Clear view offset when not tracking
      if (camera instanceof THREE.PerspectiveCamera && camera.view) {
        camera.clearViewOffset();
        camera.updateProjectionMatrix();
      }
    }
  });

  return null;
}
