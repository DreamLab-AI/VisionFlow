import { useEffect } from 'react';
import { useThree } from '@react-three/fiber';
import * as THREE from 'three'; 

interface CameraControllerProps {
  center: [number, number, number];
  size: number;
}

const CameraController: React.FC<CameraControllerProps> = ({ center, size }) => {
  const { camera } = useThree();

  useEffect(() => {
    
    if (camera instanceof THREE.PerspectiveCamera) {
        
        camera.position.set(center[0], center[1] + 10, center[2] + size * 2);
        camera.lookAt(new THREE.Vector3(center[0], center[1], center[2])); 
        camera.updateProjectionMatrix();
    } else {
         console.warn("CameraController expects a PerspectiveCamera.");
         
         camera.position.set(center[0], center[1] + 10, center[2] + size * 2);
         camera.lookAt(new THREE.Vector3(center[0], center[1], center[2]));
    }
  }, [camera, center, size]);

  return null; 
};

export default CameraController;
