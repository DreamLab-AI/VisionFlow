import React from 'react';
import { Canvas } from '@react-three/fiber';

const SimpleThreeTest: React.FC = () => {
  return (
    <div style={{ 
      position: 'fixed', 
      top: 0, 
      left: 0, 
      width: '100vw', 
      height: '100vh',
      backgroundColor: 'purple'
    }}>
      <div style={{
        position: 'absolute',
        top: '10px',
        left: '10px',
        color: 'white',
        backgroundColor: 'blue',
        padding: '10px',
        zIndex: 1000
      }}>
        Simple Three Test - Purple = container, Green = WebGL working
      </div>
      
      <Canvas
        camera={{ position: [0, 0, 5] }}
        onCreated={({ gl }) => {
          console.log('[SimpleThreeTest] Canvas created!', gl);
          // Set background to green if WebGL works
          gl.setClearColor(0x00ff00, 1);
        }}
        onError={(error) => {
          console.error('[SimpleThreeTest] WebGL Error:', error);
        }}
      >
        <mesh>
          <boxGeometry args={[1, 1, 1]} />
          <meshBasicMaterial color="red" />
        </mesh>
      </Canvas>
    </div>
  );
};

export default SimpleThreeTest;