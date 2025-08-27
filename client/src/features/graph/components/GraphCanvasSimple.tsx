import React from 'react';
import { Canvas } from '@react-three/fiber';

const GraphCanvasSimple: React.FC = () => {
  console.log('[GraphCanvasSimple] Rendering');
  
  return (
    <div style={{ 
      position: 'fixed', 
      top: 0, 
      left: 0, 
      width: '100vw', 
      height: '100vh',
      backgroundColor: '#003366'
    }}>
      <div style={{
        position: 'absolute',
        top: '10px',
        left: '10px',
        color: 'white',
        backgroundColor: 'orange',
        padding: '10px',
        zIndex: 1000
      }}>
        GraphCanvasSimple - Blue = container, Green = WebGL
      </div>
      
      <Canvas
        camera={{ position: [0, 0, 5] }}
        onCreated={({ gl }) => {
          console.log('[GraphCanvasSimple] Canvas created!', gl);
          gl.setClearColor(0x00ff00, 1);
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

export default GraphCanvasSimple;