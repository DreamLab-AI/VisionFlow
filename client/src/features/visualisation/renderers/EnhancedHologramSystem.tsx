import React, { useRef, useEffect, useMemo } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { useSettingsStore } from '@/store/settingsStore';

// Plasma field effect shader
const plasmaShader = {
  vertexShader: `
    varying vec2 vUv;
    varying vec3 vPosition;
    uniform float time;
    
    void main() {
      vUv = uv;
      vPosition = position;
      vec3 pos = position;
      
      // Add wave distortion
      float wave = sin(position.x * 10.0 + time) * 0.05;
      pos.y += wave;
      
      gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
    }
  `,
  fragmentShader: `
    uniform float time;
    uniform vec3 color;
    uniform float opacity;
    varying vec2 vUv;
    varying vec3 vPosition;
    
    void main() {
      // Create animated plasma effect
      float plasma = sin(vUv.x * 10.0 + time) * 0.5 + 0.5;
      plasma += sin(vUv.y * 8.0 - time * 0.5) * 0.5 + 0.5;
      plasma += sin((vUv.x + vUv.y) * 12.0 + time * 1.5) * 0.5 + 0.5;
      plasma /= 3.0;
      
      // Energy pulse
      float pulse = sin(time * 2.0) * 0.2 + 0.8;
      
      vec3 finalColor = mix(color, vec3(1.0), plasma * 0.3) * pulse;
      float finalOpacity = opacity * (0.5 + plasma * 0.5);
      
      gl_FragColor = vec4(finalColor, finalOpacity);
    }
  `
};

// Quantum field effect
export const QuantumField: React.FC<{
  size?: number;
  color?: THREE.Color | string;
  opacity?: number;
}> = ({ size = 100, color = '#00ffff', opacity = 0.3 }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  
  const uniforms = useMemo(() => ({
    time: { value: 0 },
    color: { value: new THREE.Color(color) },
    opacity: { value: opacity }
  }), [color, opacity]);
  
  useFrame((state) => {
    if (materialRef.current) {
      materialRef.current.uniforms.time.value = state.clock.elapsedTime;
    }
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.001;
    }
  });
  
  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[size, 64, 64]} />
      <shaderMaterial
        ref={materialRef}
        uniforms={uniforms}
        vertexShader={plasmaShader.vertexShader}
        fragmentShader={plasmaShader.fragmentShader}
        transparent
        blending={THREE.AdditiveBlending}
        depthWrite={false}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
};

// Energy rings with particle trails
export const EnergyRing: React.FC<{
  innerRadius: number;
  outerRadius: number;
  color?: string;
  opacity?: number;
  rotationSpeed?: number;
  particleCount?: number;
}> = ({
  innerRadius,
  outerRadius,
  color = '#00ffff',
  opacity = 0.8,
  rotationSpeed = 1,
  particleCount = 100
}) => {
  const groupRef = useRef<THREE.Group>(null);
  const particlesRef = useRef<THREE.Points>(null);
  
  const particles = useMemo(() => {
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const colorObj = new THREE.Color(color);
    
    for (let i = 0; i < particleCount; i++) {
      const angle = (i / particleCount) * Math.PI * 2;
      const radius = innerRadius + (outerRadius - innerRadius) * Math.random();
      
      positions[i * 3] = Math.cos(angle) * radius;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 2;
      positions[i * 3 + 2] = Math.sin(angle) * radius;
      
      colors[i * 3] = colorObj.r;
      colors[i * 3 + 1] = colorObj.g;
      colors[i * 3 + 2] = colorObj.b;
    }
    
    return { positions, colors };
  }, [particleCount, innerRadius, outerRadius, color]);
  
  useFrame((state, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * rotationSpeed;
    }
    if (particlesRef.current) {
      const positions = particlesRef.current.geometry.attributes.position.array as Float32Array;
      for (let i = 0; i < particleCount; i++) {
        positions[i * 3 + 1] = Math.sin(state.clock.elapsedTime * 2 + i * 0.1) * 2;
      }
      particlesRef.current.geometry.attributes.position.needsUpdate = true;
    }
  });
  
  return (
    <group ref={groupRef}>
      <mesh>
        <ringGeometry args={[innerRadius, outerRadius, 64]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={opacity}
          side={THREE.DoubleSide}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
      <points ref={particlesRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={particleCount}
            array={particles.positions}
            itemSize={3}
          />
          <bufferAttribute
            attach="attributes-color"
            count={particleCount}
            array={particles.colors}
            itemSize={3}
          />
        </bufferGeometry>
        <pointsMaterial
          size={0.5}
          transparent
          opacity={opacity * 0.6}
          blending={THREE.AdditiveBlending}
          vertexColors
        />
      </points>
    </group>
  );
};

// Geodesic dome with energy flow
export const GeodesicDome: React.FC<{
  size: number;
  color?: string;
  opacity?: number;
  detail?: number;
}> = ({ size, color = '#00ffff', opacity = 0.3, detail = 2 }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [pulsePhase, setPulsePhase] = React.useState(0);
  
  useFrame((state, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.x += delta * 0.1;
      meshRef.current.rotation.y += delta * 0.15;
    }
    setPulsePhase(prev => prev + delta);
  });
  
  const emissiveIntensity = Math.sin(pulsePhase * 2) * 0.5 + 0.5;
  
  return (
    <mesh ref={meshRef}>
      <geodesicPolyhedronGeometry args={[size, detail]} />
      <meshPhongMaterial
        color={color}
        emissive={color}
        emissiveIntensity={emissiveIntensity}
        transparent
        opacity={opacity}
        wireframe
        side={THREE.DoubleSide}
      />
    </mesh>
  );
};

// Buckminster Fuller sphere
export const BuckminsterSphere: React.FC<{
  size: number;
  color?: string;
  opacity?: number;
}> = ({ size, color = '#00ffff', opacity = 0.3 }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  
  useFrame((state, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.x += delta * 0.05;
      meshRef.current.rotation.z += delta * 0.08;
    }
  });
  
  return (
    <mesh ref={meshRef}>
      <dodecahedronGeometry args={[size, 0]} />
      <meshBasicMaterial
        color={color}
        transparent
        opacity={opacity}
        wireframe
        blending={THREE.AdditiveBlending}
      />
    </mesh>
  );
};

// Data flow visualization
export const DataFlowVisualization: React.FC<{
  radius: number;
  color?: string;
  flowSpeed?: number;
}> = ({ radius, color = '#00ffff', flowSpeed = 1 }) => {
  const linesRef = useRef<THREE.Group>(null);
  const lineCount = 12;
  
  useFrame((state) => {
    if (linesRef.current) {
      linesRef.current.children.forEach((line, i) => {
        const offset = (state.clock.elapsedTime * flowSpeed + i * 0.5) % 1;
        line.material.opacity = Math.sin(offset * Math.PI) * 0.8;
        line.material.needsUpdate = true;
      });
    }
  });
  
  return (
    <group ref={linesRef}>
      {Array.from({ length: lineCount }).map((_, i) => {
        const angle = (i / lineCount) * Math.PI * 2;
        const points = [];
        
        for (let j = 0; j <= 20; j++) {
          const t = j / 20;
          const spiralRadius = radius * (1 - t * 0.3);
          const height = t * radius * 2 - radius;
          const spiralAngle = angle + t * Math.PI * 2;
          
          points.push(new THREE.Vector3(
            Math.cos(spiralAngle) * spiralRadius,
            height,
            Math.sin(spiralAngle) * spiralRadius
          ));
        }
        
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        
        return (
          <line key={i} geometry={geometry}>
            <lineBasicMaterial
              color={color}
              transparent
              opacity={0}
              blending={THREE.AdditiveBlending}
            />
          </line>
        );
      })}
    </group>
  );
};

// Main enhanced hologram system
export const EnhancedHologramSystem: React.FC<{
  position?: [number, number, number];
  scale?: number;
}> = ({ position = [0, 0, 0], scale = 1 }) => {
  const settings = useSettingsStore(state => state.settings?.visualisation?.hologram);
  const groupRef = useRef<THREE.Group>(null);
  
  useEffect(() => {
    if (groupRef.current) {
      groupRef.current.layers.set(0);
      groupRef.current.layers.enable(1); // Enable bloom layer
    }
  }, []);
  
  const {
    ringCount = 5,
    ringColor = '#00ffff',
    ringOpacity = 0.8,
    sphereSizes = [40, 80],
    enableBuckminster = true,
    buckminsterSize = 50,
    buckminsterOpacity = 0.3,
    enableGeodesic = true,
    geodesicSize = 60,
    geodesicOpacity = 0.25,
    enableTriangleSphere = true,
    triangleSphereSize = 70,
    triangleSphereOpacity = 0.4,
    globalRotationSpeed = 0.5
  } = settings || {};
  
  useFrame((state, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * globalRotationSpeed * 0.1;
    }
  });
  
  return (
    <group ref={groupRef} position={position} scale={scale}>
      {/* Quantum field background */}
      <QuantumField size={100} color={ringColor} opacity={0.1} />
      
      {/* Energy rings */}
      {Array.from({ length: ringCount }).map((_, i) => {
        const baseSize = sphereSizes[i % sphereSizes.length];
        return (
          <EnergyRing
            key={`ring-${i}`}
            innerRadius={baseSize * 0.8}
            outerRadius={baseSize}
            color={ringColor}
            opacity={ringOpacity * (1 - i * 0.1)}
            rotationSpeed={1 + i * 0.3}
            particleCount={50 + i * 20}
          />
        );
      })}
      
      {/* Geodesic dome */}
      {enableGeodesic && (
        <GeodesicDome
          size={geodesicSize}
          color={ringColor}
          opacity={geodesicOpacity}
          detail={3}
        />
      )}
      
      {/* Buckminster sphere */}
      {enableBuckminster && (
        <BuckminsterSphere
          size={buckminsterSize}
          color={ringColor}
          opacity={buckminsterOpacity}
        />
      )}
      
      {/* Triangle sphere */}
      {enableTriangleSphere && (
        <mesh>
          <icosahedronGeometry args={[triangleSphereSize, 2]} />
          <meshPhongMaterial
            color={ringColor}
            emissive={ringColor}
            emissiveIntensity={0.5}
            transparent
            opacity={triangleSphereOpacity}
            wireframe
            side={THREE.DoubleSide}
          />
        </mesh>
      )}
      
      {/* Data flow visualization */}
      <DataFlowVisualization
        radius={Math.max(...sphereSizes)}
        color={ringColor}
        flowSpeed={2}
      />
    </group>
  );
};

// Export for use in other components
export default EnhancedHologramSystem;