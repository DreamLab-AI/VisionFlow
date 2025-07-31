// Enhanced Agent Node Component
// Multi-layered 3D visualization with rich data integration

import React, { useRef, useMemo, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import { Html, Text, Billboard } from '@react-three/drei';
import * as THREE from 'three';
import { 
  EnhancedAgent, 
  AgentType, 
  AgentState, 
  EnhancedAgentNodeProps,
  AgentCapability 
} from './types';

// Color schemes for different agent types and states
const AGENT_TYPE_COLORS: Record<AgentType, string> = {
  COORDINATOR: '#FFD700', // Gold
  EXECUTOR: '#FF6B6B',    // Coral
  ANALYZER: '#4ECDC4',    // Turquoise
  MONITOR: '#45B7D1',     // Sky blue
  SPECIALIST: '#96CEB4',  // Mint
  RESEARCHER: '#FECA57',  // Yellow
  CODER: '#48CAE4',       // Light blue
  TESTER: '#F38BA8',      // Pink
  REVIEWER: '#A8DADC',    // Light cyan
  ARCHITECT: '#F1FAEE'    // Off white
};

const AGENT_STATE_COLORS: Record<AgentState, string> = {
  IDLE: '#6C757D',         // Gray
  THINKING: '#FFC107',     // Amber
  EXECUTING: '#28A745',    // Green
  COMMUNICATING: '#007BFF', // Blue
  COORDINATING: '#6F42C1', // Purple
  ERROR: '#DC3545',        // Red
  TERMINATED: '#343A40'    // Dark gray
};

// Agent type to geometry mapping
const getAgentGeometry = (agentType: AgentType, scale: number = 1) => {
  switch (agentType) {
    case 'COORDINATOR':
      return new THREE.OctahedronGeometry(0.8 * scale, 1);
    case 'EXECUTOR':
      return new THREE.BoxGeometry(1.2 * scale, 1.2 * scale, 1.2 * scale);
    case 'ANALYZER':
      return new THREE.IcosahedronGeometry(0.7 * scale, 1);
    case 'MONITOR':
      return new THREE.CylinderGeometry(0.6 * scale, 0.6 * scale, 1.4 * scale, 8);
    case 'SPECIALIST':
      return new THREE.TetrahedronGeometry(0.9 * scale);
    default:
      return new THREE.SphereGeometry(0.7 * scale, 16, 16);
  }
};

// Performance ring component
const PerformanceRing: React.FC<{
  agent: EnhancedAgent;
  position: THREE.Vector3;
  showRing: boolean;
}> = ({ agent, position, showRing }) => {
  const ringRef = useRef<THREE.Mesh>(null);
  const successRate = agent.performance.successRate;
  const resourceUtil = agent.performance.resourceUtilization;

  useFrame((state) => {
    if (!ringRef.current || !showRing) return;

    // Rotate based on resource utilization
    ringRef.current.rotation.z += resourceUtil * 0.02;
    
    // Pulse based on success rate
    const pulseScale = 1 + Math.sin(state.clock.elapsedTime * 2) * 0.1 * successRate;
    ringRef.current.scale.setScalar(pulseScale);
  });

  if (!showRing) return null;

  const ringColor = new THREE.Color().lerpColors(
    new THREE.Color('#FF4444'), // Red for low performance
    new THREE.Color('#44FF44'), // Green for high performance
    successRate
  );

  return (
    <mesh ref={ringRef} position={position}>
      <ringGeometry args={[1.5, 1.8, 16]} />
      <meshBasicMaterial 
        color={ringColor}
        transparent
        opacity={0.6}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
};

// Capability badges component
const CapabilityBadges: React.FC<{
  capabilities: AgentCapability[];
  position: THREE.Vector3;
  showBadges: boolean;
}> = ({ capabilities, position, showBadges }) => {
  if (!showBadges || capabilities.length === 0) return null;

  // Group capabilities by category and show top 4
  const topCapabilities = capabilities
    .sort((a, b) => b.level - a.level)
    .slice(0, 4);

  const categoryColors = {
    coordination: '#FFD700',
    execution: '#FF6B6B',
    analysis: '#4ECDC4',
    communication: '#45B7D1',
    specialized: '#96CEB4'
  };

  return (
    <group>
      {topCapabilities.map((capability, index) => {
        const angle = (index / topCapabilities.length) * Math.PI * 2;
        const radius = 2.5;
        const badgePos = new THREE.Vector3(
          position.x + Math.cos(angle) * radius,
          position.y + Math.sin(angle) * radius,
          position.z
        );

        return (
          <Billboard key={capability.name} position={badgePos}>
            <mesh>
              <circleGeometry args={[0.3, 8]} />
              <meshBasicMaterial 
                color={categoryColors[capability.category]}
                transparent
                opacity={0.8}
              />
            </mesh>
            <Text
              fontSize={0.2}
              color="white"
              anchorX="center"
              anchorY="middle"
              position={[0, 0, 0.01]}
            >
              {capability.name.charAt(0).toUpperCase()}
            </Text>
          </Billboard>
        );
      })}
    </group>
  );
};

// State indicator component
const StateIndicator: React.FC<{
  state: AgentState;
  position: THREE.Vector3;
}> = ({ state, position }) => {
  const indicatorRef = useRef<THREE.Mesh>(null);
  const stateColor = new THREE.Color(AGENT_STATE_COLORS[state]);

  useFrame((state_frame) => {
    if (!indicatorRef.current) return;

    switch (state) {
      case 'THINKING':
        // Gentle pulse for thinking
        const thinkPulse = 1 + Math.sin(state_frame.clock.elapsedTime * 3) * 0.2;
        indicatorRef.current.scale.setScalar(thinkPulse);
        break;
      case 'COMMUNICATING':
        // Flash for communication
        const flashIntensity = Math.sin(state_frame.clock.elapsedTime * 8);
        (indicatorRef.current.material as THREE.MeshBasicMaterial).opacity = 
          0.5 + Math.abs(flashIntensity) * 0.5;
        break;
      case 'EXECUTING':
        // Rotate for execution
        indicatorRef.current.rotation.y += 0.05;
        break;
      case 'ERROR':
        // Rapid pulse for error
        const errorPulse = 1 + Math.sin(state_frame.clock.elapsedTime * 10) * 0.3;
        indicatorRef.current.scale.setScalar(errorPulse);
        break;
    }
  });

  return (
    <mesh 
      ref={indicatorRef} 
      position={[position.x, position.y + 2.5, position.z]}
    >
      <sphereGeometry args={[0.3, 8, 8]} />
      <meshBasicMaterial 
        color={stateColor}
        transparent
        opacity={0.8}
      />
    </mesh>
  );
};

// Activity pulse component
const ActivityPulse: React.FC<{
  agent: EnhancedAgent;
  position: THREE.Vector3;
  qualityLevel: 'low' | 'medium' | 'high';
}> = ({ agent, position, qualityLevel }) => {
  const pulseRef = useRef<THREE.Mesh>(null);
  const resourceUtil = agent.performance.resourceUtilization;

  useFrame((state) => {
    if (!pulseRef.current) return;

    const pulseSpeed = 1 + resourceUtil * 2;
    const pulseScale = 2 + Math.sin(state.clock.elapsedTime * pulseSpeed) * resourceUtil;
    pulseRef.current.scale.setScalar(pulseScale);
    
    // Fade in/out based on activity
    (pulseRef.current.material as THREE.MeshBasicMaterial).opacity = 
      0.1 + resourceUtil * 0.3;
  });

  const segments = qualityLevel === 'high' ? 16 : qualityLevel === 'medium' ? 12 : 8;

  return (
    <mesh ref={pulseRef} position={position}>
      <sphereGeometry args={[0.5, segments, segments]} />
      <meshBasicMaterial 
        color={AGENT_TYPE_COLORS[agent.type]}
        transparent
        opacity={0.1}
        wireframe
      />
    </mesh>
  );
};

// Enhanced activity monitor with richer data
const EnhancedActivityMonitor: React.FC<{
  agent: EnhancedAgent;
  position: THREE.Vector3;
  isHovered: boolean;
}> = ({ agent, position, isHovered }) => {
  const [displayLogs, setDisplayLogs] = useState<string[]>([]);

  React.useEffect(() => {
    // Enhanced processing logs with performance data
    const enhancedLogs = [
      ...agent.processingLogs.slice(-2),
      `⚡ ${Math.round(agent.performance.successRate * 100)}% success | ${Math.round(agent.performance.averageResponseTime)}ms avg`,
    ];
    setDisplayLogs(enhancedLogs);
  }, [agent.processingLogs, agent.performance]);

  if (!isHovered && displayLogs.length === 0) return null;

  return (
    <Html
      position={[position.x, position.y + 3, position.z]}
      center
      style={{
        width: '180px',
        height: isHovered ? '60px' : '45px',
        pointerEvents: 'none'
      }}
    >
      <div style={{
        width: '100%',
        height: '100%',
        background: 'rgba(0, 0, 0, 0.9)',
        border: `1px solid ${AGENT_TYPE_COLORS[agent.type]}`,
        borderRadius: '4px',
        padding: '4px',
        fontSize: '9px',
        fontFamily: 'monospace',
        color: AGENT_TYPE_COLORS[agent.type],
        overflow: 'hidden',
        position: 'relative',
        boxShadow: `0 0 10px ${AGENT_TYPE_COLORS[agent.type]}40`
      }}>
        {/* Header with agent info */}
        {isHovered && (
          <div style={{
            fontSize: '8px',
            opacity: 0.8,
            borderBottom: `1px solid ${AGENT_TYPE_COLORS[agent.type]}40`,
            marginBottom: '2px',
            paddingBottom: '1px'
          }}>
            {agent.type} | {agent.state} | {agent.goals.length} goals
          </div>
        )}
        
        {/* Activity logs */}
        {displayLogs.map((log, i) => (
          <div
            key={i}
            style={{
              position: 'absolute',
              top: isHovered ? `${14 + i * 14}px` : `${i * 14}px`,
              left: '4px',
              right: '4px',
              height: '14px',
              lineHeight: '14px',
              opacity: 1 - (i * 0.25),
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'clip',
              animation: 'slideUp 0.5s ease-out'
            }}
          >
            {log}
          </div>
        ))}
        
        <style>{`
          @keyframes slideUp {
            from { transform: translateY(14px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
          }
        `}</style>
      </div>
    </Html>
  );
};

// Main enhanced agent node component
export const EnhancedAgentNode: React.FC<EnhancedAgentNodeProps> = ({
  agent,
  position,
  isSelected,
  isHovered,
  showPerformanceRing,
  showCapabilityBadges,
  qualityLevel,
  onSelect,
  onHover
}) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const glowRef = useRef<THREE.Mesh>(null);
  
  const agentColor = new THREE.Color(AGENT_TYPE_COLORS[agent.type]);
  const baseSize = isSelected ? 1.2 : 1.0;
  
  // Agent geometry based on type
  const geometry = useMemo(() => 
    getAgentGeometry(agent.type, baseSize), 
    [agent.type, baseSize]
  );

  // Animate main node
  useFrame((state) => {
    if (!meshRef.current) return;

    // Gentle rotation for all agents
    meshRef.current.rotation.y += 0.005;

    // Special animations based on state
    if (agent.state === 'COORDINATING') {
      meshRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 2) * 0.1;
    }

    // Glow effect
    if (glowRef.current) {
      const glowIntensity = isHovered ? 1.5 : isSelected ? 1.2 : 0.8;
      glowRef.current.scale.setScalar(glowIntensity);
      (glowRef.current.material as THREE.MeshBasicMaterial).opacity = 
        (0.1 + agent.performance.resourceUtilization * 0.2) * glowIntensity;
    }
  });

  return (
    <group>
      {/* Main agent node */}
      <mesh
        ref={meshRef}
        geometry={geometry}
        position={position}
        onClick={() => onSelect(agent.id.id)}
        onPointerOver={() => onHover(agent.id.id)}
        onPointerOut={() => onHover(null)}
      >
        <meshStandardMaterial
          color={agentColor}
          emissive={agentColor}
          emissiveIntensity={isSelected ? 0.4 : 0.2}
          metalness={0.8}
          roughness={0.2}
          transparent
          opacity={0.9}
        />
      </mesh>

      {/* Glow effect */}
      <mesh ref={glowRef} position={position}>
        <sphereGeometry args={[baseSize * 1.5, 16, 16]} />
        <meshBasicMaterial
          color={agentColor}
          transparent
          opacity={0.1}
          side={THREE.BackSide}
        />
      </mesh>

      {/* Performance ring */}
      <PerformanceRing
        agent={agent}
        position={position}
        showRing={showPerformanceRing}
      />

      {/* Capability badges */}
      <CapabilityBadges
        capabilities={agent.capabilities}
        position={position}
        showBadges={showCapabilityBadges}
      />

      {/* State indicator */}
      <StateIndicator
        state={agent.state}
        position={position}
      />

      {/* Activity pulse */}
      <ActivityPulse
        agent={agent}
        position={position}
        qualityLevel={qualityLevel}
      />

      {/* Enhanced activity monitor */}
      <EnhancedActivityMonitor
        agent={agent}
        position={position}
        isHovered={isHovered}
      />

      {/* Agent label */}
      <Billboard position={[position.x, position.y - 2, position.z]}>
        <Text
          fontSize={isSelected ? 0.6 : 0.4}
          color={isHovered ? '#FFFFFF' : agentColor}
          anchorX="center"
          anchorY="middle"
          outlineWidth={0.02}
          outlineColor="#000000"
        >
          {agent.name}
          {isSelected && (
            `\n${agent.type}\n${Math.round(agent.performance.successRate * 100)}% success`
          )}
        </Text>
      </Billboard>
    </group>
  );
};

export default EnhancedAgentNode;