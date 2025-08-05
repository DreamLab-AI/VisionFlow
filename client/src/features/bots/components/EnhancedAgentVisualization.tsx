import React, { useRef, useEffect, useState, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { Html, Text, Billboard, Line as DreiLine, Sphere, Box, Octahedron, Icosahedron, Dodecahedron, Tetrahedron } from '@react-three/drei';
import { BotsAgent, MessageFlowEvent, CoordinationPattern } from '../types/BotsTypes';
import { createLogger } from '../../../utils/logger';
import { useBotsData } from '../contexts/BotsDataContext';

const logger = createLogger('EnhancedAgentVisualization');

// Enhanced spring-physics constants for hive mind behavior
const SPRING_PHYSICS = {
  springStrength: 0.1,
  linkDistance: 8.0,
  damping: 0.95,
  nodeRepulsion: 500.0,
  gravityStrength: 0.02,
  maxVelocity: 2.0,
  
  // Hive-mind specific forces
  queenGravity: 0.05,
  swarmCohesion: 0.08,
  hierarchicalForce: 0.03,
  
  // Message flow forces
  messageAttraction: 0.15,
  communicationDecay: 0.98,
};

// Agent type to geometry mapping for spring-physics directed graph
const getAgentGeometry = (agentType: string): THREE.BufferGeometry => {
  switch (agentType) {
    case 'queen':
      return new THREE.OctahedronGeometry(2.5); // Largest, most distinctive
    case 'coordinator':
      return new THREE.IcosahedronGeometry(1.8); // Complex coordination shape
    case 'architect':
      return new THREE.DodecahedronGeometry(1.5); // Architectural precision
    case 'specialist':
      return new THREE.ConeGeometry(1.2, 2.0, 8); // Pointed specialization
    case 'coder':
      return new THREE.BoxGeometry(1.0, 1.0, 1.0); // Structured coding
    case 'researcher':
      return new THREE.SphereGeometry(1.0); // Rounded knowledge
    case 'tester':
      return new THREE.CylinderGeometry(0.8, 0.8, 1.2, 8); // Testing pillar
    case 'analyst':
      return new THREE.TorusGeometry(0.8, 0.3, 8, 16); // Analysis ring
    case 'optimizer':
      return new THREE.TetrahedronGeometry(1.1); // Sharp optimization
    case 'monitor':
      return new THREE.RingGeometry(0.5, 1.0, 16); // Monitoring ring
    default:
      return new THREE.SphereGeometry(0.8); // Default sphere
  }
};

// Enhanced color palette for hive mind visualization
const getAgentColor = (agentType: string, status: string): THREE.Color => {
  const baseColors = {
    queen: '#FFD700',           // Royal gold
    coordinator: '#F1C40F',     // Golden coordination
    architect: '#E67E22',       // Architectural orange
    specialist: '#8E44AD',      // Purple specialization
    coder: '#2ECC71',          // Green execution
    researcher: '#3498DB',      // Blue knowledge
    tester: '#E74C3C',         // Red validation
    analyst: '#9B59B6',        // Purple analysis
    optimizer: '#F39C12',      // Orange optimization
    monitor: '#1ABC9C',        // Teal monitoring
    default: '#95A5A6',        // Gray default
  };

  const baseColor = new THREE.Color(baseColors[agentType] || baseColors.default);

  // Modify color based on status
  switch (status) {
    case 'active':
      return baseColor.clone().multiplyScalar(1.2); // Brighter
    case 'busy':
      return baseColor.clone().lerp(new THREE.Color('#FF6B6B'), 0.3); // Red tint
    case 'idle':
      return baseColor.clone().multiplyScalar(0.6); // Dimmer
    case 'error':
      return new THREE.Color('#E74C3C'); // Error red
    default:
      return baseColor;
  }
};

// Performance ring component with spring-physics inspired animation
const PerformanceRing: React.FC<{ 
  agent: BotsAgent; 
  position: THREE.Vector3; 
}> = ({ agent, position }) => {
  const ringRef = useRef<THREE.Mesh>(null);
  const [pulsePhase, setPulsePhase] = useState(Math.random() * Math.PI * 2);

  useFrame((state) => {
    if (!ringRef.current) return;

    const time = state.clock.elapsedTime;
    const successRate = (agent.successRate || 80) / 100;
    const activity = agent.activity || 0.5;

    // Color interpolation based on performance
    const color = new THREE.Color().lerpColors(
      new THREE.Color('#E74C3C'), // Red for poor performance
      new THREE.Color('#2ECC71'), // Green for excellent performance
      successRate
    );

    // Spring-physics inspired pulse animation
    const pulseSpeed = activity * 3 + 1;
    const pulseIntensity = 0.2 + activity * 0.3;
    const scale = 1 + Math.sin(time * pulseSpeed + pulsePhase) * pulseIntensity;

    ringRef.current.scale.setScalar(scale);
    (ringRef.current.material as THREE.MeshBasicMaterial).color = color;
    (ringRef.current.material as THREE.MeshBasicMaterial).opacity = 0.7 + Math.sin(time * pulseSpeed) * 0.2;
  });

  return (
    <mesh ref={ringRef} position={position}>
      <torusGeometry args={[1.5, 0.15, 8, 32]} />
      <meshBasicMaterial transparent />
    </mesh>
  );
};

// Capability badges with orbital animation
const CapabilityBadges: React.FC<{ 
  agent: BotsAgent; 
  position: THREE.Vector3; 
}> = ({ agent, position }) => {
  const capabilities = agent.capabilities?.slice(0, 3) || [];
  
  return (
    <>
      {capabilities.map((capability, index) => {
        const angle = (index / capabilities.length) * Math.PI * 2;
        const radius = 2.5;
        const orbitPosition = new THREE.Vector3(
          position.x + Math.cos(angle) * radius,
          position.y + 0.5,
          position.z + Math.sin(angle) * radius
        );

        // Get capability icon
        const icon = getCapabilityIcon(capability);
        const color = getCapabilityColor(capability);

        return (
          <Billboard key={`${agent.id}-${capability}`} position={orbitPosition}>
            <Text
              fontSize={0.4}
              color={color}
              anchorX="center"
              anchorY="middle"
              font="/fonts/roboto-mono.woff"
            >
              {icon}
            </Text>
          </Billboard>
        );
      })}
    </>
  );
};

// State indicator with spring-physics glow effect
const StateIndicator: React.FC<{ 
  agent: BotsAgent; 
  position: THREE.Vector3; 
}> = ({ agent, position }) => {
  const indicatorRef = useRef<THREE.Mesh>(null);

  const stateColors = {
    active: '#2ECC71',    // Green
    busy: '#F39C12',      // Orange
    idle: '#95A5A6',      // Gray
    error: '#E74C3C',     // Red
    initializing: '#3498DB', // Blue
    terminating: '#8E44AD',  // Purple
  };

  const stateColor = new THREE.Color(stateColors[agent.status] || stateColors.idle);

  useFrame((state) => {
    if (!indicatorRef.current) return;

    const time = state.clock.elapsedTime;
    
    // Pulsing animation for active states
    if (agent.status === 'active' || agent.status === 'busy') {
      const pulse = 1 + Math.sin(time * 4) * 0.3;
      indicatorRef.current.scale.setScalar(pulse);
    }
    
    // Flickering for error state
    if (agent.status === 'error') {
      const flicker = Math.random() > 0.1 ? 1 : 0.3;
      (indicatorRef.current.material as THREE.MeshBasicMaterial).opacity = flicker;
    }
  });

  const indicatorPosition = new THREE.Vector3(
    position.x,
    position.y + 2.0,
    position.z
  );

  return (
    <mesh ref={indicatorRef} position={indicatorPosition}>
      <sphereGeometry args={[0.3, 16, 16]} />
      <meshBasicMaterial color={stateColor} transparent />
    </mesh>
  );
};

// Enhanced agent node with spring-physics behavior
const EnhancedAgentNode: React.FC<{ 
  agent: BotsAgent; 
  index: number;
}> = ({ agent, index }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const positionRef = useRef(new THREE.Vector3(
    (Math.random() - 0.5) * 20,
    (Math.random() - 0.5) * 20,
    (Math.random() - 0.5) * 20
  ));
  const velocityRef = useRef(new THREE.Vector3(0, 0, 0));
  const { agents, messageFlow } = useBotsData();

  // Spring-physics update
  useFrame((state, delta) => {
    if (!meshRef.current) return;

    const position = positionRef.current;
    const velocity = velocityRef.current;
    const force = new THREE.Vector3(0, 0, 0);

    // Apply spring forces to connected agents
    const agentArray = Array.from(agents.values());
    
    agentArray.forEach((otherAgent, otherIndex) => {
      if (otherAgent.id === agent.id) return;

      const otherPosition = new THREE.Vector3(
        otherAgent.position?.x || 0,
        otherAgent.position?.y || 0,
        otherAgent.position?.z || 0
      );

      const distance = position.distanceTo(otherPosition);
      const direction = otherPosition.clone().sub(position).normalize();

      // Repulsion force (prevent overlap)
      if (distance < SPRING_PHYSICS.linkDistance) {
        const repulsion = SPRING_PHYSICS.nodeRepulsion / (distance * distance + 1);
        force.sub(direction.clone().multiplyScalar(repulsion));
      }

      // Attraction for communication partners
      const hasRecentCommunication = messageFlow.some(msg => 
        (msg.from_agent === agent.id && msg.to_agent === otherAgent.id) ||
        (msg.from_agent === otherAgent.id && msg.to_agent === agent.id)
      );

      if (hasRecentCommunication) {
        const attraction = SPRING_PHYSICS.messageAttraction * (distance - SPRING_PHYSICS.linkDistance);
        force.add(direction.clone().multiplyScalar(attraction));
      }
    });

    // Queen gravity (coordinator agents attract others)
    if (agent.type !== 'coordinator') {
      const coordinators = agentArray.filter(a => a.type === 'coordinator');
      coordinators.forEach(coordinator => {
        const coordPosition = new THREE.Vector3(
          coordinator.position?.x || 0,
          coordinator.position?.y || 0,
          coordinator.position?.z || 0
        );
        const distance = position.distanceTo(coordPosition);
        const direction = coordPosition.clone().sub(position).normalize();
        const gravity = SPRING_PHYSICS.queenGravity / (distance + 1);
        force.add(direction.multiplyScalar(gravity));
      });
    }

    // Central gravity to prevent drift
    const centerForce = position.clone().multiplyScalar(-SPRING_PHYSICS.gravityStrength);
    force.add(centerForce);

    // Update velocity and position
    velocity.add(force.multiplyScalar(delta));
    velocity.multiplyScalar(SPRING_PHYSICS.damping);
    
    // Clamp velocity
    if (velocity.length() > SPRING_PHYSICS.maxVelocity) {
      velocity.normalize().multiplyScalar(SPRING_PHYSICS.maxVelocity);
    }

    position.add(velocity.clone().multiplyScalar(delta));

    // Update mesh position
    meshRef.current.position.copy(position);

    // Update agent position data
    if (agent.position) {
      agent.position.x = position.x;
      agent.position.y = position.y;
      agent.position.z = position.z;
    }
  });

  const geometry = useMemo(() => getAgentGeometry(agent.type), [agent.type]);
  const color = useMemo(() => getAgentColor(agent.type, agent.status), [agent.type, agent.status]);

  return (
    <group>
      {/* Main agent node */}
      <mesh ref={meshRef} geometry={geometry}>
        <meshPhongMaterial color={color} />
      </mesh>

      {/* Performance ring */}
      <PerformanceRing agent={agent} position={positionRef.current} />

      {/* Capability badges */}
      <CapabilityBadges agent={agent} position={positionRef.current} />

      {/* State indicator */}
      <StateIndicator agent={agent} position={positionRef.current} />

      {/* Agent label */}
      <Billboard position={[positionRef.current.x, positionRef.current.y - 2.5, positionRef.current.z]}>
        <Text
          fontSize={0.5}
          color="#FFFFFF"
          anchorX="center"
          anchorY="middle"
          font="/fonts/roboto-mono.woff"
        >
          {agent.name || agent.id}
        </Text>
      </Billboard>
    </group>
  );
};

// Message flow visualization with particle effects
const MessageFlowVisualization: React.FC = () => {
  const { messageFlow, agents } = useBotsData();
  const particleRefs = useRef<Map<string, THREE.Mesh>>(new Map());

  useFrame((state, delta) => {
    const time = state.clock.elapsedTime;

    messageFlow.forEach(message => {
      const sourceAgent = agents.get(message.from_agent);
      const targetAgent = agents.get(message.to_agent);

      if (!sourceAgent || !targetAgent) return;

      const sourcePos = new THREE.Vector3(
        sourceAgent.position?.x || 0,
        sourceAgent.position?.y || 0,
        sourceAgent.position?.z || 0
      );

      const targetPos = new THREE.Vector3(
        targetAgent.position?.x || 0,
        targetAgent.position?.y || 0,
        targetAgent.position?.z || 0
      );

      // Animate particle along path
      const progress = ((time * 2) % 2) / 2; // 2-second travel time
      const currentPos = sourcePos.clone().lerp(targetPos, progress);

      const particle = particleRefs.current.get(message.id);
      if (particle) {
        particle.position.copy(currentPos);
      }
    });
  });

  return (
    <>
      {messageFlow.map(message => (
        <mesh
          key={message.id}
          ref={(ref) => {
            if (ref) particleRefs.current.set(message.id, ref);
          }}
        >
          <sphereGeometry args={[0.1, 8, 8]} />
          <meshBasicMaterial 
            color={getMessageColor(message.message_type)} 
            transparent
            opacity={0.8}
          />
        </mesh>
      ))}
    </>
  );
};

// Communication links with spring-physics inspired rendering
const CommunicationLinks: React.FC = () => {
  const { agents, messageFlow } = useBotsData();
  
  const links = useMemo(() => {
    const linkMap = new Map<string, { count: number; latency: number }>();
    
    messageFlow.forEach(message => {
      const key = [message.from_agent, message.to_agent].sort().join('-');
      const existing = linkMap.get(key) || { count: 0, latency: 0 };
      linkMap.set(key, {
        count: existing.count + 1,
        latency: existing.latency + message.latency_ms
      });
    });

    return Array.from(linkMap.entries()).map(([key, data]) => {
      const [from, to] = key.split('-');
      return { from, to, count: data.count, avgLatency: data.latency / data.count };
    });
  }, [messageFlow]);

  return (
    <>
      {links.map(link => {
        const sourceAgent = agents.get(link.from);
        const targetAgent = agents.get(link.to);

        if (!sourceAgent || !targetAgent) return null;

        const sourcePos = [
          sourceAgent.position?.x || 0,
          sourceAgent.position?.y || 0,
          sourceAgent.position?.z || 0
        ] as [number, number, number];

        const targetPos = [
          targetAgent.position?.x || 0,
          targetAgent.position?.y || 0,
          targetAgent.position?.z || 0
        ] as [number, number, number];

        const linkColor = new THREE.Color().setHSL(
          0.3, // Green hue
          0.8,
          Math.min(link.count / 10, 1) // Brightness based on message count
        );

        return (
          <DreiLine
            key={`${link.from}-${link.to}`}
            points={[sourcePos, targetPos]}
            color={linkColor}
            lineWidth={Math.max(1, Math.min(link.count, 5))}
            transparent
            opacity={0.4}
          />
        );
      })}
    </>
  );
};

// Main enhanced agent visualization component
const EnhancedAgentVisualization: React.FC = () => {
  const { agents } = useBotsData();
  const agentArray = Array.from(agents.values());

  useEffect(() => {
    logger.info(`Rendering ${agentArray.length} enhanced agents with spring-physics`);
  }, [agentArray.length]);

  return (
    <group>
      {/* Enhanced agent nodes */}
      {agentArray.map((agent, index) => (
        <EnhancedAgentNode key={agent.id} agent={agent} index={index} />
      ))}

      {/* Message flow visualization */}
      <MessageFlowVisualization />

      {/* Communication links */}
      <CommunicationLinks />

      {/* Ambient lighting for better visualization */}
      <ambientLight intensity={0.3} />
      <pointLight position={[10, 10, 10]} intensity={0.5} />
      <pointLight position={[-10, -10, -10]} intensity={0.3} />
    </group>
  );
};

// Utility functions
const getCapabilityIcon = (capability: string): string => {
  const icons = {
    swarm_orchestration: '👑',
    system_design: '🏗️',
    rust_development: '🦀',
    code_analysis: '🔍',
    unit_testing: '🧪',
    metrics_collection: '📊',
    gpu_shader_optimization: '⚡',
    system_monitoring: '👁️',
    default: '⚙️',
  };

  return icons[capability] || icons.default;
};

const getCapabilityColor = (capability: string): string => {
  const colors = {
    swarm_orchestration: '#FFD700',
    system_design: '#E67E22',
    rust_development: '#2ECC71',
    code_analysis: '#3498DB',
    unit_testing: '#E74C3C',
    metrics_collection: '#9B59B6',
    gpu_shader_optimization: '#F39C12',
    system_monitoring: '#1ABC9C',
    default: '#95A5A6',
  };

  return colors[capability] || colors.default;
};

const getMessageColor = (messageType: string): string => {
  const colors = {
    coordination: '#F1C40F',
    task_assignment: '#3498DB',
    status_update: '#2ECC71',
    error_report: '#E74C3C',
    data_transfer: '#9B59B6',
    default: '#95A5A6',
  };

  return colors[messageType] || colors.default;
};

export default EnhancedAgentVisualization;