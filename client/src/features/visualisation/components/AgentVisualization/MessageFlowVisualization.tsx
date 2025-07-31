/**
 * Message Flow and Connection Visualization
 * Real-time visualization of agent communication and coordination patterns
 */
import React, { useRef, useMemo, useEffect, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import { Line, Trail, Text, Billboard } from '@react-three/drei';
import * as THREE from 'three';
import {
  MessageFlowData,
  ConnectionVisualization,
  CoordinationPattern,
  AgentNodeData,
  MessageFlowProps,
  CoordinationVisualizationProps,
  AgentId
} from './types';

// Message Particle Component
const MessageParticle: React.FC<{
  message: MessageFlowData;
  sourcePosition: THREE.Vector3;
  targetPosition: THREE.Vector3;
  onComplete: () => void;
}> = ({ message, sourcePosition, targetPosition, onComplete }) => {
  const particleRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.MeshBasicMaterial>(null);
  const [progress, setProgress] = useState(0);
  
  const startTime = useRef(Date.now());
  const duration = useMemo(() => {
    // Duration based on priority and distance
    const distance = sourcePosition.distanceTo(targetPosition);
    const baseDuration = Math.max(1000, distance * 100); // 1s minimum
    const priorityMultiplier = {
      urgent: 0.5,
      high: 0.7,
      normal: 1.0,
      low: 1.5
    }[message.priority];
    return baseDuration * priorityMultiplier;
  }, [sourcePosition, targetPosition, message.priority]);

  const color = useMemo(() => {
    const colors = {
      request: '#00ff00',
      response: '#0080ff',
      inform: '#ffff00',
      query: '#ff8000',
      command: '#ff0040',
      broadcast: '#ff00ff'
    };
    return colors[message.type] || '#ffffff';
  }, [message.type]);

  useFrame(() => {
    if (particleRef.current && materialRef.current) {
      const elapsed = Date.now() - startTime.current;
      const newProgress = Math.min(elapsed / duration, 1);
      setProgress(newProgress);

      // Update position along the path
      const currentPosition = new THREE.Vector3();
      currentPosition.lerpVectors(sourcePosition, targetPosition, newProgress);
      particleRef.current.position.copy(currentPosition);

      // Animate particle properties
      const size = 0.1 + Math.sin(elapsed * 0.01) * 0.05;
      particleRef.current.scale.setScalar(size);

      // Pulse based on priority
      const pulseSpeed = {
        urgent: 10,
        high: 6,
        normal: 3,
        low: 1
      }[message.priority];
      const pulse = Math.sin(elapsed * 0.01 * pulseSpeed) * 0.3 + 0.7;
      materialRef.current.opacity = pulse;

      // Complete animation
      if (newProgress >= 1) {
        onComplete();
      }
    }
  });

  return (
    <mesh ref={particleRef}>
      <sphereGeometry args={[0.1, 8, 8]} />
      <meshBasicMaterial
        ref={materialRef}
        color={color}
        transparent
        opacity={0.8}
      />
    </mesh>
  );
};

// Connection Line Component
const ConnectionLine: React.FC<{
  connection: ConnectionVisualization;
  sourcePosition: THREE.Vector3;
  targetPosition: THREE.Vector3;
  showLatency: boolean;
}> = ({ connection, sourcePosition, targetPosition, showLatency }) => {
  const lineRef = useRef<any>(null);
  const materialRef = useRef<THREE.LineBasicMaterial>(null);

  useFrame((state) => {
    if (materialRef.current) {
      const time = state.clock.getElapsedTime();
      
      // Animate based on connection strength and activity
      const activity = Math.min(connection.messageCount / 100, 1);
      const pulse = Math.sin(time * 2 + connection.strength * 10) * 0.3 + 0.7;
      materialRef.current.opacity = connection.strength * 0.3 + activity * 0.4 * pulse;

      // Color based on reliability
      const hue = connection.reliability * 0.33; // Green for high reliability
      materialRef.current.color.setHSL(hue, 0.8, 0.6);
    }
  });

  const lineWidth = useMemo(() => {
    return Math.max(1, connection.strength * 5 + connection.messageCount * 0.01);
  }, [connection.strength, connection.messageCount]);

  const points = useMemo(() => [
    sourcePosition,
    targetPosition
  ], [sourcePosition, targetPosition]);

  return (
    <group>
      <Line
        ref={lineRef}
        points={points}
        color="#00ffff"
        lineWidth={lineWidth}
        transparent
      >
        <lineBasicMaterial
          ref={materialRef}
          transparent
          opacity={0.6}
        />
      </Line>
      
      {/* Latency indicator */}
      {showLatency && connection.latency > 100 && (
        <Billboard position={new THREE.Vector3().lerpVectors(sourcePosition, targetPosition, 0.5)}>
          <Text
            fontSize={0.1}
            color={connection.latency > 1000 ? '#ff4444' : '#ffaa00'}
            anchorX="center"
            anchorY="middle"
          >
            {Math.round(connection.latency)}ms
          </Text>
        </Billboard>
      )}
    </group>
  );
};

// Coordination Pattern Overlay
const CoordinationPatternOverlay: React.FC<{
  pattern: CoordinationPattern;
  agentPositions: Map<string, THREE.Vector3>;
}> = ({ pattern, agentPositions }) => {
  const groupRef = useRef<THREE.Group>(null);
  const materialRef = useRef<THREE.MeshBasicMaterial>(null);

  const participantPositions = useMemo(() => {
    return pattern.participants
      .map(p => agentPositions.get(`${p.namespace || 'default'}:${p.id}`))
      .filter(Boolean) as THREE.Vector3[];
  }, [pattern.participants, agentPositions]);

  const center = useMemo(() => {
    if (participantPositions.length === 0) return new THREE.Vector3();
    const center = new THREE.Vector3();
    participantPositions.forEach(pos => center.add(pos));
    center.divideScalar(participantPositions.length);
    return center;
  }, [participantPositions]);

  const radius = useMemo(() => {
    if (participantPositions.length === 0) return 1;
    return Math.max(
      ...participantPositions.map(pos => pos.distanceTo(center))
    ) + 1;
  }, [participantPositions, center]);

  useFrame((state) => {
    if (groupRef.current && materialRef.current) {
      const time = state.clock.getElapsedTime();
      
      // Rotate based on pattern type
      const rotationSpeed = {
        hierarchical: 0.5,
        mesh: 1.0,
        pipeline: 0.3,
        consensus: 0.8,
        barrier: 0.2
      }[pattern.type] || 0.5;
      
      groupRef.current.rotation.y = time * rotationSpeed;

      // Animate opacity based on pattern status and progress
      const baseOpacity = {
        forming: 0.3,
        active: 0.6,
        completing: 0.8,
        dissolved: 0.1
      }[pattern.status] || 0.3;

      const progressPulse = Math.sin(time * 3) * 0.1 + 0.9;
      materialRef.current.opacity = baseOpacity * progressPulse * pattern.progress;

      // Color based on efficiency
      const hue = pattern.efficiency * 0.33;
      materialRef.current.color.setHSL(hue, 0.8, 0.5);
    }
  });

  const patternColor = useMemo(() => {
    const colors = {
      hierarchical: '#ff6600',
      mesh: '#00ff66',
      pipeline: '#6600ff',
      consensus: '#ffff00',
      barrier: '#ff0066'
    };
    return colors[pattern.type] || '#ffffff';
  }, [pattern.type]);

  if (participantPositions.length < 2) return null;

  return (
    <group ref={groupRef} position={center}>
      {/* Pattern visualization based on type */}
      {pattern.type === 'hierarchical' && (
        <HierarchicalPattern
          positions={participantPositions}
          center={center}
          color={patternColor}
          materialRef={materialRef}
        />
      )}
      
      {pattern.type === 'mesh' && (
        <MeshPattern
          positions={participantPositions}
          center={center}
          color={patternColor}
          materialRef={materialRef}
        />
      )}
      
      {pattern.type === 'pipeline' && (
        <PipelinePattern
          positions={participantPositions}
          center={center}
          color={patternColor}
          materialRef={materialRef}
        />
      )}
      
      {pattern.type === 'consensus' && (
        <ConsensusPattern
          positions={participantPositions}
          center={center}
          radius={radius}
          progress={pattern.progress}
          color={patternColor}
          materialRef={materialRef}
        />
      )}
      
      {pattern.type === 'barrier' && (
        <BarrierPattern
          positions={participantPositions}
          center={center}
          radius={radius}
          progress={pattern.progress}
          color={patternColor}
          materialRef={materialRef}
        />
      )}

      {/* Pattern label */}
      <Billboard position={[0, radius + 0.5, 0]}>
        <Text
          fontSize={0.2}
          color={patternColor}
          anchorX="center"
          anchorY="bottom"
          outlineWidth={0.02}
          outlineColor="black"
        >
          {pattern.type.toUpperCase()}
        </Text>
        <Text
          fontSize={0.15}
          color="#aaaaaa"
          anchorX="center"
          anchorY="top"
          position={[0, -0.3, 0]}
        >
          {Math.round(pattern.progress * 100)}% | {pattern.participants.length} agents
        </Text>
      </Billboard>
    </group>
  );
};

// Pattern-specific visualization components
const HierarchicalPattern: React.FC<{
  positions: THREE.Vector3[];
  center: THREE.Vector3;
  color: string;
  materialRef: React.RefObject<THREE.MeshBasicMaterial>;
}> = ({ positions, center, color, materialRef }) => {
  const lines = useMemo(() => {
    if (positions.length === 0) return [];
    const centerRelative = positions[0].clone().sub(center);
    return positions.slice(1).map(pos => [
      centerRelative,
      pos.clone().sub(center)
    ]);
  }, [positions, center]);

  return (
    <>
      {lines.map((line, index) => (
        <Line
          key={index}
          points={line}
          color={color}
          lineWidth={2}
          transparent
        >
          <lineBasicMaterial ref={materialRef} transparent />
        </Line>
      ))}
    </>
  );
};

const MeshPattern: React.FC<{
  positions: THREE.Vector3[];
  center: THREE.Vector3;
  color: string;
  materialRef: React.RefObject<THREE.MeshBasicMaterial>;
}> = ({ positions, center, color, materialRef }) => {
  const lines = useMemo(() => {
    const relativePositions = positions.map(pos => pos.clone().sub(center));
    const connections = [];
    for (let i = 0; i < relativePositions.length; i++) {
      for (let j = i + 1; j < relativePositions.length; j++) {
        connections.push([relativePositions[i], relativePositions[j]]);
      }
    }
    return connections;
  }, [positions, center]);

  return (
    <>
      {lines.map((line, index) => (
        <Line
          key={index}
          points={line}
          color={color}
          lineWidth={1}
          transparent
        >
          <lineBasicMaterial ref={materialRef} transparent />
        </Line>
      ))}
    </>
  );
};

const PipelinePattern: React.FC<{
  positions: THREE.Vector3[];
  center: THREE.Vector3;
  color: string;
  materialRef: React.RefObject<THREE.MeshBasicMaterial>;
}> = ({ positions, center, color, materialRef }) => {
  const lines = useMemo(() => {
    const relativePositions = positions.map(pos => pos.clone().sub(center));
    const connections = [];
    for (let i = 0; i < relativePositions.length - 1; i++) {
      connections.push([relativePositions[i], relativePositions[i + 1]]);
    }
    return connections;
  }, [positions, center]);

  return (
    <>
      {lines.map((line, index) => (
        <Line
          key={index}
          points={line}
          color={color}
          lineWidth={3}
          transparent
        >
          <lineBasicMaterial ref={materialRef} transparent />
        </Line>
      ))}
    </>
  );
};

const ConsensusPattern: React.FC<{
  positions: THREE.Vector3[];
  center: THREE.Vector3;
  radius: number;
  progress: number;
  color: string;
  materialRef: React.RefObject<THREE.MeshBasicMaterial>;
}> = ({ positions, center, radius, progress, color, materialRef }) => {
  return (
    <mesh>
      <ringGeometry args={[radius * 0.8, radius, 32, 1, 0, Math.PI * 2 * progress]} />
      <meshBasicMaterial
        ref={materialRef}
        color={color}
        transparent
        side={THREE.DoubleSide}
      />
    </mesh>
  );
};

const BarrierPattern: React.FC<{
  positions: THREE.Vector3[];
  center: THREE.Vector3;
  radius: number;
  progress: number;
  color: string;
  materialRef: React.RefObject<THREE.MeshBasicMaterial>;
}> = ({ positions, center, radius, progress, color, materialRef }) => {
  return (
    <mesh>
      <cylinderGeometry args={[radius, radius, 0.1, 32, 1, false, 0, Math.PI * 2 * progress]} />
      <meshBasicMaterial
        ref={materialRef}
        color={color}
        transparent
        wireframe
      />
    </mesh>
  );
};

// Main Message Flow Visualization Component
export const MessageFlowVisualization: React.FC<MessageFlowProps> = ({
  connections,
  messages,
  animationSpeed,
  showLatency
}) => {
  const [activeMessages, setActiveMessages] = useState<Map<string, MessageFlowData>>(new Map());
  const agentPositions = useRef<Map<string, THREE.Vector3>>(new Map());

  // Update agent positions (would be passed from parent)
  useEffect(() => {
    // This would be updated from the parent component with actual agent positions
    // For now, we'll use placeholder positions
  }, []);

  const handleMessageComplete = (messageId: string) => {
    setActiveMessages(prev => {
      const newMap = new Map(prev);
      newMap.delete(messageId);
      return newMap;
    });
  };

  // Add new messages to active messages
  useEffect(() => {
    messages.forEach(message => {
      if (!activeMessages.has(message.id) && message.status === 'sending') {
        setActiveMessages(prev => new Map(prev).set(message.id, message));
      }
    });
  }, [messages, activeMessages]);

  return (
    <group>
      {/* Connection lines */}
      {connections.map(connection => {
        const sourcePos = agentPositions.current.get(`${connection.source.namespace || 'default'}:${connection.source.id}`);
        const targetPos = agentPositions.current.get(`${connection.target.namespace || 'default'}:${connection.target.id}`);
        
        if (!sourcePos || !targetPos) return null;

        return (
          <ConnectionLine
            key={`${connection.source.id}-${connection.target.id}`}
            connection={connection}
            sourcePosition={sourcePos}
            targetPosition={targetPos}
            showLatency={showLatency}
          />
        );
      })}

      {/* Active message particles */}
      {Array.from(activeMessages.values()).map(message => {
        const sourcePos = agentPositions.current.get(`${message.from.namespace || 'default'}:${message.from.id}`);
        
        // Handle both single and multiple targets
        const targets = Array.isArray(message.to) ? message.to : [message.to];
        
        return targets.map(target => {
          const targetPos = agentPositions.current.get(`${target.namespace || 'default'}:${target.id}`);
          
          if (!sourcePos || !targetPos) return null;

          return (
            <MessageParticle
              key={`${message.id}-${target.id}`}
              message={message}
              sourcePosition={sourcePos}
              targetPosition={targetPos}
              onComplete={() => handleMessageComplete(message.id)}
            />
          );
        });
      })}
    </group>
  );
};

// Coordination Pattern Visualization Component
export const CoordinationVisualization: React.FC<CoordinationVisualizationProps> = ({
  patterns,
  agents,
  showHierarchy,
  animateFormation
}) => {
  const agentPositions = useMemo(() => {
    const positions = new Map<string, THREE.Vector3>();
    agents.forEach((agent, id) => {
      positions.set(id, agent.position);
    });
    return positions;
  }, [agents]);

  return (
    <group>
      {patterns
        .filter(pattern => pattern.status !== 'dissolved')
        .map(pattern => (
          <CoordinationPatternOverlay
            key={pattern.id}
            pattern={pattern}
            agentPositions={agentPositions}
          />
        ))}
    </group>
  );
};

export default {
  MessageFlowVisualization,
  CoordinationVisualization,
  MessageParticle,
  ConnectionLine
};