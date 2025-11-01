import React, { useEffect, useRef, useMemo } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { useSettingsStore } from '@/stores/settingsStore';
import { Text } from '@react-three/drei';



interface AgentNode {
  id: string;
  type: string;
  status: 'active' | 'idle' | 'error' | 'warning';
  health: number;
  cpuUsage: number;
  memoryUsage: number;
  workload: number;
  currentTask?: string;
  position?: { x: number; y: number; z: number };
  metadata?: Record<string, unknown>;
}

interface AgentConnection {
  source: string;
  target: string;
  type: 'communication' | 'coordination' | 'dependency';
  weight?: number;
}

interface AgentNodesLayerProps {
  agents: AgentNode[];
  connections?: AgentConnection[];
}

const STATUS_COLORS = {
  active: '#10b981',    
  idle: '#fbbf24',      
  error: '#ef4444',     
  warning: '#f97316'    
};

export const AgentNodesLayer: React.FC<AgentNodesLayerProps> = ({
  agents,
  connections = []
}) => {
  const { settings } = useSettingsStore();
  const groupRef = useRef<THREE.Group>(null);

  
  const showAgents = settings?.agents?.visualization?.show_in_graph ?? true;
  const nodeSize = settings?.agents?.visualization?.node_size ?? 1.5;
  const baseColor = settings?.agents?.visualization?.node_color ?? '#ff8800';
  const showConnections = settings?.agents?.visualization?.show_connections ?? true;
  const connectionColor = settings?.agents?.visualization?.connection_color ?? '#fbbf24';
  const animateActivity = settings?.agents?.visualization?.animate_activity ?? true;

  if (!showAgents || agents.length === 0) {
    return null;
  }

  return (
    <group ref={groupRef}>
      {}
      {agents.map((agent) => (
        <AgentNode
          key={agent.id}
          agent={agent}
          nodeSize={nodeSize}
          baseColor={baseColor}
          animateActivity={animateActivity}
        />
      ))}

      {}
      {showConnections && connections.map((connection, index) => (
        <AgentConnection
          key={`${connection.source}-${connection.target}-${index}`}
          connection={connection}
          agents={agents}
          color={connectionColor}
        />
      ))}
    </group>
  );
};


const AgentNode: React.FC<{
  agent: AgentNode;
  nodeSize: number;
  baseColor: string;
  animateActivity: boolean;
}> = ({ agent, nodeSize, baseColor, animateActivity }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const glowRef = useRef<THREE.Mesh>(null);
  const pulseRef = useRef({ phase: 0 });

  
  const position: [number, number, number] = agent.position
    ? [agent.position.x, agent.position.y, agent.position.z]
    : [Math.random() * 20 - 10, Math.random() * 20 - 10, Math.random() * 20 - 10];

  
  const statusColor = STATUS_COLORS[agent.status] || baseColor;

  
  const scaledSize = nodeSize * (1 + agent.workload / 100);

  
  useFrame((state, delta) => {
    if (!animateActivity || agent.status !== 'active') return;

    if (meshRef.current && glowRef.current) {
      
      pulseRef.current.phase += delta * 2;
      const pulseScale = 1 + Math.sin(pulseRef.current.phase) * 0.1;

      meshRef.current.scale.setScalar(pulseScale);
      glowRef.current.scale.setScalar(pulseScale * 1.3);

      
      meshRef.current.rotation.y += delta * 0.5;
    }
  });

  
  const geometry = useMemo(() => {
    switch (agent.type) {
      case 'researcher':
        return new THREE.OctahedronGeometry(scaledSize, 0);
      case 'coder':
        return new THREE.BoxGeometry(scaledSize * 1.5, scaledSize * 1.5, scaledSize * 1.5);
      case 'analyzer':
        return new THREE.TetrahedronGeometry(scaledSize, 0);
      case 'tester':
        return new THREE.ConeGeometry(scaledSize, scaledSize * 2, 6);
      case 'optimizer':
        return new THREE.TorusGeometry(scaledSize * 0.8, scaledSize * 0.3, 8, 12);
      case 'coordinator':
        return new THREE.IcosahedronGeometry(scaledSize, 0);
      default:
        return new THREE.SphereGeometry(scaledSize, 16, 16);
    }
  }, [agent.type, scaledSize]);

  return (
    <group position={position}>
      {}
      <mesh ref={glowRef}>
        <sphereGeometry args={[scaledSize * 1.5, 16, 16]} />
        <meshBasicMaterial
          color={statusColor}
          transparent
          opacity={0.2}
          side={THREE.BackSide}
        />
      </mesh>

      {}
      <mesh ref={meshRef} geometry={geometry}>
        <meshStandardMaterial
          color={statusColor}
          emissive={statusColor}
          emissiveIntensity={agent.status === 'active' ? 0.5 : 0.2}
          metalness={0.8}
          roughness={0.2}
        />
      </mesh>

      {}
      <Text
        position={[0, scaledSize + 1, 0]}
        fontSize={0.5}
        color={statusColor}
        anchorX="center"
        anchorY="bottom"
      >
        {agent.type.toUpperCase()}
      </Text>

      {}
      <Text
        position={[0, scaledSize + 1.5, 0]}
        fontSize={0.3}
        color="#ffffff"
        anchorX="center"
        anchorY="bottom"
      >
        {agent.status} • {agent.health}%
      </Text>

      {}
      {agent.currentTask && (
        <Text
          position={[0, -(scaledSize + 1), 0]}
          fontSize={0.25}
          color="#aaaaaa"
          anchorX="center"
          anchorY="top"
          maxWidth={10}
        >
          {agent.currentTask}
        </Text>
      )}

      {}
      <group position={[0, -(scaledSize + 0.5), 0]}>
        {}
        <mesh position={[0, 0, 0]}>
          <planeGeometry args={[2, 0.2]} />
          <meshBasicMaterial color="#333333" />
        </mesh>
        {}
        <mesh position={[-(1 - agent.health / 100), 0, 0.01]}>
          <planeGeometry args={[(agent.health / 100) * 2, 0.2]} />
          <meshBasicMaterial color={agent.health > 70 ? '#10b981' : agent.health > 40 ? '#fbbf24' : '#ef4444'} />
        </mesh>
      </group>

      {}
      {agent.status === 'active' && agent.workload > 0 && (
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[scaledSize * 1.8, 0.05, 8, 32, (agent.workload / 100) * Math.PI * 2]} />
          <meshBasicMaterial color="#00ffff" />
        </mesh>
      )}
    </group>
  );
};


const AgentConnection: React.FC<{
  connection: AgentConnection;
  agents: AgentNode[];
  color: string;
}> = ({ connection, agents, color }) => {
  const lineRef = useRef<THREE.Line>(null);

  
  const sourceAgent = agents.find(a => a.id === connection.source);
  const targetAgent = agents.find(a => a.id === connection.target);

  if (!sourceAgent || !targetAgent || !sourceAgent.position || !targetAgent.position) {
    return null;
  }

  const sourcePos = new THREE.Vector3(
    sourceAgent.position.x,
    sourceAgent.position.y,
    sourceAgent.position.z
  );

  const targetPos = new THREE.Vector3(
    targetAgent.position.x,
    targetAgent.position.y,
    targetAgent.position.z
  );

  
  const points = useMemo(() => {
    const midPoint = new THREE.Vector3()
      .addVectors(sourcePos, targetPos)
      .multiplyScalar(0.5);

    
    const direction = new THREE.Vector3().subVectors(targetPos, sourcePos);
    const perpendicular = new THREE.Vector3(-direction.y, direction.x, 0).normalize();
    midPoint.add(perpendicular.multiplyScalar(2));

    const curve = new THREE.QuadraticBezierCurve3(sourcePos, midPoint, targetPos);
    return curve.getPoints(50);
  }, [sourcePos, targetPos]);

  const geometry = useMemo(() => {
    return new THREE.BufferGeometry().setFromPoints(points);
  }, [points]);

  
  useFrame((state) => {
    if (lineRef.current) {
      const material = lineRef.current.material as THREE.LineBasicMaterial;
      material.opacity = 0.3 + Math.sin(state.clock.elapsedTime * 2) * 0.2;
    }
  });

  
  const lineWidth = connection.weight ? connection.weight * 2 : 2;
  const opacity = connection.type === 'communication' ? 0.5 : 0.3;

  return (
    <line ref={lineRef} geometry={geometry}>
      <lineBasicMaterial
        color={color}
        linewidth={lineWidth}
        transparent
        opacity={opacity}
      />
    </line>
  );
};


export const useAgentNodes = () => {
  const [agents, setAgents] = React.useState<AgentNode[]>([]);
  const [connections, setConnections] = React.useState<AgentConnection[]>([]);
  const { settings } = useSettingsStore();

  useEffect(() => {
    const pollAgents = async () => {
      try {
        const response = await fetch('/api/bots/agents');
        if (response.ok) {
          const data = await response.json();
          setAgents(data.agents || []);
        }
      } catch (error) {
        console.error('Failed to fetch agent telemetry:', error);
      }
    };

    const pollConnections = async () => {
      try {
        const response = await fetch('/api/bots/data');
        if (response.ok) {
          const data = await response.json();
          setConnections(data.edges || []);
        }
      } catch (error) {
        console.error('Failed to fetch agent connections:', error);
      }
    };

    
    const interval = (settings?.agents?.monitoring?.telemetry_poll_interval || 5) * 1000;

    const timer = setInterval(() => {
      pollAgents();
      pollConnections();
    }, interval);

    pollAgents();
    pollConnections();

    return () => clearInterval(timer);
  }, [settings?.agents?.monitoring?.telemetry_poll_interval]);

  return { agents, connections };
};

export default AgentNodesLayer;
