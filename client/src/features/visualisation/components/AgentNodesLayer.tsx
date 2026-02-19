import React, { useEffect, useRef, useMemo } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { useSettingsStore } from '@/store/settingsStore';
import { Text, Html } from '@react-three/drei';
import { isWebGPURenderer } from '../../../rendering/rendererFactory';



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

// Health-based glow color (aligned with BotsVisualization palette)
const getGlowColor = (health: number): string => {
  if (health >= 95) return '#00FF00';
  if (health >= 80) return '#2ECC71';
  if (health >= 65) return '#F1C40F';
  if (health >= 50) return '#F39C12';
  if (health >= 25) return '#E67E22';
  return '#E74C3C';
};

export const AgentNodesLayer: React.FC<AgentNodesLayerProps> = ({
  agents,
  connections = []
}) => {
  const { settings } = useSettingsStore();
  const groupRef = useRef<THREE.Group>(null);

  // Type assertion for extended settings that may include agents
  const agentViz = (settings as any)?.agents?.visualization;
  const showAgents = agentViz?.show_in_graph ?? true;
  const nodeSize = agentViz?.node_size ?? 1.5;
  const baseColor = agentViz?.node_color ?? '#ff8800';
  const showConnections = agentViz?.show_connections ?? true;
  const connectionColor = agentViz?.connection_color ?? '#fbbf24';
  const animateActivity = agentViz?.animate_activity ?? true;

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
  const nucleusRef = useRef<THREE.Mesh>(null);
  const pulseRef = useRef({ phase: 0 });

  const position: [number, number, number] = useMemo(() => {
    if (agent.position && (agent.position.x !== 0 || agent.position.y !== 0 || agent.position.z !== 0)) {
      return [agent.position.x, agent.position.y, agent.position.z];
    }
    // Deterministic fallback position from agent ID hash
    let hash = 0;
    for (let i = 0; i < agent.id.length; i++) {
      hash = ((hash << 5) - hash) + agent.id.charCodeAt(i);
      hash |= 0;
    }
    const pseudoRandom = (seed: number) => {
      const x = Math.sin(seed) * 10000;
      return x - Math.floor(x);
    };
    return [
      pseudoRandom(hash) * 20 - 10,
      pseudoRandom(hash + 1) * 20 - 10,
      pseudoRandom(hash + 2) * 20 - 10
    ];
  }, [agent.id, agent.position?.x, agent.position?.y, agent.position?.z]);

  const statusColor = STATUS_COLORS[agent.status] || baseColor;
  const glowColor = useMemo(() => getGlowColor(agent.health), [agent.health]);

  const scaledSize = nodeSize * (1 + agent.workload / 100);

  useFrame((state, delta) => {
    if (!meshRef.current || !glowRef.current) return;

    if (animateActivity && (agent.status === 'active' || agent.status === 'warning')) {
      // Organic breathing: asymmetric inhale/exhale
      pulseRef.current.phase += delta * 2;
      const breathCycle = Math.sin(pulseRef.current.phase);
      const breathScale = breathCycle > 0
        ? 1 + breathCycle * 0.08
        : 1 + breathCycle * 0.04;

      meshRef.current.scale.setScalar(breathScale);

      // Membrane breathes with slight delay
      const membraneBreath = 1.3 + Math.sin(pulseRef.current.phase - 0.3) * 0.06;
      glowRef.current.scale.setScalar(membraneBreath);

      // Gentle rotation
      meshRef.current.rotation.y += delta * 0.5;

      // Nucleus glow pulse
      if (nucleusRef.current) {
        const nucleusMat = nucleusRef.current.material as THREE.MeshBasicMaterial;
        if (nucleusMat) {
          const nucleusPulse = Math.pow(Math.sin(pulseRef.current.phase * 0.6 + 0.5) * 0.5 + 0.5, 2);
          nucleusMat.opacity = 0.2 + nucleusPulse * 0.3;
        }
      }
    } else if (agent.status === 'error') {
      // Distress flicker
      pulseRef.current.phase += delta * 8;
      const distress = Math.sin(pulseRef.current.phase) * Math.sin(pulseRef.current.phase * 0.66) * 0.15;
      meshRef.current.scale.setScalar(1 + Math.abs(distress));
      glowRef.current.scale.setScalar(1.3 + Math.abs(distress) * 0.5);
    } else {
      // Idle: very subtle life sign
      if (nucleusRef.current) {
        pulseRef.current.phase += delta * 0.5;
        const idleMat = nucleusRef.current.material as THREE.MeshBasicMaterial;
        if (idleMat) {
          idleMat.opacity = 0.1 + Math.sin(pulseRef.current.phase) * 0.05;
        }
      }
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

  useEffect(() => {
    return () => { geometry?.dispose(); };
  }, [geometry]);

  return (
    <group position={position}>
      {/* Outer membrane (bioluminescent) */}
      <mesh ref={glowRef} scale={[1.3, 1.3, 1.3]}>
        <sphereGeometry args={[scaledSize * 0.75, 24, 24]} />
        <meshStandardMaterial
          color={glowColor}
          transparent
          opacity={0.08}
          side={THREE.BackSide}
          depthWrite={false}
          emissive={glowColor}
          emissiveIntensity={0.3}
        />
      </mesh>

      {/* Inner nucleus glow */}
      <mesh ref={nucleusRef} scale={[0.4, 0.4, 0.4]}>
        <sphereGeometry args={[scaledSize * 0.8, 12, 12]} />
        <meshBasicMaterial
          color={statusColor}
          transparent
          opacity={0.25}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </mesh>

      {/* Main body */}
      <mesh ref={meshRef} geometry={geometry}>
        <meshStandardMaterial
          color={statusColor}
          emissive={glowColor}
          emissiveIntensity={agent.status === 'active' ? 0.5 : 0.2}
          metalness={0.3}
          roughness={0.7}
        />
      </mesh>

      {/* Agent type label — Html on WebGPU (troika Text triggers drawIndexed(Infinity)) */}
      {isWebGPURenderer ? (
        <Html position={[0, scaledSize + 1.5, 0]} center style={{ pointerEvents: 'none', whiteSpace: 'nowrap' }}>
          <div style={{ color: statusColor, fontSize: '12px', fontWeight: 'bold', textShadow: '0 0 4px black' }}>
            {agent.type.toUpperCase()}
          </div>
          <div style={{ color: '#fff', fontSize: '10px', textShadow: '0 0 3px black' }}>
            {agent.status} | {agent.health}%
          </div>
          {agent.currentTask && (
            <div style={{ color: '#aaa', fontSize: '9px', maxWidth: '120px', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {agent.currentTask}
            </div>
          )}
        </Html>
      ) : (
        <>
          <Text
            position={[0, scaledSize + 1, 0]}
            fontSize={0.5}
            color={statusColor}
            anchorX="center"
            anchorY="bottom"
            outlineWidth={0.03}
            outlineColor="black"
          >
            {agent.type.toUpperCase()}
          </Text>
          <Text
            position={[0, scaledSize + 1.5, 0]}
            fontSize={0.3}
            color="#ffffff"
            anchorX="center"
            anchorY="bottom"
            outlineWidth={0.02}
            outlineColor="black"
          >
            {agent.status} | {agent.health}%
          </Text>
          {agent.currentTask && (
            <Text
              position={[0, -(scaledSize + 1), 0]}
              fontSize={0.25}
              color="#aaaaaa"
              anchorX="center"
              anchorY="top"
              maxWidth={10}
              outlineWidth={0.01}
              outlineColor="black"
            >
              {agent.currentTask}
            </Text>
          )}
        </>
      )}

      {/* Health bar with gradient glow */}
      <group position={[0, -(scaledSize + 0.5), 0]}>
        {/* Background track */}
        <mesh position={[0, 0, 0]}>
          <planeGeometry args={[2, 0.15]} />
          <meshBasicMaterial color="#1a1a1a" transparent opacity={0.6} />
        </mesh>
        {/* Health fill with bioluminescent color */}
        <mesh position={[-(1 - agent.health / 100), 0, 0.01]}>
          <planeGeometry args={[(agent.health / 100) * 2, 0.15]} />
          <meshBasicMaterial
            color={glowColor}
            transparent
            opacity={0.9}
          />
        </mesh>
        {/* Glow overlay on health bar */}
        <mesh position={[-(1 - agent.health / 100), 0, 0.02]}>
          <planeGeometry args={[(agent.health / 100) * 2, 0.25]} />
          <meshBasicMaterial
            color={glowColor}
            transparent
            opacity={0.15}
            blending={THREE.AdditiveBlending}
            depthWrite={false}
          />
        </mesh>
      </group>

      {/* Workload ring */}
      {agent.status === 'active' && agent.workload > 0 && (
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[scaledSize * 1.8, 0.05, 8, 32, (agent.workload / 100) * Math.PI * 2]} />
          <meshBasicMaterial
            color={glowColor}
            transparent
            opacity={0.6}
          />
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

  // Safe: early return above guarantees position is defined
  const sourcePos = useMemo(() => new THREE.Vector3(
    sourceAgent.position!.x, sourceAgent.position!.y, sourceAgent.position!.z
  ), [sourceAgent.position?.x, sourceAgent.position?.y, sourceAgent.position?.z]);

  const targetPos = useMemo(() => new THREE.Vector3(
    targetAgent.position!.x, targetAgent.position!.y, targetAgent.position!.z
  ), [targetAgent.position?.x, targetAgent.position?.y, targetAgent.position?.z]);

  
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

  useEffect(() => {
    return () => { geometry?.dispose(); };
  }, [geometry]);


  useFrame((state) => {
    if (lineRef.current) {
      const material = lineRef.current.material as THREE.LineBasicMaterial;
      material.opacity = 0.3 + Math.sin(state.clock.elapsedTime * 2) * 0.2;
    }
  });


  const lineWidth = connection.weight ? connection.weight * 2 : 2;
  const opacity = connection.type === 'communication' ? 0.5 : 0.3;

  const lineMaterial = useMemo(() => new THREE.LineBasicMaterial({
    color,
    linewidth: lineWidth,
    transparent: true,
    opacity
  }), [color, lineWidth, opacity]);

  const lineObject = useMemo(() => {
    if (!geometry) return null;
    return new THREE.Line(geometry, lineMaterial);
  }, [geometry, lineMaterial]);

  useEffect(() => {
    return () => {
      lineMaterial?.dispose();
    };
  }, [lineMaterial]);

  return (
    <>
      {lineObject && <primitive object={lineObject} ref={lineRef} />}
    </>
  );
};


export const useAgentNodes = () => {
  const [agents, setAgents] = React.useState<AgentNode[]>([]);
  const [connections, setConnections] = React.useState<AgentConnection[]>([]);
  const { settings } = useSettingsStore();

  // Type assertion for extended settings with agents
  const agentMonitoring = (settings as any)?.agents?.monitoring;
  const pollInterval = agentMonitoring?.telemetry_poll_interval || 5;

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

    // Poll at the configured interval
    const interval = pollInterval * 1000;

    const timer = setInterval(() => {
      pollAgents();
      pollConnections();
    }, interval);

    pollAgents();
    pollConnections();

    return () => clearInterval(timer);
  }, [pollInterval]);

  return { agents, connections };
};

export default AgentNodesLayer;
