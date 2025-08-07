import React, { useRef, useEffect, useState, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { Html, Text, Billboard, Line as DreiLine } from '@react-three/drei';
import { BotsAgent, BotsEdge, BotsState, TokenUsage } from '../types/BotsTypes';
import { createLogger } from '../../../utils/logger';
import { useSettingsStore } from '../../../store/settingsStore';
import { useBotsBinaryUpdates } from '../hooks/useBotsBinaryUpdates';
import { BotsDebugInfo } from './BotsVisualizationDebugInfo';
import { debugState } from '../../../utils/debugState';
import { useBotsData } from '../contexts/BotsDataContext';

const logger = createLogger('BotsVisualization');

// Generate mock processing logs for visualization
const generateMockProcessingLogs = (agentType: string, status: string): string[] => {
  const logs: string[] = [];

  const activityTemplates = {
    coordinator: [
      'Analyzing swarm topology and agent distribution patterns...',
      'Coordinating task allocation across 7 active agents...',
      'Optimizing communication channels for minimal latency...',
      'Monitoring agent health metrics and workload balance...',
      'Synchronizing distributed memory across swarm nodes...'
    ],
    researcher: [
      'Scanning knowledge base for relevant documentation...',
      'Analyzing code patterns in /src/components/*.tsx files...',
      'Extracting semantic relationships from API endpoints...',
      'Cross-referencing implementation with best practices...',
      'Compiling research findings into actionable insights...'
    ],
    coder: [
      'Implementing authentication middleware with JWT tokens...',
      'Refactoring database connection pool for performance...',
      'Writing unit tests for UserService.createUser() method...',
      'Optimizing React component render cycles in Dashboard...',
      'Debugging WebSocket connection timeout issues...'
    ],
    analyst: [
      'Profiling application performance bottlenecks...',
      'Analyzing database query execution plans...',
      'Evaluating code complexity metrics across modules...',
      'Identifying potential security vulnerabilities...',
      'Generating performance optimization recommendations...'
    ],
    architect: [
      'Designing microservice communication patterns...',
      'Evaluating architectural trade-offs for scalability...',
      'Creating domain model for user management system...',
      'Planning database schema migrations strategy...',
      'Documenting API contract specifications...'
    ],
    tester: [
      'Executing integration test suite for API endpoints...',
      'Running load tests with 1000 concurrent users...',
      'Validating edge cases in payment processing flow...',
      'Checking accessibility compliance for UI components...',
      'Generating code coverage reports for modules...'
    ],
    queen: [
      'Orchestrating hive mind collective intelligence...',
      'Distributing tasks across specialized worker agents...',
      'Optimizing swarm topology for maximum efficiency...',
      'Monitoring collective progress towards objectives...',
      'Synchronizing knowledge base across all agents...'
    ],
    default: [
      'Processing task queue items sequentially...',
      'Analyzing incoming data streams for patterns...',
      'Executing scheduled maintenance operations...',
      'Monitoring system resources and performance...',
      'Updating internal state and synchronizing...'
    ]
  };

  const templates = activityTemplates[agentType] || activityTemplates.default;

  // Generate 3 random logs
  for (let i = 0; i < 3; i++) {
    const template = templates[Math.floor(Math.random() * templates.length)];
    logs.push(template);
  }

  return logs;
};

// Get VisionFlow colors from settings or use defaults
const getVisionFlowColors = (settings: any) => {
  const visionflowSettings = settings?.visualisation?.graphs?.visionflow;
  const baseColor = visionflowSettings?.nodes?.baseColor || '#F1C40F';

  // Default gold and green color palette for bots
  return {
    // Primary agent types - Greens for roles
    coder: '#2ECC71',       // Emerald green
    tester: '#27AE60',      // Nephritis green
    researcher: '#1ABC9C',  // Turquoise
    reviewer: '#16A085',    // Green sea
    documenter: '#229954',  // Forest green
    specialist: '#239B56',  // Emerald dark

    // Meta roles - Golds for coordination
    queen: '#F39C12',       // Orange gold (leader)
    coordinator: baseColor,  // Primary gold
    architect: '#F1C40F',   // Sunflower gold
    monitor: '#E67E22',     // Carrot orange
    analyst: '#D68910',     // Dark gold
    optimizer: '#B7950B',   // Dark gold variant

    // Connections
    edge: '#3498DB',        // Bright blue
    activeEdge: '#2980B9',  // Peter river blue

    // States
    active: '#2ECC71',
    busy: '#F39C12',
    idle: '#95A5A6',
    error: '#E74C3C'
  };
};

// Agent Status Badges Component
interface AgentStatusBadgesProps {
  agent: BotsAgent;
  logs?: string[];
}

const AgentStatusBadges: React.FC<AgentStatusBadgesProps> = ({ agent, logs = [] }) => {
  const [logKey, setLogKey] = useState(0);
  const [displayLogs, setDisplayLogs] = useState<{ text: string; key: number }[]>([]);

  useEffect(() => {
    // Keep only the last 3 logs with unique keys for animation
    const newLogs = logs.slice(-3).map((log, index) => ({
      text: log,
      key: logKey + index
    }));
    setDisplayLogs(newLogs);
    setLogKey(prev => prev + logs.length);
  }, [logs]);

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      gap: '4px',
      minWidth: '250px',
      maxWidth: '350px',
    }}>
      {/* Agent Name and Type */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        marginBottom: '4px'
      }}>
        <span style={{
          fontWeight: 'bold',
          fontSize: '14px',
          color: '#1A1A1A'
        }}>
          {agent.name || agent.id}
        </span>
        <span style={{
          fontSize: '11px',
          padding: '2px 6px',
          borderRadius: '3px',
          backgroundColor: 'rgba(0, 0, 0, 0.1)',
          color: '#333'
        }}>
          {agent.type}
        </span>
      </div>

      {/* Status and Health Row */}
      <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
        <div style={{
          padding: '3px 8px',
          borderRadius: '12px',
          fontSize: '11px',
          backgroundColor: agent.status === 'active' ? '#2ECC71' :
                          agent.status === 'busy' ? '#F39C12' :
                          agent.status === 'idle' ? '#95A5A6' : '#E74C3C',
          color: 'white',
          fontWeight: '500'
        }}>
          {agent.status}
        </div>

        <div style={{
          padding: '3px 8px',
          borderRadius: '12px',
          fontSize: '11px',
          backgroundColor: agent.health > 80 ? '#27AE60' :
                          agent.health > 50 ? '#F39C12' : '#E74C3C',
          color: 'white'
        }}>
          Health: {agent.health.toFixed(0)}%
        </div>

        {agent.cpuUsage > 0 && (
          <div style={{
            padding: '3px 8px',
            borderRadius: '12px',
            fontSize: '11px',
            backgroundColor: 'rgba(52, 152, 219, 0.8)',
            color: 'white'
          }}>
            CPU: {agent.cpuUsage.toFixed(0)}%
          </div>
        )}
      </div>

      {/* Task Info */}
      {(agent.tasksActive > 0 || agent.tasksCompleted > 0) && (
        <div style={{
          fontSize: '10px',
          color: '#666',
          marginTop: '2px'
        }}>
          Tasks: {agent.tasksActive} active, {agent.tasksCompleted} completed
        </div>
      )}

      {/* Current Task or Activity */}
      {(agent.currentTask || displayLogs.length > 0) && (
        <div style={{
          marginTop: '4px',
          fontSize: '10px',
          color: '#444',
          lineHeight: '1.3',
          maxHeight: '60px',
          overflow: 'hidden'
        }}>
          {agent.currentTask ? (
            <div style={{ fontStyle: 'italic' }}>{agent.currentTask}</div>
          ) : (
            displayLogs.map((log, index) => (
              <div
                key={log.key}
                style={{
                  opacity: 1 - (index * 0.3),
                  animation: 'fadeIn 0.5s ease-in',
                  marginBottom: '2px'
                }}
              >
                • {log.text}
              </div>
            ))
          )}
        </div>
      )}

      {/* Swarm Info */}
      {agent.swarmId && (
        <div style={{
          fontSize: '9px',
          color: '#888',
          marginTop: '2px'
        }}>
          Swarm: {agent.swarmId}
          {agent.parentQueenId && ` • Queen: ${agent.parentQueenId.slice(0, 8)}...`}
        </div>
      )}
    </div>
  );
};

// Node Component
interface BotsNodeProps {
  agent: BotsAgent;
  position: THREE.Vector3;
  index: number;
  color: string;
}

const BotsNode: React.FC<BotsNodeProps> = ({ agent, position, index, color }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const glowRef = useRef<THREE.Mesh>(null);
  const [hover, setHover] = useState(false);

  // Health-based glow color
  const glowColor = useMemo(() => {
    if (agent.health > 80) return '#2ECC71';
    if (agent.health > 50) return '#F39C12';
    return '#E74C3C';
  }, [agent.health]);

  // Size based on workload
  const size = 1 + (agent.cpuUsage / 100) * 0.5;

  // Shape based on status
  const geometry = useMemo(() => {
    switch (agent.status) {
      case 'error':
      case 'terminating':
        return new THREE.TetrahedronGeometry(size);
      case 'initializing':
        return new THREE.BoxGeometry(size, size, size);
      case 'idle':
        return new THREE.SphereGeometry(size * 0.8, 16, 16);
      case 'busy':
      default:
        return new THREE.SphereGeometry(size, 32, 32);
    }
  }, [agent.status, size]);

  useFrame((state) => {
    if (!meshRef.current || !glowRef.current) return;

    // Update position
    meshRef.current.position.copy(position);
    glowRef.current.position.copy(position);

    // Pulse animation for active agents
    if (agent.status === 'active' || agent.status === 'busy') {
      const pulse = Math.sin(state.clock.elapsedTime * 2 + index) * 0.1 + 1;
      meshRef.current.scale.setScalar(pulse);
      glowRef.current.scale.setScalar(pulse * 1.5);
    }

    // Rotation for busy agents
    if (agent.status === 'busy') {
      meshRef.current.rotation.y += 0.01;
    }
  });

  // Generate mock logs if not provided
  const mockLogs = agent.processingLogs || generateMockProcessingLogs(agent.type, agent.status);

  return (
    <group>
      {/* Glow effect */}
      <mesh ref={glowRef}>
        <sphereGeometry args={[size * 1.5, 16, 16]} />
        <meshBasicMaterial
          color={glowColor}
          transparent
          opacity={0.15 + (hover ? 0.1 : 0)}
          depthWrite={false}
        />
      </mesh>

      {/* Main node */}
      <mesh
        ref={meshRef}
        geometry={geometry}
        onPointerOver={() => setHover(true)}
        onPointerOut={() => setHover(false)}
      >
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.3}
          metalness={0.3}
          roughness={0.7}
        />
      </mesh>

      {/* Agent info on hover or when active */}
      {(hover || agent.status === 'active' || agent.status === 'busy') && (
        <Html
          position={[0, size + 1, 0]}
          center
          distanceFactor={10}
          style={{
            transition: 'all 0.2s',
            opacity: hover ? 1 : 0.8,
            pointerEvents: 'none'
          }}
        >
          <AgentStatusBadges agent={agent} logs={mockLogs} />
        </Html>
      )}

      {/* 3D Text label */}
      <Text
        position={[0, -size - 0.5, 0]}
        fontSize={0.4}
        color="white"
        anchorX="center"
        anchorY="middle"
        outlineWidth={0.05}
        outlineColor="black"
      >
        {agent.name || agent.id.slice(0, 8)}
      </Text>
    </group>
  );
};

// Edge Component
interface BotsEdgeProps {
  edge: BotsEdge;
  sourcePos: THREE.Vector3;
  targetPos: THREE.Vector3;
  color: string;
}

const BotsEdgeComponent: React.FC<BotsEdgeProps> = ({ edge, sourcePos, targetPos, color }) => {
  const [isActive, setIsActive] = useState(false);

  useEffect(() => {
    const checkActivity = () => {
      const timeSinceLastMessage = Date.now() - edge.lastMessageTime;
      setIsActive(timeSinceLastMessage < 5000); // Active if communicated within 5 seconds
    };

    checkActivity();
    const interval = setInterval(checkActivity, 1000);
    return () => clearInterval(interval);
  }, [edge.lastMessageTime]);

  const lineWidth = isActive ? 2 : 1;
  const opacity = isActive ? 0.8 : 0.3;

  return (
    <DreiLine
      points={[sourcePos, targetPos]}
      color={isActive ? '#2980B9' : color}
      lineWidth={lineWidth}
      opacity={opacity}
      transparent
      dashed={!isActive}
      dashScale={5}
      dashSize={1}
      dashOffset={0}
    />
  );
};

// Main Visualization Component
export const BotsVisualization: React.FC = () => {
  const settings = useSettingsStore(state => state.settings);
  const { botsData: contextBotsData } = useBotsData();

  // Component state
  const [botsData, setBotsData] = useState<BotsState>({
    agents: new Map(),
    edges: new Map(),
    communications: [],
    tokenUsage: { total: 0, byAgent: {} },
    lastUpdate: 0,
  });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [mcpConnected, setMcpConnected] = useState(false);

  // Refs for physics simulation
  const positionsRef = useRef<Map<string, THREE.Vector3>>(new Map());
  const velocitiesRef = useRef<Map<string, THREE.Vector3>>(new Map());

  // Binary updates for GPU physics
  const { positions: positionBufferRef, requestUpdate } = useBotsBinaryUpdates({
    enabled: true,
  });

  // Colors
  const colors = useMemo(() => getVisionFlowColors(settings), [settings]);

  // Process data from context
  useEffect(() => {
    if (!contextBotsData) {
      logger.debug('[VISIONFLOW] No context data available yet');
      return;
    }

    logger.info('[VISIONFLOW] Processing bots data from context', contextBotsData);
    setIsLoading(false);

    // Update local state with context data
    const agents = contextBotsData.agents || [];
    const agentMap = new Map<string, BotsAgent>();
    agents.forEach((agent, index) => {
      agentMap.set(agent.id, agent);
      // Initialize position if not exists
      if (!positionsRef.current.has(agent.id)) {
        const radius = 25;
        const angle = (index / agents.length) * Math.PI * 2;
        const height = (Math.random() - 0.5) * 15;
        const newPosition = new THREE.Vector3(
          Math.cos(angle) * radius,
          height,
          Math.sin(angle) * radius
        );
        positionsRef.current.set(agent.id, newPosition);
        velocitiesRef.current.set(agent.id, new THREE.Vector3());
      }
    });

    // Generate edges based on parent relationships
    const edgeMap = new Map<string, BotsEdge>();
    agents.forEach(agent => {
      if (agent.parentQueenId) {
        const edgeId = `${agent.parentQueenId}-${agent.id}`;
        edgeMap.set(edgeId, {
          id: edgeId,
          source: agent.parentQueenId,
          target: agent.id,
          dataVolume: 0,
          messageCount: 0,
          lastMessageTime: Date.now()
        });
      }
    });

    setBotsData({
      agents: agentMap,
      edges: edgeMap,
      communications: [],
      tokenUsage: (contextBotsData as any).tokenUsage || { total: 0, byAgent: {} },
      lastUpdate: Date.now(),
    });

    setMcpConnected(agentMap.size > 0);
  }, [contextBotsData]);

  // Simple physics simulation
  useFrame((state, delta) => {
    if (botsData.agents.size === 0 || !positionBufferRef) return;

    const dt = Math.min(delta, 0.016);
    const physics = {
      damping: 0.95,
      nodeRepulsion: 15,
      linkDistance: 20,
      centerGravity: 0.01
    };

    // Update velocities and positions
    botsData.agents.forEach(node => {
      const pos = positionsRef.current.get(node.id);
      const vel = velocitiesRef.current.get(node.id);

      if (!pos || !vel) return;

      // Apply forces
      const force = new THREE.Vector3();

      // Center gravity
      force.add(pos.clone().multiplyScalar(-physics.centerGravity));

      // Node repulsion
      botsData.agents.forEach(otherNode => {
        if (otherNode.id === node.id) return;
        const otherPos = positionsRef.current.get(otherNode.id);
        if (!otherPos) return;

        const diff = pos.clone().sub(otherPos);
        const distance = diff.length();
        if (distance > 0 && distance < physics.nodeRepulsion * 2) {
          const repulsion = diff.normalize().multiplyScalar(
            physics.nodeRepulsion / (distance * distance)
          );
          force.add(repulsion);
        }
      });

      // Edge attraction
      botsData.edges.forEach(edge => {
        let otherNodeId: string | null = null;
        if (edge.source === node.id) otherNodeId = edge.target;
        if (edge.target === node.id) otherNodeId = edge.source;

        if (otherNodeId) {
          const otherPos = positionsRef.current.get(otherNodeId);
          if (otherPos) {
            const diff = otherPos.clone().sub(pos);
            const distance = diff.length();
            if (distance > physics.linkDistance) {
              const attraction = diff.normalize().multiplyScalar(
                (distance - physics.linkDistance) * 0.1
              );
              force.add(attraction);
            }
          }
        }
      });

      // Update velocity and position
      vel.add(force.multiplyScalar(dt));
      vel.multiplyScalar(physics.damping);
      pos.add(vel.clone().multiplyScalar(dt));
    });
  });

  if (error) {
    return (
      <Html center>
        <div style={{ color: '#E74C3C', padding: '20px', textAlign: 'center' }}>
          <h3>VisionFlow Error</h3>
          <p>{error}</p>
        </div>
      </Html>
    );
  }

  if (isLoading) {
    return (
      <Html center>
        <div style={{ color: '#F1C40F', padding: '20px', textAlign: 'center' }}>
          <h3>Loading VisionFlow...</h3>
          <p>Initializing hive mind visualization</p>
        </div>
      </Html>
    );
  }

  if (botsData.agents.size === 0) {
    return (
      <Html center>
        <div style={{ color: '#95A5A6', padding: '20px', textAlign: 'center' }}>
          <h3>No Agents Active</h3>
          <p>Spawn a hive mind to begin</p>
          {!mcpConnected && (
            <p style={{ fontSize: '12px', marginTop: '10px' }}>
              MCP not connected
            </p>
          )}
        </div>
      </Html>
    );
  }

  return (
    <group>
      {/* Render edges */}
      {Array.from(botsData.edges.values()).map(edge => {
        const sourcePos = positionsRef.current.get(edge.source);
        const targetPos = positionsRef.current.get(edge.target);

        if (!sourcePos || !targetPos) return null;

        return (
          <BotsEdgeComponent
            key={edge.id}
            edge={edge}
            sourcePos={sourcePos}
            targetPos={targetPos}
            color={colors.edge}
          />
        );
      })}

      {/* Render nodes */}
      {Array.from(botsData.agents.values()).map((node, index) => {
        const position = positionsRef.current.get(node.id);
        if (!position) return null;

        const nodeColor = colors[node.type] || colors.coordinator;

        return (
          <BotsNode
            key={node.id}
            agent={node}
            position={position}
            index={index}
            color={nodeColor}
          />
        );
      })}

      {/* Debug info */}
      {debugState.isEnabled() && (
        <BotsDebugInfo
          nodeCount={botsData.agents.size}
          edgeCount={botsData.edges.size}
          mcpConnected={mcpConnected}
          dataSource="mcp"
          isLoading={isLoading}
          error={error}
        />
      )}
    </group>
  );
};