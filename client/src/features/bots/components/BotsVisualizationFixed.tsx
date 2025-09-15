import React, { useRef, useEffect, useState, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { Html, Text, Billboard, Line as DreiLine } from '@react-three/drei';
import { BotsAgent, BotsEdge, BotsState, TokenUsage } from '../types/BotsTypes';
import { createLogger } from '../../../utils/logger';
import { useSettingsStore } from '../../../store/settingsStore';
import { debugState } from '../../../utils/clientDebugState';
import { useBotsData } from '../contexts/BotsDataContext';

const logger = createLogger('BotsVisualization');

// CSS animations for enhanced visualizations
const pulseKeyframes = `
  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.7; transform: scale(0.95); }
  }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(-2px); }
    to { opacity: 1; transform: translateY(0); }
  }
`;

// Inject CSS if it hasn't been added already
if (typeof document !== 'undefined' && !document.querySelector('#bots-visualization-styles')) {
  const style = document.createElement('style');
  style.id = 'bots-visualization-styles';
  style.textContent = pulseKeyframes;
  document.head.appendChild(style);
}

// Helper to format processing logs (no mock generation)
const formatProcessingLogs = (logs: string[] | undefined): string[] => {
  // Return actual logs if provided, otherwise empty array
  return logs || [];
};

// Get VisionFlow colors from settings or use defaults
const getVisionFlowColors = (settings: any) => {
  const visionflowSettings = settings?.visualisation?.graphs?.visionflow;
  const baseColor = visionflowSettings?.nodes?.baseColor || '#F1C40F';
  
  // Get agent colors from server settings (provided via dev_config.toml)
  const agentColors = settings?.visualisation?.rendering?.agentColors;
  
  if (agentColors) {
    // Use server-provided colors
    return {
      // Agent types from server
      coder: agentColors.coder || '#2ECC71',
      tester: agentColors.tester || '#27AE60',
      researcher: agentColors.researcher || '#1ABC9C',
      reviewer: agentColors.reviewer || '#16A085',
      documenter: agentColors.documenter || '#229954',
      specialist: agentColors.default || '#239B56',
      queen: agentColors.queen || '#FFD700',
      coordinator: agentColors.coordinator || baseColor,
      architect: agentColors.architect || '#F1C40F',
      monitor: agentColors.default || '#E67E22',
      analyst: agentColors.analyst || '#D68910',
      optimizer: agentColors.optimizer || '#B7950B',

      // Connections (not in server config, use defaults)
      edge: '#3498DB',        // Bright blue
      activeEdge: '#2980B9',  // Peter river blue

      // States (not in server config, use defaults)
      active: '#2ECC71',
      busy: '#F39C12',
      idle: '#95A5A6',
      error: '#E74C3C'
    };
  }

  // Fallback to hardcoded colors if server doesn't provide them
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

        {agent.memoryUsage && agent.memoryUsage > 0 && (
          <div style={{
            padding: '3px 8px',
            borderRadius: '12px',
            fontSize: '11px',
            backgroundColor: 'rgba(155, 89, 182, 0.8)',
            color: 'white'
          }}>
            MEM: {agent.memoryUsage.toFixed(0)}%
          </div>
        )}

        {agent.successRate !== undefined && (
          <div style={{
            padding: '3px 8px',
            borderRadius: '12px',
            fontSize: '11px',
            backgroundColor: agent.successRate > 0.8 ? '#27AE60' :
                            agent.successRate > 0.6 ? '#F39C12' : '#E74C3C',
            color: 'white'
          }}>
            Success: {(agent.successRate * 100).toFixed(0)}%
          </div>
        )}
      </div>

      {/* Token Usage Row */}
      {(agent.tokens || agent.tokenRate) && (
        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginTop: '2px' }}>
          {agent.tokens && (
            <div style={{
              padding: '2px 6px',
              borderRadius: '10px',
              fontSize: '10px',
              backgroundColor: 'rgba(230, 126, 34, 0.8)',
              color: 'white'
            }}>
              Tokens: {agent.tokens.toLocaleString()}
            </div>
          )}
          {agent.tokenRate && (
            <div style={{
              padding: '2px 6px',
              borderRadius: '10px',
              fontSize: '10px',
              backgroundColor: 'rgba(231, 76, 60, 0.8)',
              color: 'white',
              animation: agent.tokenRate > 10 ? 'pulse 1.5s ease-in-out infinite' : 'none'
            }}>
              Rate: {agent.tokenRate.toFixed(1)}/min
            </div>
          )}
        </div>
      )}

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

      {/* Agent Capabilities */}
      {agent.capabilities && agent.capabilities.length > 0 && (
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '3px',
          marginTop: '4px'
        }}>
          {agent.capabilities.slice(0, 4).map(cap => (
            <span
              key={cap}
              style={{
                fontSize: '9px',
                padding: '1px 4px',
                borderRadius: '3px',
                backgroundColor: 'rgba(0, 123, 255, 0.1)',
                color: '#0056b3',
                border: '1px solid rgba(0, 123, 255, 0.2)'
              }}
            >
              {cap.replace(/_/g, ' ')}
            </span>
          ))}
          {agent.capabilities.length > 4 && (
            <span style={{ fontSize: '9px', color: '#999' }}>
              +{agent.capabilities.length - 4} more
            </span>
          )}
        </div>
      )}

      {/* Agent Mode and Age Info */}
      {(agent.agentMode || agent.age) && (
        <div style={{
          fontSize: '9px',
          color: '#666',
          marginTop: '2px',
          display: 'flex',
          gap: '8px'
        }}>
          {agent.agentMode && (
            <span>Mode: {agent.agentMode}</span>
          )}
          {agent.age && (
            <span>Age: {Math.floor(agent.age / 1000 / 60)}m</span>
          )}
        </div>
      )}

      {/* multi-agent Info */}
      {agent.swarmId && (
        <div style={{
          fontSize: '9px',
          color: '#888',
          marginTop: '2px'
        }}>
          swarm: {agent.swarmId}
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
  const groupRef = useRef<THREE.Group>(null);
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
    if (!groupRef.current || !meshRef.current || !glowRef.current) return;

    // Update group position (this moves everything including labels)
    groupRef.current.position.copy(position);

    // Enhanced pulse animation based on token rate and status
    if (agent.status === 'active' || agent.status === 'busy') {
      // Base pulse speed influenced by token rate
      const tokenMultiplier = agent.tokenRate ? Math.min(agent.tokenRate / 10, 3) : 1;
      const pulseSpeed = 2 * tokenMultiplier;
      const pulse = Math.sin(state.clock.elapsedTime * pulseSpeed + index) * 0.1 + 1;
      
      meshRef.current.scale.setScalar(pulse);
      
      // Glow intensity based on token rate
      const glowIntensity = agent.tokenRate ? Math.min(agent.tokenRate / 20, 2) : 1;
      glowRef.current.scale.setScalar(pulse * 1.5 * glowIntensity);
    }

    // Enhanced rotation for busy agents (faster with higher token rate)
    if (agent.status === 'busy') {
      const rotationSpeed = agent.tokenRate ? 0.01 * (1 + agent.tokenRate / 50) : 0.01;
      meshRef.current.rotation.y += rotationSpeed;
    }

    // Special high-activity animation for high token rate agents
    if (agent.tokenRate && agent.tokenRate > 30) {
      const vibration = Math.sin(state.clock.elapsedTime * 20) * 0.02;
      meshRef.current.position.y += vibration;
    }
  });

  // Use actual logs or empty array
  const processingLogs = formatProcessingLogs(agent.processingLogs);

  return (
    <group ref={groupRef}>
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
          center
          distanceFactor={10}
          style={{
            transition: 'all 0.2s',
            opacity: hover ? 1 : 0.8,
            pointerEvents: 'none',
            position: 'absolute',
            top: `${-size * 20}px`, // Adjust positioning relative to the node
            left: '0',
          }}
        >
          <AgentStatusBadges agent={agent} logs={processingLogs} />
        </Html>
      )}

      {/* 3D Text label */}
      <Billboard
        follow={true}
        lockX={false}
        lockY={false}
        lockZ={false}
      >
        <Text
          position={[0, -size - 0.5, 0]}
          fontSize={0.4}
          color="white"
          anchorX="center"
          anchorY="middle"
          outlineWidth={0.05}
          outlineColor="black"
        >
          {agent.name || String(agent.id).slice(0, 8)}
        </Text>
      </Billboard>
    </group>
  );
};

// Enhanced Edge Component with data flow visualization
interface BotsEdgeProps {
  edge: BotsEdge;
  sourcePos: THREE.Vector3;
  targetPos: THREE.Vector3;
  color: string;
  sourceAgent?: BotsAgent;
  targetAgent?: BotsAgent;
}

const BotsEdgeComponent: React.FC<BotsEdgeProps> = ({ 
  edge, 
  sourcePos, 
  targetPos, 
  color, 
  sourceAgent, 
  targetAgent 
}) => {
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

  // Calculate communication bandwidth based on connected agents' token rates
  const sourceTokenRate = sourceAgent?.tokenRate || 0;
  const targetTokenRate = targetAgent?.tokenRate || 0;
  const avgTokenRate = (sourceTokenRate + targetTokenRate) / 2;
  
  // Enhanced visual properties based on data flow
  const baseWidth = Math.max(1, edge.dataVolume / 1000); // Base width from data volume
  const tokenWidth = avgTokenRate > 0 ? Math.min(avgTokenRate / 10, 3) : 0; // Additional width from token rate
  const lineWidth = isActive ? Math.max(2, baseWidth + tokenWidth) : Math.max(1, baseWidth);
  
  // Opacity and color intensity based on activity
  const baseOpacity = isActive ? 0.8 : 0.3;
  const tokenOpacity = avgTokenRate > 10 ? Math.min(avgTokenRate / 50, 0.4) : 0;
  const opacity = Math.min(baseOpacity + tokenOpacity, 1);
  
  // Color variation based on communication intensity
  const edgeColor = isActive ? 
    (avgTokenRate > 20 ? '#E67E22' : // Orange for high token flow
     avgTokenRate > 10 ? '#3498DB' : // Blue for medium token flow  
     '#2980B9') : // Dark blue for low activity
    color;

  // Animation properties for high-bandwidth connections
  const shouldAnimate = isActive && avgTokenRate > 15;
  const dashOffset = shouldAnimate ? -Date.now() * 0.001 : 0;

  return (
    <>
      {/* Main connection line */}
      <DreiLine
        points={[sourcePos, targetPos]}
        color={edgeColor}
        lineWidth={lineWidth}
        opacity={opacity}
        transparent
        dashed={!isActive || shouldAnimate}
        dashScale={shouldAnimate ? 10 : 5}
        dashSize={shouldAnimate ? 2 : 1}
        dashOffset={dashOffset}
      />
      
      {/* High-bandwidth indicator - additional glowing line */}
      {avgTokenRate > 25 && isActive && (
        <DreiLine
          points={[sourcePos, targetPos]}
          color="#F39C12"
          lineWidth={lineWidth * 0.5}
          opacity={0.4}
          transparent
          dashed={true}
          dashScale={15}
          dashSize={3}
          dashOffset={-dashOffset * 1.5}
        />
      )}
    </>
  );
};

// Main Visualization Component
// Note: This is a pure rendering component that receives positions from server physics simulation
// via binary protocol. No client-side physics computation is performed.
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

  // Refs for server-authoritative positions
  const positionsRef = useRef<Map<string, THREE.Vector3>>(new Map());

  // Note: Binary position updates removed - now handled via full graph updates
  // The server sends complete graph data including positions via requestBotsGraph

  // Colors
  const colors = useMemo(() => getVisionFlowColors(settings), [settings]);

  // Process data from context
  useEffect(() => {
    if (!contextBotsData) {
      logger.debug('[VISIONFLOW] No context data available yet');
      return;
    }

    logger.debug('[VISIONFLOW] Processing bots data from context', contextBotsData);
    setIsLoading(false);

    // Update local state with context data
    const agents = contextBotsData.agents || [];
    const agentMap = new Map<string, BotsAgent>();
    agents.forEach((agent, index) => {
      agentMap.set(agent.id, agent);
      
      // Check if agent has server-provided position data
      if (agent.position && (agent.position.x !== undefined || agent.position.y !== undefined || agent.position.z !== undefined)) {
        // Always update with server position when available
        const serverPosition = new THREE.Vector3(
          agent.position.x || 0,
          agent.position.y || 0,
          agent.position.z || 0
        );
        positionsRef.current.set(agent.id, serverPosition);
      } else if (!positionsRef.current.has(agent.id)) {
        // Only set initial calculated position if no server position and no existing position
        const radius = 25;
        const angle = (index / agents.length) * Math.PI * 2;
        const height = (Math.random() - 0.5) * 15;
        const newPosition = new THREE.Vector3(
          Math.cos(angle) * radius,
          height,
          Math.sin(angle) * radius
        );
        positionsRef.current.set(agent.id, newPosition);
      }
    });

    // Use edges from context (provided by backend with full graph data)
    const edges = (contextBotsData as any).edges || [];
    const edgeMap = new Map<string, BotsEdge>();
    edges.forEach((edge: BotsEdge) => {
      edgeMap.set(edge.id, edge);
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

  // Request server position updates periodically
  useFrame(() => {
    // The server handles all physics computation
    // We just render the positions received via binary protocol
    // No client-side physics simulation needed
  });

  // Position updates are now handled automatically via WebSocket polling
  // The BotsWebSocketIntegration service polls for graph updates every 2 seconds

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
    // Don't render anything in the 3D scene when no agents are active
    // This message should be shown in the control panel instead
    return null;
  }

  return (
    <group>
      {/* Render edges */}
      {Array.from(botsData.edges.values()).map(edge => {
        const sourcePos = positionsRef.current.get(edge.source);
        const targetPos = positionsRef.current.get(edge.target);

        if (!sourcePos || !targetPos) return null;

        // Get source and target agents for enhanced edge visualization
        const sourceAgent = botsData.agents.get(edge.source);
        const targetAgent = botsData.agents.get(edge.target);

        return (
          <BotsEdgeComponent
            key={edge.id}
            edge={edge}
            sourcePos={sourcePos}
            targetPos={targetPos}
            color={colors.edge}
            sourceAgent={sourceAgent}
            targetAgent={targetAgent}
          />
        );
      })}

      {/* Render nodes using server-authoritative positions */}
      {Array.from(botsData.agents.values()).map((node, index) => {
        const position = positionsRef.current.get(node.id);
        if (!position) return null; // Wait for server position data

        const nodeColor = colors[node.type] || colors.coordinator;

        return (
          <BotsNode
            key={node.id}
            agent={node}
            position={position} // Server-computed position via binary protocol
            index={index}
            color={nodeColor}
          />
        );
      })}

      {/* Debug info removed - now in control panel */}
    </group>
  );
};