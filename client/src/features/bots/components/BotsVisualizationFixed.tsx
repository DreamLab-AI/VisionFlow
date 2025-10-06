import React, { useRef, useEffect, useState, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { Html, Text, Billboard, Line as DreiLine } from '@react-three/drei';
import { BotsAgent, BotsEdge, BotsState, TokenUsage } from '../types/BotsTypes';
import { createLogger } from '../../../utils/loggerConfig';
import { useTelemetry, useThreeJSTelemetry } from '../../../telemetry/useTelemetry';
import { agentTelemetry } from '../../../telemetry/AgentTelemetry';
import { useSettingsStore } from '../../../store/settingsStore';
import { debugState } from '../../../utils/clientDebugState';
import { useBotsData } from '../contexts/BotsDataContext';
import { AgentPollingStatus } from './AgentPollingStatus';

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

// Smooth position interpolation for real-time updates
function lerpVector3(current: THREE.Vector3, target: THREE.Vector3, alpha: number): void {
  current.x += (target.x - current.x) * alpha;
  current.y += (target.y - current.y) * alpha;
  current.z += (target.z - current.z) * alpha;
}

// Dynamic color generation based on agent type hash
const generateAgentTypeColor = (agentType: string): string => {
  // Create a simple hash function for consistent colors per agent type
  let hash = 0;
  for (let i = 0; i < agentType.length; i++) {
    const char = agentType.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32-bit integer
  }

  // Use predefined color palettes based on agent category
  const coordColors = ['#F1C40F', '#F39C12', '#E67E22', '#D68910', '#B7950B']; // Golds/oranges for coordination
  const devColors = ['#2ECC71', '#27AE60', '#1ABC9C', '#16A085', '#229954']; // Greens for development
  const specialColors = ['#9B59B6', '#8E44AD', '#E74C3C', '#C0392B', '#3498DB']; // Various for special roles

  // Categorize agent types
  const coordTypes = ['queen', 'coordinator', 'architect', 'monitor', 'manager'];
  const devTypes = ['coder', 'tester', 'reviewer', 'documenter', 'developer'];

  let colorPalette = specialColors; // Default
  if (coordTypes.some(type => agentType.toLowerCase().includes(type))) {
    colorPalette = coordColors;
  } else if (devTypes.some(type => agentType.toLowerCase().includes(type))) {
    colorPalette = devColors;
  }

  // Select color based on hash
  const colorIndex = Math.abs(hash) % colorPalette.length;
  return colorPalette[colorIndex];
};

// Get VisionFlow colors from settings with dynamic fallback
const getVisionFlowColors = (settings: any) => {
  const visionflowSettings = settings?.visualisation?.graphs?.visionflow;
  const baseColor = visionflowSettings?.nodes?.baseColor || '#F1C40F';

  // Get agent colors from server settings (provided via dev_config.toml)
  const agentColors = settings?.visualisation?.rendering?.agentColors;

  if (agentColors && Object.keys(agentColors).length > 0) {
    // Use server-provided colors with dynamic fallback for missing types
    return {
      // Create a function to get colors dynamically
      getAgentColor: (agentType: string) => {
        return agentColors[agentType] || generateAgentTypeColor(agentType);
      },

      // Predefined server colors (if available)
      coder: agentColors.coder || generateAgentTypeColor('coder'),
      tester: agentColors.tester || generateAgentTypeColor('tester'),
      researcher: agentColors.researcher || generateAgentTypeColor('researcher'),
      reviewer: agentColors.reviewer || generateAgentTypeColor('reviewer'),
      documenter: agentColors.documenter || generateAgentTypeColor('documenter'),
      specialist: agentColors.default || generateAgentTypeColor('specialist'),
      queen: agentColors.queen || generateAgentTypeColor('queen'),
      coordinator: agentColors.coordinator || baseColor,
      architect: agentColors.architect || generateAgentTypeColor('architect'),
      monitor: agentColors.default || generateAgentTypeColor('monitor'),
      analyst: agentColors.analyst || generateAgentTypeColor('analyst'),
      optimizer: agentColors.optimizer || generateAgentTypeColor('optimizer'),

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

  // Dynamic color generation when no server colors available
  return {
    // Create a function to get colors dynamically
    getAgentColor: (agentType: string) => generateAgentTypeColor(agentType),

    // Generate colors for common agent types
    coder: generateAgentTypeColor('coder'),
    tester: generateAgentTypeColor('tester'),
    researcher: generateAgentTypeColor('researcher'),
    reviewer: generateAgentTypeColor('reviewer'),
    documenter: generateAgentTypeColor('documenter'),
    specialist: generateAgentTypeColor('specialist'),
    queen: generateAgentTypeColor('queen'),
    coordinator: baseColor,
    architect: generateAgentTypeColor('architect'),
    monitor: generateAgentTypeColor('monitor'),
    analyst: generateAgentTypeColor('analyst'),
    optimizer: generateAgentTypeColor('optimizer'),

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
  const [displayMode, setDisplayMode] = useState<'overview' | 'performance' | 'tasks' | 'network' | 'resources'>('overview');
  const telemetry = useTelemetry(`BotsNode-${agent.id}`);
  const threeJSTelemetry = useThreeJSTelemetry(agent.id);
  const lastPositionRef = useRef<THREE.Vector3>();
  const currentPositionRef = useRef<THREE.Vector3>(position.clone());
  const targetPositionRef = useRef<THREE.Vector3>(position.clone());
  const settings = useSettingsStore(state => state.settings);

  // Enhanced health-based glow color with more precise mapping
  const glowColor = useMemo(() => {
    const health = agent.health || 0;
    if (health >= 95) return '#00FF00'; // Bright green for excellent health
    if (health >= 80) return '#2ECC71'; // Green for good health
    if (health >= 65) return '#F1C40F'; // Yellow for moderate health
    if (health >= 50) return '#F39C12'; // Orange for poor health
    if (health >= 25) return '#E67E22'; // Dark orange for bad health
    return '#E74C3C'; // Red for critical health
  }, [agent.health]);

  // Enhanced size calculation based on workload with multiple factors
  const baseSize = 1.0;
  const cpuScale = agent.cpuUsage ? (agent.cpuUsage / 100) * 0.8 : 0;
  const workloadScale = agent.workload ? agent.workload * 0.6 : 0;
  const activityScale = agent.activity ? agent.activity * 0.4 : 0;
  const tokenScale = agent.tokenRate ? Math.min(agent.tokenRate / 50, 0.5) : 0;

  const size = baseSize + cpuScale + workloadScale + activityScale + tokenScale;
  const clampedSize = Math.max(0.5, Math.min(size, 3.0)); // Clamp between 0.5 and 3.0

  // Enhanced shape based on status and agent type
  const geometry = useMemo(() => {
    const radius = clampedSize;

    switch (agent.status) {
      case 'error':
        return new THREE.TetrahedronGeometry(radius * 1.2); // Sharp edges for error
      case 'terminating':
        return new THREE.OctahedronGeometry(radius); // Diamond shape for terminating
      case 'initializing':
        return new THREE.BoxGeometry(radius, radius, radius); // Cube for initializing
      case 'idle':
        return new THREE.SphereGeometry(radius * 0.8, 12, 12); // Smaller sphere, lower poly for idle
      case 'offline':
        return new THREE.CylinderGeometry(radius * 0.5, radius * 0.5, radius); // Flat cylinder for offline
      case 'busy':
        // Different shapes for different agent types when busy
        switch (agent.type) {
          case 'queen':
            return new THREE.IcosahedronGeometry(radius * 1.3, 1); // Large complex shape for queen
          case 'coordinator':
            return new THREE.DodecahedronGeometry(radius * 1.1); // 12-sided for coordinators
          case 'architect':
            return new THREE.ConeGeometry(radius, radius * 1.5, 8); // Pyramid for architects
          default:
            return new THREE.SphereGeometry(radius, 32, 32); // High-poly sphere for active agents
        }
      case 'active':
      default:
        return new THREE.SphereGeometry(radius, 24, 24); // Medium-poly sphere for active
    }
  }, [agent.status, agent.type, clampedSize]);

  useFrame((state) => {
    if (!groupRef.current || !meshRef.current || !glowRef.current) return;

    telemetry.startRender();

    // Log position updates for debugging clustering
    if (!lastPositionRef.current || !lastPositionRef.current.equals(position)) {
      threeJSTelemetry.logPositionUpdate(
        { x: position.x, y: position.y, z: position.z },
        { agentType: agent.type, agentStatus: agent.status }
      );
      lastPositionRef.current = position.clone();
    }

    // Smooth position interpolation for real-time updates
    targetPositionRef.current.copy(position);
    
    // Interpolate current position towards target
    // TODO: Extract to configuration - magic number for animation smoothness
    const lerpFactor = 0.15; // TODO: Config - Adjust for smoothness (0.05 = very smooth, 0.3 = responsive)
    lerpVector3(currentPositionRef.current, targetPositionRef.current, lerpFactor);

    // Update group position with interpolated value
    groupRef.current.position.copy(currentPositionRef.current);

    // Enhanced pulse animation based on token rate, health, and status
    // TODO: Extract to configuration - magic numbers for pulse calculations
    if (agent.status === 'active' || agent.status === 'busy') {
      // Base pulse speed influenced by token rate and health
      const tokenMultiplier = agent.tokenRate ? Math.min(agent.tokenRate / 10, 3) : 1; // TODO: Config - tokenRate divisor (10) and max multiplier (3)
      const healthMultiplier = agent.health ? Math.max(0.3, agent.health / 100) : 1; // TODO: Config - health min (0.3) and divisor (100)
      const pulseSpeed = 2 * tokenMultiplier * healthMultiplier; // TODO: Config - base pulse speed multiplier (2)
      const pulse = Math.sin(state.clock.elapsedTime * pulseSpeed + index) * 0.15 + 1; // TODO: Config - pulse amplitude (0.15) and offset (1)

      meshRef.current.scale.setScalar(pulse * clampedSize);

      // Dynamic glow intensity based on multiple factors
      const tokenGlow = agent.tokenRate ? Math.min(agent.tokenRate / 20, 2) : 1;
      const healthGlow = agent.health ? (agent.health / 100) : 0.5;
      const statusGlow = agent.status === 'busy' ? 1.5 : 1.0;
      const glowIntensity = tokenGlow * healthGlow * statusGlow;

      glowRef.current.scale.setScalar(pulse * 1.5 * glowIntensity);
    } else if (agent.status === 'error') {
      // Error state: rapid red pulsing
      const errorPulse = Math.sin(state.clock.elapsedTime * 8 + index) * 0.3 + 1;
      meshRef.current.scale.setScalar(errorPulse * clampedSize);
      glowRef.current.scale.setScalar(errorPulse * 2.0);
    }

    // Enhanced rotation for busy agents (faster with higher token rate)
    if (agent.status === 'busy') {
      const rotationSpeed = agent.tokenRate ? 0.01 * (1 + agent.tokenRate / 50) : 0.01;
      meshRef.current.rotation.y += rotationSpeed;
    }

    // Enhanced high-activity animations
    if (agent.tokenRate && agent.tokenRate > 30) {
      // Gentle floating motion for high-activity agents
      const vibration = Math.sin(state.clock.elapsedTime * 15 + index) * 0.03;
      const float = Math.cos(state.clock.elapsedTime * 3 + index) * 0.1;
      meshRef.current.position.y += vibration + float;
    }

    // Memory pressure indicator - slight shake if memory usage is high
    if (agent.memoryUsage && agent.memoryUsage > 80) {
      const shake = Math.sin(state.clock.elapsedTime * 25) * 0.01;
      meshRef.current.position.x += shake;
      meshRef.current.position.z += shake * 0.7;
    }

    // Critical health warning - dramatic pulsing
    if (agent.health && agent.health < 25) {
      const criticalPulse = Math.sin(state.clock.elapsedTime * 12) * 0.5 + 1;
      meshRef.current.scale.multiplyScalar(criticalPulse);
    }

    telemetry.endRender();

    // Log animation frame for telemetry
    threeJSTelemetry.logAnimationFrame(
      { x: position.x, y: position.y, z: position.z },
      { x: meshRef.current.rotation.x, y: meshRef.current.rotation.y, z: meshRef.current.rotation.z }
    );
  });

  // Use actual logs or empty array
  const processingLogs = formatProcessingLogs(agent.processingLogs);

  return (
    <group ref={groupRef}>
      {/* Glow effect */}
      {/* TODO: Config - base opacity (0.15), hover opacity (0.1), tokenRate opacity (0.2/100) */}
      <mesh ref={glowRef}>
        <sphereGeometry args={[clampedSize * 1.5, 16, 16]} />
        <meshBasicMaterial
          color={glowColor}
          transparent
          opacity={0.15 + (hover ? 0.1 : 0) + (agent.tokenRate ? Math.min(agent.tokenRate / 100, 0.2) : 0)}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </mesh>

      {/* Main node */}
      <mesh
        ref={meshRef}
        geometry={geometry}
        onPointerOver={() => {
          setHover(true);
          telemetry.logInteraction('hover_start', {
            agentId: agent.id,
            agentType: agent.type,
            health: agent.health,
            cpuUsage: agent.cpuUsage,
            tokenRate: agent.tokenRate,
            status: agent.status,
            nodeSize: clampedSize
          });
        }}
        onPointerOut={() => {
          setHover(false);
          telemetry.logInteraction('hover_end', {
            agentId: agent.id,
            agentType: agent.type,
            hoverDuration: 'hover_ended'
          });
        }}
        onClick={() => {
          // Cycle through display modes on click
          const modes: Array<'overview' | 'performance' | 'tasks' | 'network' | 'resources'> =
            ['overview', 'performance', 'tasks', 'network', 'resources'];
          const currentIndex = modes.indexOf(displayMode);
          const nextMode = modes[(currentIndex + 1) % modes.length];
          setDisplayMode(nextMode);

          telemetry.logInteraction('click', {
            agentId: agent.id,
            agentType: agent.type,
            displayMode: nextMode,
            position: { x: position.x, y: position.y, z: position.z },
            health: agent.health,
            status: agent.status,
            currentTask: agent.currentTask,
            capabilities: agent.capabilities?.slice(0, 3)
          });
        }}
      >
        <meshStandardMaterial
          color={color}
          emissive={glowColor}
          emissiveIntensity={(() => {
            // Get glow settings from central store
            const glowSettings = settings?.visualisation?.glow;
            const baseIntensity = glowSettings?.nodeGlowStrength ?? 0.7;

            // Apply status-based modulation
            if (agent.status === 'active' || agent.status === 'busy') {
              return baseIntensity * 0.7; // 70% of base for active agents
            } else {
              return baseIntensity * 0.3; // 30% of base for idle agents
            }
          })()}
          metalness={0.3}
          roughness={0.7}
          transparent={agent.status === 'error' || agent.status === 'terminating'}
          opacity={agent.status === 'error' || agent.status === 'terminating' ? 0.7 : 1.0}
        />
      </mesh>

      {/* Enhanced agent info display */}
      {(hover || agent.status === 'active' || agent.status === 'busy') && (
        <Html
          center
          distanceFactor={8}
          style={{
            transition: 'all 0.3s ease-in-out',
            opacity: hover ? 1 : 0.85,
            pointerEvents: 'none',
            position: 'absolute',
            top: `${-clampedSize * 25}px`,
            left: '0',
            transform: hover ? 'scale(1.05)' : 'scale(1)',
            filter: hover ? 'drop-shadow(0 4px 8px rgba(0,0,0,0.3))' : 'none'
          }}
        >
          <AgentStatusBadges agent={agent} logs={processingLogs} />
        </Html>
      )}

      {/* Performance indicators for high-performance agents */}
      {(agent.tokenRate > 30 || agent.cpuUsage > 80) && (
        <group>
          {/* CPU usage ring */}
          <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, clampedSize + 0.2, 0]}>
            <ringGeometry args={[clampedSize * 1.1, clampedSize * 1.3, 16]} />
            <meshBasicMaterial
              color={agent.cpuUsage > 90 ? '#E74C3C' : agent.cpuUsage > 70 ? '#F39C12' : '#2ECC71'}
              transparent
              opacity={0.6}
              side={THREE.DoubleSide}
            />
          </mesh>

          {/* Token rate indicator particles */}
          {agent.tokenRate > 50 && [
            ...Array(Math.min(Math.floor(agent.tokenRate / 10), 8))
          ].map((_, i) => {
            const angle = (i / 8) * Math.PI * 2;
            const radius = clampedSize * 2;
            const x = Math.cos(angle + Date.now() * 0.001) * radius;
            const z = Math.sin(angle + Date.now() * 0.001) * radius;
            return (
              <mesh key={i} position={[x, 0, z]}>
                <sphereGeometry args={[0.03, 6, 6]} />
                <meshBasicMaterial
                  color="#F39C12"
                  transparent
                  opacity={0.8}
                />
              </mesh>
            );
          })}
        </group>
      )}

      {/* Enhanced 3D Text labels - content changes based on display mode */}
      <Billboard
        follow={true}
        lockX={false}
        lockY={false}
        lockZ={false}
      >
        {/* Mode indicator */}
        <Text
          position={[0, clampedSize + 0.8, 0]}
          fontSize={0.18}
          color="#3498DB"
          anchorX="center"
          anchorY="middle"
          outlineWidth={0.02}
          outlineColor="black"
        >
          [{displayMode.toUpperCase()}]
        </Text>

        {/* Agent name/ID always shown */}
        <Text
          position={[0, -clampedSize - 0.7, 0]}
          fontSize={0.4}
          color="white"
          anchorX="center"
          anchorY="middle"
          outlineWidth={0.05}
          outlineColor="black"
        >
          {agent.name || String(agent.id).slice(0, 8)}
        </Text>

        {/* Dynamic content based on display mode */}
        {displayMode === 'overview' && (
          <>
            <Text
              position={[0, -clampedSize - 1.1, 0]}
              fontSize={0.25}
              color={color} // Use the passed color prop instead of looking up in colors array
              anchorX="center"
              anchorY="middle"
              outlineWidth={0.03}
              outlineColor="black"
            >
              {agent.type.toUpperCase()}
            </Text>
            <Text
              position={[0, -clampedSize - 1.4, 0]}
              fontSize={0.2}
              color={glowColor}
              anchorX="center"
              anchorY="middle"
              outlineWidth={0.02}
              outlineColor="black"
            >
              Health: {agent.health ? `${agent.health.toFixed(0)}%` : 'N/A'}
            </Text>
            <Text
              position={[0, -clampedSize - 1.7, 0]}
              fontSize={0.15}
              color="#95A5A6"
              anchorX="center"
              anchorY="middle"
              outlineWidth={0.02}
              outlineColor="black"
            >
              Status: {agent.status}
            </Text>
          </>
        )}

        {displayMode === 'performance' && (
          <>
            <Text
              position={[0, -clampedSize - 1.1, 0]}
              fontSize={0.2}
              color={agent.cpuUsage > 80 ? '#E74C3C' : agent.cpuUsage > 50 ? '#F39C12' : '#2ECC71'}
              anchorX="center"
              anchorY="middle"
              outlineWidth={0.02}
              outlineColor="black"
            >
              CPU: {agent.cpuUsage?.toFixed(0) || 0}%
            </Text>
            <Text
              position={[0, -clampedSize - 1.4, 0]}
              fontSize={0.2}
              color="#9B59B6"
              anchorX="center"
              anchorY="middle"
              outlineWidth={0.02}
              outlineColor="black"
            >
              MEM: {agent.memoryUsage?.toFixed(0) || 0}%
            </Text>
            <Text
              position={[0, -clampedSize - 1.7, 0]}
              fontSize={0.18}
              color={agent.tokenRate > 20 ? '#E67E22' : '#3498DB'}
              anchorX="center"
              anchorY="middle"
              outlineWidth={0.02}
              outlineColor="black"
            >
              Tokens: {agent.tokenRate?.toFixed(1) || 0}/min
            </Text>
            <Text
              position={[0, -clampedSize - 2.0, 0]}
              fontSize={0.15}
              color="#F39C12"
              anchorX="center"
              anchorY="middle"
              outlineWidth={0.02}
              outlineColor="black"
            >
              Total: {agent.tokens?.toLocaleString() || 0}
            </Text>
          </>
        )}

        {displayMode === 'tasks' && (
          <>
            <Text
              position={[0, -clampedSize - 1.1, 0]}
              fontSize={0.2}
              color="#2ECC71"
              anchorX="center"
              anchorY="middle"
              outlineWidth={0.02}
              outlineColor="black"
            >
              Active: {agent.tasksActive || 0}
            </Text>
            <Text
              position={[0, -clampedSize - 1.4, 0]}
              fontSize={0.2}
              color="#3498DB"
              anchorX="center"
              anchorY="middle"
              outlineWidth={0.02}
              outlineColor="black"
            >
              Done: {agent.tasksCompleted || 0}
            </Text>
            <Text
              position={[0, -clampedSize - 1.7, 0]}
              fontSize={0.15}
              color="#95A5A6"
              anchorX="center"
              anchorY="middle"
              outlineWidth={0.02}
              outlineColor="black"
            >
              {agent.currentTask ? agent.currentTask.substring(0, 20) + '...' : 'Idle'}
            </Text>
            {agent.successRate !== undefined && (
              <Text
                position={[0, -clampedSize - 2.0, 0]}
                fontSize={0.15}
                color={agent.successRate > 0.8 ? '#27AE60' : agent.successRate > 0.6 ? '#F39C12' : '#E74C3C'}
                anchorX="center"
                anchorY="middle"
                outlineWidth={0.02}
                outlineColor="black"
              >
                Success: {(agent.successRate * 100).toFixed(0)}%
              </Text>
            )}
          </>
        )}

        {displayMode === 'network' && (
          <>
            <Text
              position={[0, -clampedSize - 1.1, 0]}
              fontSize={0.18}
              color="#E67E22"
              anchorX="center"
              anchorY="middle"
              outlineWidth={0.02}
              outlineColor="black"
            >
              Swarm: {agent.swarmId?.substring(0, 8) || 'None'}
            </Text>
            <Text
              position={[0, -clampedSize - 1.4, 0]}
              fontSize={0.18}
              color="#F39C12"
              anchorX="center"
              anchorY="middle"
              outlineWidth={0.02}
              outlineColor="black"
            >
              Mode: {agent.agentMode || 'Default'}
            </Text>
            {agent.parentQueenId && (
              <Text
                position={[0, -clampedSize - 1.7, 0]}
                fontSize={0.15}
                color="#FFD700"
                anchorX="center"
                anchorY="middle"
                outlineWidth={0.02}
                outlineColor="black"
              >
                Queen: {agent.parentQueenId.substring(0, 8)}
              </Text>
            )}
            <Text
              position={[0, -clampedSize - 2.0, 0]}
              fontSize={0.15}
              color="#95A5A6"
              anchorX="center"
              anchorY="middle"
              outlineWidth={0.02}
              outlineColor="black"
            >
              Age: {agent.age ? Math.floor(agent.age / 1000 / 60) : 0}m
            </Text>
          </>
        )}

        {displayMode === 'resources' && (
          <>
            <Text
              position={[0, -clampedSize - 1.1, 0]}
              fontSize={0.18}
              color="#3498DB"
              anchorX="center"
              anchorY="middle"
              outlineWidth={0.02}
              outlineColor="black"
            >
              Workload: {(agent.workload * 100 || 0).toFixed(0)}%
            </Text>
            <Text
              position={[0, -clampedSize - 1.4, 0]}
              fontSize={0.18}
              color="#2ECC71"
              anchorX="center"
              anchorY="middle"
              outlineWidth={0.02}
              outlineColor="black"
            >
              Activity: {(agent.activity * 100 || 0).toFixed(0)}%
            </Text>
            {agent.capabilities && agent.capabilities.length > 0 && (
              <Text
                position={[0, -clampedSize - 1.7, 0]}
                fontSize={0.15}
                color="#9B59B6"
                anchorX="center"
                anchorY="middle"
                outlineWidth={0.02}
                outlineColor="black"
              >
                Caps: {agent.capabilities.length} total
              </Text>
            )}
            <Text
              position={[0, -clampedSize - 2.0, 0]}
              fontSize={0.13}
              color="#95A5A6"
              anchorX="center"
              anchorY="middle"
              outlineWidth={0.02}
              outlineColor="black"
            >
              {agent.capabilities?.[0]?.replace(/_/g, ' ') || 'None'}
            </Text>
          </>
        )}
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
  
  // Enhanced visual properties based on data flow and communication intensity
  const baseWidth = Math.max(0.5, edge.dataVolume / 1000); // Base width from data volume
  const tokenWidth = avgTokenRate > 0 ? Math.min(avgTokenRate / 10, 2) : 0; // Additional width from token rate
  const messageWidth = edge.messageCount > 0 ? Math.min(edge.messageCount / 100, 1.5) : 0; // Width from message frequency
  const lineWidth = isActive ? Math.max(1.5, baseWidth + tokenWidth + messageWidth) : Math.max(0.5, baseWidth * 0.5);
  
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

  // Enhanced animation properties for different communication types
  const shouldAnimate = isActive && (avgTokenRate > 15 || edge.messageCount > 50);
  const animationSpeed = Math.min(avgTokenRate / 10, 3) + Math.min(edge.messageCount / 100, 2);
  const dashOffset = shouldAnimate ? -Date.now() * 0.001 * animationSpeed : 0;

  // Pulse effect for very high activity connections
  const shouldPulse = avgTokenRate > 40 || edge.messageCount > 200;
  const pulseIntensity = shouldPulse ? Math.sin(Date.now() * 0.005) * 0.3 + 1 : 1;

  return (
    <>
      {/* Main connection line */}
      <DreiLine
        points={[sourcePos, targetPos]}
        color={edgeColor}
        lineWidth={lineWidth * pulseIntensity}
        opacity={opacity * pulseIntensity}
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
          lineWidth={lineWidth * 0.5 * pulseIntensity}
          opacity={0.4 * pulseIntensity}
          transparent
          dashed={true}
          dashScale={15}
          dashSize={3}
          dashOffset={-dashOffset * 1.5}
        />
      )}

      {/* Ultra-high activity indicator - third layer for extreme communication */}
      {avgTokenRate > 50 && edge.messageCount > 300 && isActive && (
        <DreiLine
          points={[sourcePos, targetPos]}
          color="#E74C3C"
          lineWidth={lineWidth * 0.3 * pulseIntensity}
          opacity={0.6 * pulseIntensity}
          transparent
          dashed={true}
          dashScale={20}
          dashSize={5}
          dashOffset={-dashOffset * 2}
        />
      )}

      {/* Communication direction indicator - small particles */}
      {isActive && shouldAnimate && (
        <group>
          {[0.2, 0.5, 0.8].map((t, i) => {
            const particlePos = new THREE.Vector3().lerpVectors(sourcePos, targetPos, t + (Date.now() * 0.001 * animationSpeed) % 1);
            return (
              <mesh key={i} position={particlePos}>
                <sphereGeometry args={[0.05, 8, 8]} />
                <meshBasicMaterial
                  color={avgTokenRate > 30 ? '#F39C12' : '#3498DB'}
                  transparent
                  opacity={0.8 * pulseIntensity}
                />
              </mesh>
            );
          })}
        </group>
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
  const telemetry = useTelemetry('BotsVisualization');

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

      // Log agent state changes
      agentTelemetry.logAgentAction(agent.id, agent.type, 'state_update', {
        status: agent.status,
        health: agent.health,
        cpuUsage: agent.cpuUsage,
        tokenRate: agent.tokenRate
      });
      
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

        // Log initial position calculation - check for clustering
        agentTelemetry.logThreeJSOperation('position_update', agent.id, {
          x: newPosition.x,
          y: newPosition.y,
          z: newPosition.z
        }, undefined, {
          reason: 'initial_calculation',
          agentType: agent.type,
          index,
          totalAgents: agents.length
        });
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

    // Log visualization update
    agentTelemetry.logAgentAction('visualization', 'system', 'data_update', {
      agentCount: agentMap.size,
      edgeCount: edgeMap.size,
      hasContextData: !!contextBotsData
    });
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

        // Use dynamic color generation or server-provided colors
        const nodeColor = colors.getAgentColor ? colors.getAgentColor(node.type) : (colors[node.type] || colors.coordinator);

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