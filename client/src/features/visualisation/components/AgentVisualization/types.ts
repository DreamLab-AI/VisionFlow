// Enhanced Agent Visualization Types
// Integrating rich data structures from agentic-flow system

import * as THREE from 'three';

// Core Agent Data Models (from agentic-flow extraction)
export interface AgentId {
  id: string;
  namespace?: string;
}

export enum AgentType {
  COORDINATOR = 'coordinator',
  EXECUTOR = 'executor',
  ANALYZER = 'analyzer',
  MONITOR = 'monitor',
  SPECIALIST = 'specialist',
  RESEARCHER = 'researcher',
  CODER = 'coder',
  TESTER = 'tester',
  REVIEWER = 'reviewer',
  ARCHITECT = 'architect'
}

export enum AgentState {
  IDLE = 'idle',
  THINKING = 'thinking',
  EXECUTING = 'executing',
  COMMUNICATING = 'communicating',
  COORDINATING = 'coordinating',
  ERROR = 'error',
  TERMINATED = 'terminated'
}

export interface PerformanceMetrics {
  tasksCompleted: number;
  successRate: number; // 0-1
  averageResponseTime: number; // ms
  resourceUtilization: number; // 0-1
  communicationEfficiency: number; // 0-1
  uptime: number; // ms
  messageQueueSize: number;
  connectionCount: number;
  memoryUsage: number; // bytes
}

export interface AgentCapability {
  name: string;
  category: 'coordination' | 'execution' | 'analysis' | 'communication' | 'specialized';
  level: number; // 1-5
  description?: string;
}

export interface Goal {
  id: string;
  description: string;
  type: 'ACHIEVE' | 'MAINTAIN' | 'QUERY' | 'PERFORM' | 'PREVENT';
  priority: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  status: 'PENDING' | 'ACTIVE' | 'COMPLETED' | 'FAILED' | 'SUSPENDED';
  progress: number; // 0-1
  deadline?: Date;
  dependencies: string[];
}

export interface EnhancedAgent {
  // Core identity
  id: AgentId;
  name: string;
  type: AgentType;
  state: AgentState;
  
  // Capabilities and goals
  capabilities: AgentCapability[];
  goals: Goal[];
  
  // Performance data
  performance: PerformanceMetrics;
  
  // Metadata
  createdAt: Date;
  updatedAt: Date;
  version: string;
  tags: string[];
  
  // 3D positioning
  position?: THREE.Vector3;
  velocity?: THREE.Vector3;
  
  // Team relationships
  teamId?: string;
  teamRole?: 'leader' | 'member' | 'specialist';
  parentAgentId?: string;
  childAgentIds: string[];
  
  // Activity tracking
  currentTask?: string;
  lastActivity: Date;
  processingLogs: string[];
}

// Message Flow and Communication
export enum MessageType {
  REQUEST = 'request',
  RESPONSE = 'response',
  INFORM = 'inform',
  QUERY = 'query',
  COMMAND = 'command',
  BROADCAST = 'broadcast',
  COORDINATION = 'coordination',
  ERROR = 'error'
}

export enum MessagePriority {
  CRITICAL = 'critical',
  HIGH = 'high',
  NORMAL = 'normal',
  LOW = 'low'
}

export interface Message {
  id: string;
  from: AgentId;
  to: AgentId | AgentId[];
  type: MessageType;
  priority: MessagePriority;
  content: any;
  timestamp: Date;
  replyTo?: string;
  ttl?: number;
  latency?: number;
  success: boolean;
}

export interface MessageFlow {
  source: string;
  target: string;
  messages: Message[];
  averageLatency: number;
  messageRate: number; // messages per second
  successRate: number;
  bandwidth: number; // relative 0-1
}

// Coordination Patterns
export enum CoordinationPattern {
  HIERARCHICAL = 'hierarchical',
  MESH = 'mesh',
  PIPELINE = 'pipeline',
  CONSENSUS = 'consensus',
  BARRIER = 'barrier',
  MATRIX = 'matrix'
}

export interface CoordinationInstance {
  id: string;
  pattern: CoordinationPattern;
  participants: AgentId[];
  status: 'forming' | 'active' | 'completing' | 'completed' | 'failed';
  progress: number; // 0-1
  startedAt: Date;
  completedAt?: Date;
  metadata: Record<string, any>;
}

// System Metrics
export interface SystemMetrics {
  activeAgents: number;
  totalAgents: number;
  averagePerformance: number;
  messageRate: number;
  errorRate: number;
  networkHealth: number; // 0-1
  resourceUtilization: number; // 0-1
  coordinationEfficiency: number; // 0-1
}

// Visualization State
export interface VisualizationState {
  agents: Map<string, EnhancedAgent>;
  messageFlows: Map<string, MessageFlow>;
  coordinationInstances: Map<string, CoordinationInstance>;
  systemMetrics: SystemMetrics;
  selectedAgentId?: string;
  hoveredAgentId?: string;
  showPerformanceRings: boolean;
  showCapabilityBadges: boolean;
  showMessageFlow: boolean;
  showCoordinationPatterns: boolean;
  qualityLevel: 'low' | 'medium' | 'high';
}

// WebSocket Events
export interface AgentEvent {
  type: 'agent_updated' | 'agent_created' | 'agent_removed' | 'agent_state_changed';
  agentId: AgentId;
  timestamp: Date;
  data: Partial<EnhancedAgent>;
}

export interface MessageEvent {
  type: 'message_sent' | 'message_received' | 'message_failed';
  message: Message;
  timestamp: Date;
}

export interface CoordinationEvent {
  type: 'coordination_started' | 'coordination_updated' | 'coordination_completed';
  coordination: CoordinationInstance;
  timestamp: Date;
}

export interface SystemEvent {
  type: 'metrics_updated' | 'system_alert' | 'performance_warning';
  data: any;
  timestamp: Date;
}

// UI Component Props
export interface EnhancedAgentNodeProps {
  agent: EnhancedAgent;
  position: THREE.Vector3;
  isSelected: boolean;
  isHovered: boolean;
  showPerformanceRing: boolean;
  showCapabilityBadges: boolean;
  qualityLevel: 'low' | 'medium' | 'high';
  onSelect: (agentId: string) => void;
  onHover: (agentId: string | null) => void;
}

export interface FloatingPanelProps {
  title: string;
  position: { x: number; y: number };
  size: { width: number; height: number };
  isVisible: boolean;
  isPinned: boolean;
  children: React.ReactNode;
  onMove: (position: { x: number; y: number }) => void;
  onResize: (size: { width: number; height: number }) => void;
  onPin: (pinned: boolean) => void;
  onClose: () => void;
}

export interface MessageFlowVisualizationProps {
  messageFlows: MessageFlow[];
  showLatencyIndicators: boolean;
  showPriorityColors: boolean;
  animationSpeed: number;
  qualityLevel: 'low' | 'medium' | 'high';
}

// Configuration
export interface VisualizationConfig {
  rendering: {
    qualityLevel: 'low' | 'medium' | 'high';
    maxVisibleAgents: number;
    cullingDistance: number;
    animationSpeed: number;
  };
  features: {
    performanceRings: boolean;
    capabilityBadges: boolean;
    messageFlow: boolean;
    coordinationPatterns: boolean;
    floatingPanels: boolean;
    soundEffects: boolean;
  };
  colors: {
    agentTypes: Record<AgentType, string>;
    agentStates: Record<AgentState, string>;
    messagePriorities: Record<MessagePriority, string>;
    coordinationPatterns: Record<CoordinationPattern, string>;
  };
  layout: {
    nodeSpacing: number;
    clusterRadius: number;
    repulsionForce: number;
    attractionForce: number;
  };
}