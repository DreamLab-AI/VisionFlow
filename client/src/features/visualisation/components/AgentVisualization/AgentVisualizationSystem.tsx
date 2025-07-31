/**
 * Complete Agent Visualization System
 * Main component that orchestrates all agent visualization features
 */
import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars, Environment } from '@react-three/drei';
import * as THREE from 'three';
import {
  AgentNodeData,
  NodeVisualizationConfig,
  FloatingPanelData,
  ConnectionVisualization,
  CoordinationPattern,
  MessageFlowData,
  VisualizationSettings,
  HUDElement,
  SystemAlert,
  DashboardMetrics,
  AgentId
} from './types';
import { EnhancedAgentNode } from './EnhancedAgentNode';
import { MessageFlowVisualization, CoordinationVisualization } from './MessageFlowVisualization';
import {
  PerformanceDashboardPanel,
  CoordinationActivityPanel,
  MessageFlowPanel,
  SystemHealthPanel
} from './FloatingPanels';
import { useAgentDataStore } from './WebSocketIntegration';

// Main visualization system component
export interface AgentVisualizationSystemProps {
  wsConfig: {
    url: string;
    reconnectAttempts: number;
    reconnectDelay: number;
    heartbeatInterval: number;
    messageBufferSize: number;
  };
  initialSettings?: Partial<VisualizationSettings>;
  onAgentSelect?: (agentId: AgentId) => void;
  onAgentHover?: (agentId: AgentId | null) => void;
  className?: string;
}

export const AgentVisualizationSystem: React.FC<AgentVisualizationSystemProps> = ({
  wsConfig,
  initialSettings,
  onAgentSelect,
  onAgentHover,
  className = ''
}) => {
  // WebSocket data store
  const { store, connected, agents, messages, patterns, metrics, alerts } = useAgentDataStore(wsConfig);

  // Visualization state
  const [settings, setSettings] = useState<VisualizationSettings>({
    nodes: {
      showLabels: true,
      showPerformanceRings: true,
      showCapabilityBadges: true,
      animateStateChanges: true,
      nodeSize: 1.0,
      labelDistance: 2.0
    },
    connections: {
      showMessageFlow: true,
      showLatency: true,
      animateMessages: true,
      connectionOpacity: 0.6,
      flowSpeed: 1.0
    },
    panels: {
      defaultVisible: true,
      autoHide: false,
      updateFrequency: 1000,
      maxPanels: 4
    },
    effects: {
      enableParticles: true,
      enableGlow: true,
      enableBloom: true,
      qualityLevel: 'medium'
    },
    ...initialSettings
  });

  const [selectedAgent, setSelectedAgent] = useState<AgentId | null>(null);
  const [hoveredAgent, setHoveredAgent] = useState<AgentId | null>(null);
  const [floatingPanels, setFloatingPanels] = useState<Map<string, FloatingPanelData>>(new Map());
  const [hudElements, setHudElements] = useState<HUDElement[]>([]);

  // Node visualization configs
  const nodeConfigs = useMemo(() => {
    const configs = new Map<string, NodeVisualizationConfig>();
    agents.forEach((agent, id) => {
      configs.set(id, {
        showPerformanceRing: settings.nodes.showPerformanceRings,
        showCapabilityBadges: settings.nodes.showCapabilityBadges,
        showStateIndicator: true,
        showActivityPulse: true,
        enableHoverEffects: true,
        layers: [
          { id: 'core', type: 'core', visible: true, opacity: 1.0, animationSpeed: 1.0 },
          { id: 'performance', type: 'performance', visible: settings.nodes.showPerformanceRings, opacity: 0.8, animationSpeed: 0.5 },
          { id: 'capability', type: 'capability', visible: settings.nodes.showCapabilityBadges, opacity: 0.9, animationSpeed: 0.3 },
          { id: 'state', type: 'state', visible: true, opacity: 1.0, animationSpeed: 2.0 },
          { id: 'activity', type: 'activity', visible: true, opacity: 0.6, animationSpeed: 1.5 }
        ]
      });
    });
    return configs;
  }, [agents, settings]);

  // Connection data processing
  const connections = useMemo(() => {
    const connectionMap = new Map<string, ConnectionVisualization>();
    
    // Process recent messages to build connections
    const recentMessages = messages.filter(msg => 
      Date.now() - msg.timestamp.getTime() < 300000 // Last 5 minutes
    );

    recentMessages.forEach(msg => {
      const targets = Array.isArray(msg.to) ? msg.to : [msg.to];
      
      targets.forEach(target => {
        const connectionKey = `${msg.from.id}-${target.id}`;
        const existing = connectionMap.get(connectionKey);
        
        if (existing) {
          existing.messageCount++;
          existing.lastActivity = msg.timestamp;
          if (msg.latency) {
            existing.latency = (existing.latency + msg.latency) / 2; // Average latency
          }
          if (msg.status === 'delivered') {
            existing.reliability = Math.min(1, existing.reliability + 0.1);
          }
        } else {
          connectionMap.set(connectionKey, {
            source: msg.from,
            target: target,
            strength: 0.3,
            latency: msg.latency || 0,
            reliability: msg.status === 'delivered' ? 0.8 : 0.5,
            messageCount: 1,
            lastActivity: msg.timestamp,
            type: Array.isArray(msg.to) && msg.to.length > 1 ? 'broadcast' : 'direct'
          });
        }
      });
    });

    return Array.from(connectionMap.values());
  }, [messages]);

  // Event handlers
  const handleAgentHover = useCallback((agentId: AgentId | null) => {
    setHoveredAgent(agentId);
    onAgentHover?.(agentId);
  }, [onAgentHover]);

  const handleAgentClick = useCallback((agentId: AgentId) => {
    setSelectedAgent(selectedAgent?.id === agentId.id ? null : agentId);
    onAgentSelect?.(agentId);
  }, [selectedAgent, onAgentSelect]);

  const handleAgentContextMenu = useCallback((agentId: AgentId, event: React.MouseEvent) => {
    event.preventDefault();
    // Show context menu for agent actions
    console.log('Context menu for agent:', agentId);
  }, []);

  // Panel management
  const togglePanel = useCallback((type: string) => {
    setFloatingPanels(prev => {
      const newPanels = new Map(prev);
      const existingPanel = newPanels.get(type);
      
      if (existingPanel) {
        newPanels.delete(type);
      } else {
        const panelData: FloatingPanelData = {
          id: type,
          type: type as any,
          position: { x: 20 + newPanels.size * 300, y: 20 },
          size: { width: 280, height: 300 },
          visible: true,
          pinned: false,
          data: null,
          updateFrequency: settings.panels.updateFrequency
        };
        newPanels.set(type, panelData);
      }
      
      return newPanels;
    });
  }, [settings.panels.updateFrequency]);

  const handlePanelClose = useCallback((panelId: string) => {
    setFloatingPanels(prev => {
      const newPanels = new Map(prev);
      newPanels.delete(panelId);
      return newPanels;
    });
  }, []);

  const handlePanelPin = useCallback((panelId: string) => {
    setFloatingPanels(prev => {
      const newPanels = new Map(prev);
      const panel = newPanels.get(panelId);
      if (panel) {
        panel.pinned = !panel.pinned;
        newPanels.set(panelId, panel);
      }
      return newPanels;
    });
  }, []);

  // Initialize default panels
  useEffect(() => {
    if (settings.panels.defaultVisible && floatingPanels.size === 0) {
      const defaultPanels = ['performance', 'system-health'];
      defaultPanels.forEach((type, index) => {
        const panelData: FloatingPanelData = {
          id: type,
          type: type as any,
          position: { x: 20 + index * 300, y: 20 },
          size: { width: 280, height: 300 },
          visible: true,
          pinned: false,
          data: null,
          updateFrequency: settings.panels.updateFrequency
        };
        setFloatingPanels(prev => new Map(prev).set(type, panelData));
      });
    }
  }, [settings.panels.defaultVisible, settings.panels.updateFrequency, floatingPanels.size]);

  // Update HUD elements based on system state
  useEffect(() => {
    const newHudElements: HUDElement[] = [
      {
        id: 'connection-status',
        type: 'status',
        position: 'top-right',
        priority: 1,
        data: { connected, agentCount: agents.size },
        visible: true
      }
    ];

    // Add critical alerts to HUD
    const criticalAlerts = alerts.filter(alert => alert.level === 'critical' && !alert.acknowledged);
    criticalAlerts.forEach((alert, index) => {
      newHudElements.push({
        id: `alert-${alert.id}`,
        type: 'alert',
        position: 'top-left',
        priority: 10 + index,
        data: alert,
        visible: true
      });
    });

    setHudElements(newHudElements);
  }, [connected, agents.size, alerts]);

  return (
    <div className={`relative w-full h-full ${className}`}>
      {/* 3D Visualization Canvas */}
      <Canvas
        camera={{ position: [10, 10, 10], fov: 50 }}
        gl={{ 
          antialias: settings.effects.qualityLevel !== 'low',
          alpha: true,
          powerPreference: 'high-performance'
        }}
        dpr={settings.effects.qualityLevel === 'high' ? 2 : 1}
      >
        {/* Environment and lighting */}
        <ambientLight intensity={0.3} />
        <directionalLight position={[10, 10, 5]} intensity={0.8} />
        <pointLight position={[-10, -10, -5]} intensity={0.5} />
        
        {settings.effects.enableParticles && <Stars />}
        {settings.effects.qualityLevel === 'high' && (
          <Environment preset="night" />
        )}

        {/* Agent nodes */}
        {Array.from(agents.entries()).map(([id, agent]) => {
          const config = nodeConfigs.get(id);
          if (!config) return null;

          return (
            <EnhancedAgentNode
              key={id}
              data={agent}
              config={config}
              onHover={handleAgentHover}
              onClick={handleAgentClick}
              onContextMenu={handleAgentContextMenu}
            />
          );
        })}

        {/* Message flow visualization */}
        {settings.connections.showMessageFlow && (
          <MessageFlowVisualization
            connections={connections}
            messages={messages.filter(msg => msg.status === 'sending')}
            animationSpeed={settings.connections.flowSpeed}
            showLatency={settings.connections.showLatency}
          />
        )}

        {/* Coordination patterns */}
        <CoordinationVisualization
          patterns={patterns}
          agents={agents}
          showHierarchy={true}
          animateFormation={true}
        />

        {/* Camera controls */}
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          dampingFactor={0.1}
          enableDamping
          maxDistance={100}
          minDistance={2}
        />
      </Canvas>

      {/* Floating panels */}
      {Array.from(floatingPanels.entries()).map(([id, panelData]) => {
        switch (panelData.type) {
          case 'performance':
            return metrics ? (
              <PerformanceDashboardPanel
                key={id}
                metrics={metrics}
                panelData={panelData}
                onClose={() => handlePanelClose(id)}
                onPin={() => handlePanelPin(id)}
              />
            ) : null;

          case 'coordination':
            return (
              <CoordinationActivityPanel
                key={id}
                patterns={patterns}
                panelData={panelData}
                onClose={() => handlePanelClose(id)}
                onPin={() => handlePanelPin(id)}
              />
            );

          case 'message-flow':
            return (
              <MessageFlowPanel
                key={id}
                messages={messages}
                panelData={panelData}
                onClose={() => handlePanelClose(id)}
                onPin={() => handlePanelPin(id)}
              />
            );

          case 'system-health':
            return metrics ? (
              <SystemHealthPanel
                key={id}
                alerts={alerts}
                systemMetrics={metrics.overview}
                panelData={panelData}
                onClose={() => handlePanelClose(id)}
                onPin={() => handlePanelPin(id)}
              />
            ) : null;

          default:
            return null;
        }
      })}

      {/* HUD Elements */}
      <div className="absolute inset-0 pointer-events-none">
        {hudElements.map((element) => (
          <HUDElementComponent
            key={element.id}
            element={element}
            onInteract={() => {}} // Make interactive if needed
          />
        ))}
      </div>

      {/* Control Panel */}
      <div className="absolute top-4 left-4 space-y-2">
        <button
          onClick={() => togglePanel('performance')}
          className={`px-3 py-1 rounded text-sm ${
            floatingPanels.has('performance') 
              ? 'bg-cyan-500/30 text-cyan-300' 
              : 'bg-black/50 text-cyan-500 hover:bg-cyan-500/20'
          }`}
        >
          Performance
        </button>
        <button
          onClick={() => togglePanel('coordination')}
          className={`px-3 py-1 rounded text-sm ${
            floatingPanels.has('coordination') 
              ? 'bg-cyan-500/30 text-cyan-300' 
              : 'bg-black/50 text-cyan-500 hover:bg-cyan-500/20'
          }`}
        >
          Coordination
        </button>
        <button
          onClick={() => togglePanel('message-flow')}
          className={`px-3 py-1 rounded text-sm ${
            floatingPanels.has('message-flow') 
              ? 'bg-cyan-500/30 text-cyan-300' 
              : 'bg-black/50 text-cyan-500 hover:bg-cyan-500/20'
          }`}
        >
          Messages
        </button>
        <button
          onClick={() => togglePanel('system-health')}
          className={`px-3 py-1 rounded text-sm ${
            floatingPanels.has('system-health') 
              ? 'bg-cyan-500/30 text-cyan-300' 
              : 'bg-black/50 text-cyan-500 hover:bg-cyan-500/20'
          }`}
        >
          System Health
        </button>
      </div>

      {/* Connection Status */}
      <div className="absolute bottom-4 right-4">
        <div className={`px-3 py-1 rounded text-sm ${
          connected 
            ? 'bg-green-500/20 text-green-300 border border-green-500/30' 
            : 'bg-red-500/20 text-red-300 border border-red-500/30'
        }`}>
          {connected ? 'Connected' : 'Disconnected'} • {agents.size} agents
        </div>
      </div>
    </div>
  );
};

// HUD Element Component
const HUDElementComponent: React.FC<{
  element: HUDElement;
  onInteract: () => void;
}> = ({ element, onInteract }) => {
  const positionClasses = {
    'top-left': 'top-4 left-4',
    'top-right': 'top-4 right-4',
    'bottom-left': 'bottom-4 left-4',
    'bottom-right': 'bottom-4 right-4',
    'center': 'top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2'
  };

  if (!element.visible) return null;

  return (
    <div className={`absolute ${positionClasses[element.position]} pointer-events-auto`}>
      {element.type === 'alert' && (
        <div className={`p-2 rounded border ${
          element.data.level === 'critical' 
            ? 'bg-red-900/50 border-red-500/50 text-red-300'
            : 'bg-yellow-900/50 border-yellow-500/50 text-yellow-300'
        }`}>
          <div className="text-xs font-semibold">{element.data.level.toUpperCase()}</div>
          <div className="text-sm">{element.data.message}</div>
        </div>
      )}
      
      {element.type === 'status' && (
        <div className="bg-black/50 border border-cyan-500/30 rounded px-2 py-1 text-cyan-300 text-sm">
          {element.data.connected ? '🟢' : '🔴'} {element.data.agentCount} agents
        </div>
      )}
    </div>
  );
};

export default AgentVisualizationSystem;