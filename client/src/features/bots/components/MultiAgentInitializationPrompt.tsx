import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom';
import { createLogger } from '../../../utils/logger';
import { apiService } from '../../../services/apiService';
import { botsWebSocketIntegration } from '../services/BotsWebSocketIntegration';

const logger = createLogger('MultiAgentInitializationPrompt');

interface MultiAgentInitializationPromptProps {
  onClose: () => void;
  onInitialized: () => void;
}

export const MultiAgentInitializationPrompt: React.FC<MultiAgentInitializationPromptProps> = ({
  onClose,
  onInitialized
}) => {
  const [isLoading, setIsLoading] = useState(false);
  const [mcpConnected, setMcpConnected] = useState<boolean | null>(null);
  const [topology, setTopology] = useState<'mesh' | 'hierarchical' | 'ring' | 'star'>('mesh');
  const [maxAgents, setMaxAgents] = useState(8);
  const [enableNeural, setEnableNeural] = useState(true);
  const [agentTypes, setAgentTypes] = useState({
    queen: false,
    coordinator: true,
    researcher: true,
    coder: true,
    analyst: true,
    tester: true,
    architect: true,
    optimizer: true,
    reviewer: false,
    documenter: false,
    monitor: false,
    specialist: false,
  });
  const [customPrompt, setCustomPrompt] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [portalContainer, setPortalContainer] = useState<HTMLElement | null>(null);

  // Check MCP connection status
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const response = await fetch(`${apiService.getBaseUrl()}/bots/status`);
        const data = await response.json();
        setMcpConnected(data.connected);
      } catch (error) {
        setMcpConnected(false);
      }
    };

    // Initial check
    checkConnection();

    // Poll every 3 seconds
    const interval = setInterval(checkConnection, 3000);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    // Create a portal container at the root level
    const container = document.createElement('div');
    container.id = 'multi-agent-modal-portal';
    container.style.position = 'fixed';
    container.style.top = '0';
    container.style.left = '0';
    container.style.width = '100%';
    container.style.height = '100%';
    container.style.zIndex = '9999999';
    container.style.pointerEvents = 'none';
    document.body.appendChild(container);
    setPortalContainer(container);

    return () => {
      document.body.removeChild(container);
    };
  }, []);

  const handleInitialize = async () => {
    setIsLoading(true);
    setError(null);

    try {
      // Count selected agent types
      const selectedAgentTypes = Object.entries(agentTypes)
        .filter(([_, enabled]) => enabled)
        .map(([type, _]) => type);

      if (selectedAgentTypes.length === 0) {
        setError('Please select at least one agent type');
        setIsLoading(false);
        return;
      }

      if (!customPrompt.trim()) {
        setError('Please provide a task for the hive mind');
        setIsLoading(false);
        return;
      }

      // Prepare initialization configuration for hive mind
      const config = {
        topology,
        maxAgents,
        strategy: 'adaptive', // Adaptive hive mind strategy
        enableNeural,
        agentTypes: selectedAgentTypes,
        customPrompt: customPrompt.trim(),
      };

      logger.info('Spawning hive mind with config:', config);

      // Call API to initialize multi-agent
      const fullUrl = `${apiService.getBaseUrl()}/bots/initialize-multi-agent`;
      logger.info('Calling API endpoint:', fullUrl);
      const response = await apiService.post('/bots/initialize-multi-agent', config);

      if (response.success) {
        logger.info('Hive mind spawned successfully:', response);
        // Restart polling to get the new agents
        botsWebSocketIntegration.restartPolling();
        onInitialized();
      } else {
        throw new Error(response.error || 'Failed to spawn hive mind');
      }
    } catch (err) {
      logger.error('Failed to spawn hive mind:', err);
      setError(err instanceof Error ? err.message : 'Failed to spawn hive mind');
    } finally {
      setIsLoading(false);
    }
  };

  if (!portalContainer) return null;

  return ReactDOM.createPortal(
    <div style={{
      position: 'fixed',
      top: '50%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
      backgroundColor: '#141414',  // Solid black background
      border: '2px solid #F1C40F',
      borderRadius: '8px',
      padding: '20px',
      maxWidth: '500px',
      width: '90%',
      maxHeight: '80vh',
      overflow: 'auto',
      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.9), 0 0 80px rgba(241, 196, 15, 0.4)',
      pointerEvents: 'auto',
    }}>
      <h2 style={{
        margin: '0 0 20px 0',
        color: '#F1C40F',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span>🧠 Spawn Hive Mind</span>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '14px',
            color: mcpConnected === null ? '#999' : mcpConnected ? '#2ECC71' : '#E74C3C'
          }}>
            <div style={{
              width: '10px',
              height: '10px',
              borderRadius: '50%',
              backgroundColor: mcpConnected === null ? '#999' : mcpConnected ? '#2ECC71' : '#E74C3C',
              boxShadow: mcpConnected ? '0 0 8px #2ECC71' : mcpConnected === false ? '0 0 8px #E74C3C' : 'none',
              animation: mcpConnected === null ? 'pulse 1.5s infinite' : 'none'
            }} />
            <span style={{ fontSize: '12px' }}>
              {mcpConnected === null ? 'Checking...' : mcpConnected ? 'MCP Connected' : 'MCP Disconnected'}
            </span>
          </div>
        </div>
        <button
          onClick={onClose}
          style={{
            background: 'transparent',
            border: 'none',
            color: '#E74C3C',
            fontSize: '24px',
            cursor: 'pointer',
            padding: '0',
            width: '30px',
            height: '30px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          ×
        </button>
      </h2>
      
      <style>{`
        @keyframes pulse {
          0% { opacity: 1; }
          50% { opacity: 0.5; }
          100% { opacity: 1; }
        }
      `}</style>

      {error && (
        <div style={{
          backgroundColor: 'rgba(231, 76, 60, 0.2)',
          border: '1px solid #E74C3C',
          borderRadius: '4px',
          padding: '10px',
          marginBottom: '15px',
          color: '#E74C3C',
        }}>
          {error}
        </div>
      )}
      
      {mcpConnected === false && (
        <div style={{
          backgroundColor: 'rgba(243, 156, 18, 0.2)',
          border: '1px solid #F39C12',
          borderRadius: '4px',
          padding: '10px',
          marginBottom: '15px',
          color: '#F39C12',
        }}>
          ⚠️ MCP service is not connected. The multi-agent system may not initialize properly.
        </div>
      )}

      <div style={{ marginBottom: '20px' }}>
        <label style={{ display: 'block', marginBottom: '8px', color: '#F1C40F' }}>
          Topology
        </label>
        <select
          value={topology}
          onChange={(e) => setTopology(e.target.value as any)}
          style={{
            width: '100%',
            padding: '8px',
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
            border: '1px solid rgba(255, 255, 255, 0.3)',
            borderRadius: '4px',
            color: 'white',
          }}
        >
          <option value="mesh" style={{ backgroundColor: '#1a1a1a', color: 'white' }}>
            Mesh - Fully connected, best for collaboration
          </option>
          <option value="hierarchical" style={{ backgroundColor: '#1a1a1a', color: 'white' }}>
            Hierarchical - Structured with clear command chain
          </option>
          <option value="ring" style={{ backgroundColor: '#1a1a1a', color: 'white' }}>
            Ring - Sequential processing pipeline
          </option>
          <option value="star" style={{ backgroundColor: '#1a1a1a', color: 'white' }}>
            Star - Central coordinator with workers
          </option>
        </select>
      </div>

      <div style={{ marginBottom: '20px' }}>
        <label style={{ display: 'block', marginBottom: '8px', color: '#F1C40F' }}>
          Maximum Agents: {maxAgents}
        </label>
        <input
          type="range"
          min="3"
          max="20"
          value={maxAgents}
          onChange={(e) => setMaxAgents(Number(e.target.value))}
          style={{ width: '100%' }}
        />
      </div>

      <div style={{ marginBottom: '20px' }}>
        <label style={{ display: 'block', marginBottom: '8px', color: '#F1C40F' }}>
          Agent Types
        </label>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(2, 1fr)',
          gap: '8px',
          fontSize: '14px',
        }}>
          {Object.entries(agentTypes).map(([type, enabled]) => {
            const agentTypeInfo: Record<string, { icon: string; description: string }> = {
              queen: { icon: '👑', description: 'Hive mind leader' },
              coordinator: { icon: '🎯', description: 'Task orchestration' },
              researcher: { icon: '🔍', description: 'Information gathering' },
              coder: { icon: '💻', description: 'Code implementation' },
              analyst: { icon: '📊', description: 'Data analysis' },
              tester: { icon: '🧪', description: 'Quality assurance' },
              architect: { icon: '🏗️', description: 'System design' },
              optimizer: { icon: '⚡', description: 'Performance tuning' },
              reviewer: { icon: '👁️', description: 'Code review' },
              documenter: { icon: '📝', description: 'Documentation' },
              monitor: { icon: '📡', description: 'System monitoring' },
              specialist: { icon: '🔧', description: 'Specialized tasks' },
            };

            const info = agentTypeInfo[type] || { icon: '🤖', description: type };

            return (
              <label key={type} style={{
                display: 'flex',
                alignItems: 'center',
                cursor: 'pointer',
                color: enabled ? '#F1C40F' : 'rgba(255, 255, 255, 0.5)',
                padding: '4px',
                borderRadius: '4px',
                backgroundColor: enabled ? 'rgba(241, 196, 15, 0.1)' : 'transparent',
                transition: 'all 0.2s ease',
              }}
              title={info.description}
              >
                <input
                  type="checkbox"
                  checked={enabled}
                  onChange={(e) => setAgentTypes(prev => ({
                    ...prev,
                    [type]: e.target.checked
                  }))}
                  style={{ marginRight: '8px' }}
                />
                <span style={{ marginRight: '4px' }}>{info.icon}</span>
                {type.charAt(0).toUpperCase() + type.slice(1)}
              </label>
            );
          })}
        </div>
        <div style={{
          fontSize: '12px',
          color: 'rgba(255, 255, 255, 0.6)',
          marginTop: '8px',
        }}>
          {topology === 'hierarchical' && !agentTypes.queen && (
            <span style={{ color: '#F39C12' }}>
              💡 Tip: Enable Queen agent for hierarchical topology
            </span>
          )}
        </div>
      </div>

      <div style={{ marginBottom: '20px' }}>
        <label style={{
          display: 'flex',
          alignItems: 'center',
          marginBottom: '8px',
          color: '#F1C40F',
          cursor: 'pointer',
        }}>
          <input
            type="checkbox"
            checked={enableNeural}
            onChange={(e) => setEnableNeural(e.target.checked)}
            style={{ marginRight: '8px' }}
          />
          Enable Neural Enhancements
        </label>
        <div style={{
          fontSize: '12px',
          color: 'rgba(255, 255, 255, 0.6)',
          marginLeft: '24px',
        }}>
          Activates WASM-accelerated neural networks for collective intelligence
        </div>
      </div>

      <div style={{ marginBottom: '20px' }}>
        <label style={{ display: 'block', marginBottom: '8px', color: '#F1C40F' }}>
          Task for Hive Mind <span style={{ color: '#E74C3C' }}>*</span>
        </label>
        <textarea
          value={customPrompt}
          onChange={(e) => setCustomPrompt(e.target.value)}
          placeholder="Describe the task for the hive mind to accomplish..."
          required
          style={{
            width: '100%',
            minHeight: '100px',
            padding: '8px',
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
            border: '1px solid rgba(255, 255, 255, 0.3)',
            borderRadius: '4px',
            color: 'white',
            resize: 'vertical',
          }}
        />
        <div style={{
          fontSize: '12px',
          color: 'rgba(255, 255, 255, 0.6)',
          marginTop: '4px',
        }}>
          Example: "Build a REST API with user authentication and database integration"
        </div>
      </div>

      <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
        <button
          onClick={onClose}
          disabled={isLoading}
          style={{
            padding: '10px 20px',
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
            border: '1px solid rgba(255, 255, 255, 0.3)',
            borderRadius: '4px',
            color: 'white',
            cursor: isLoading ? 'not-allowed' : 'pointer',
            opacity: isLoading ? 0.5 : 1,
          }}
        >
          Cancel
        </button>
        <button
          onClick={handleInitialize}
          disabled={isLoading}
          style={{
            padding: '10px 20px',
            backgroundColor: '#F1C40F',
            border: 'none',
            borderRadius: '4px',
            color: 'black',
            fontWeight: 'bold',
            cursor: isLoading ? 'not-allowed' : 'pointer',
            opacity: isLoading ? 0.8 : 1,
          }}
        >
          {isLoading ? 'Spawning...' : 'Spawn Hive Mind'}
        </button>
      </div>
    </div>,
    portalContainer
  );
};