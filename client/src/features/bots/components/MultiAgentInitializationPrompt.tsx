import React, { useState, useEffect, useMemo } from 'react';
import ReactDOM from 'react-dom';
import { createLogger } from '../../../utils/loggerConfig';
import { unifiedApiClient } from '../../../services/api/UnifiedApiClient';
import { botsWebSocketIntegration } from '../services/BotsWebSocketIntegration';
import {
  skillDefinitions,
  categoryLabels,
  categoryIcons,
  SkillDefinition,
} from '../../settings/components/panels/skillDefinitions';

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
  const [selectedSkills, setSelectedSkills] = useState<Set<string>>(new Set());
  const [skillSearchQuery, setSkillSearchQuery] = useState('');
  const [showSkills, setShowSkills] = useState(false);

  // Filter skills based on search
  const filteredSkills = useMemo(() => {
    if (!skillSearchQuery) return skillDefinitions;
    const query = skillSearchQuery.toLowerCase();
    return skillDefinitions.filter(
      (skill) =>
        skill.name.toLowerCase().includes(query) ||
        skill.description.toLowerCase().includes(query) ||
        skill.tags.some((tag) => tag.toLowerCase().includes(query))
    );
  }, [skillSearchQuery]);

  // Group skills by category
  const skillsByCategory = useMemo(() => {
    const categories: Record<string, SkillDefinition[]> = {};
    for (const skill of filteredSkills) {
      if (!categories[skill.category]) {
        categories[skill.category] = [];
      }
      categories[skill.category].push(skill);
    }
    return categories;
  }, [filteredSkills]);

  const toggleSkill = (skillId: string) => {
    setSelectedSkills((prev) => {
      const next = new Set(prev);
      if (next.has(skillId)) {
        next.delete(skillId);
      } else {
        next.add(skillId);
      }
      return next;
    });
  };

  
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const response = await unifiedApiClient.getData('/bots/status');
        // API returns { success: true, data: { connected: true, ... } }
        setMcpConnected(response?.data?.connected ?? response?.connected ?? false);
      } catch (error) {
        setMcpConnected(false);
      }
    };

    
    checkConnection();

    
    const interval = setInterval(checkConnection, 3000);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    
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

      
      const config = {
        topology,
        maxAgents,
        strategy: 'adaptive',
        enableNeural,
        agentTypes: selectedAgentTypes,
        skills: Array.from(selectedSkills),
        customPrompt: customPrompt.trim(),
      };

      logger.info('Spawning hive mind with config:', config);

      
      logger.info('Calling API endpoint: /bots/initialize-swarm');
      const response = await unifiedApiClient.postData('/bots/initialize-swarm', config);

      if (response.success) {
        logger.info('Hive mind spawned successfully:', response);
        
        
        
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
      backgroundColor: '#141414',  
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
          onChange={(e) => setTopology(e.target.value as typeof topology)}
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

      {/* Skills Section */}
      <div style={{ marginBottom: '20px' }}>
        <button
          onClick={() => setShowSkills(!showSkills)}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            width: '100%',
            padding: '10px',
            backgroundColor: selectedSkills.size > 0 ? 'rgba(241, 196, 15, 0.2)' : 'rgba(255, 255, 255, 0.05)',
            border: selectedSkills.size > 0 ? '1px solid #F1C40F' : '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#F1C40F',
            cursor: 'pointer',
            fontSize: '14px',
          }}
        >
          <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            ⚡ Skills
            {selectedSkills.size > 0 && (
              <span style={{
                backgroundColor: '#F1C40F',
                color: 'black',
                padding: '2px 8px',
                borderRadius: '12px',
                fontSize: '12px',
                fontWeight: 'bold',
              }}>
                {selectedSkills.size} selected
              </span>
            )}
          </span>
          <span>{showSkills ? '▼' : '▶'}</span>
        </button>

        {showSkills && (
          <div style={{
            marginTop: '10px',
            padding: '10px',
            backgroundColor: 'rgba(0, 0, 0, 0.3)',
            borderRadius: '4px',
            maxHeight: '300px',
            overflow: 'auto',
          }}>
            {/* Search */}
            <input
              type="text"
              value={skillSearchQuery}
              onChange={(e) => setSkillSearchQuery(e.target.value)}
              placeholder="Search skills..."
              style={{
                width: '100%',
                padding: '8px',
                marginBottom: '10px',
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                border: '1px solid rgba(255, 255, 255, 0.3)',
                borderRadius: '4px',
                color: 'white',
              }}
            />

            {/* Skills by category */}
            {Object.entries(skillsByCategory).map(([category, skills]) => (
              <div key={category} style={{ marginBottom: '10px' }}>
                <div style={{
                  fontSize: '12px',
                  color: '#F1C40F',
                  marginBottom: '6px',
                  fontWeight: 'bold',
                }}>
                  {categoryIcons[category]} {categoryLabels[category]}
                </div>
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(2, 1fr)',
                  gap: '4px',
                }}>
                  {skills.map((skill) => (
                    <label
                      key={skill.id}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        padding: '4px 6px',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '12px',
                        backgroundColor: selectedSkills.has(skill.id)
                          ? 'rgba(241, 196, 15, 0.2)'
                          : 'transparent',
                        color: selectedSkills.has(skill.id)
                          ? '#F1C40F'
                          : 'rgba(255, 255, 255, 0.7)',
                      }}
                      title={skill.description}
                    >
                      <input
                        type="checkbox"
                        checked={selectedSkills.has(skill.id)}
                        onChange={() => toggleSkill(skill.id)}
                        style={{ marginRight: '2px' }}
                      />
                      <span>{skill.icon}</span>
                      <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {skill.name}
                      </span>
                      {skill.mcpServer && (
                        <span style={{
                          fontSize: '9px',
                          padding: '1px 4px',
                          backgroundColor: 'rgba(46, 204, 113, 0.3)',
                          color: '#2ECC71',
                          borderRadius: '3px',
                        }}>
                          MCP
                        </span>
                      )}
                    </label>
                  ))}
                </div>
              </div>
            ))}

            {filteredSkills.length === 0 && (
              <div style={{ textAlign: 'center', color: 'rgba(255, 255, 255, 0.5)', padding: '20px' }}>
                No skills match your search
              </div>
            )}
          </div>
        )}
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