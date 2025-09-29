/**
 * Cognitive Chat - AI chat interface with cognitive agent integration
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useNeural } from '../contexts/NeuralContext';
import '../styles/neural-theme.css';

interface ChatMessage {
  id: string;
  type: 'user' | 'agent' | 'system';
  content: string;
  timestamp: Date;
  agentId?: string;
  agentType?: string;
  cognitivePattern?: string;
  metadata?: {
    confidence?: number;
    processingTime?: number;
    memoryUsed?: number;
    reasoning?: string[];
  };
}

interface CognitiveAgent {
  id: string;
  type: string;
  name: string;
  pattern: string;
  status: 'active' | 'idle' | 'thinking';
  specializations: string[];
  confidence: number;
}

const cognitivePatterns = {
  convergent: { icon: '🎯', color: 'neural-badge-primary', description: 'Focused analysis' },
  divergent: { icon: '💡', color: 'neural-badge-success', description: 'Creative exploration' },
  lateral: { icon: '🔄', color: 'neural-badge-accent', description: 'Lateral thinking' },
  systems: { icon: '🏗️', color: 'neural-badge-warning', description: 'Systems thinking' },
  critical: { icon: '⚖️', color: 'neural-badge-error', description: 'Critical analysis' },
  adaptive: { icon: '🧬', color: 'neural-badge-secondary', description: 'Adaptive reasoning' }
};

const CognitiveChat: React.FC = () => {
  const neural = useNeural();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [autoAssign, setAutoAssign] = useState(true);
  const [chatMode, setChatMode] = useState<'single' | 'swarm' | 'cognitive'>('cognitive');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Mock cognitive agents for demonstration
  const [cognitiveAgents] = useState<CognitiveAgent[]>([
    {
      id: 'cog-1',
      type: 'researcher',
      name: 'Dr. Analyze',
      pattern: 'convergent',
      status: 'active',
      specializations: ['data analysis', 'research', 'fact-checking'],
      confidence: 0.92
    },
    {
      id: 'cog-2',
      type: 'coder',
      name: 'Dev.GPT',
      pattern: 'divergent',
      status: 'active',
      specializations: ['programming', 'architecture', 'debugging'],
      confidence: 0.88
    },
    {
      id: 'cog-3',
      type: 'coordinator',
      name: 'Synth',
      pattern: 'systems',
      status: 'active',
      specializations: ['coordination', 'planning', 'optimization'],
      confidence: 0.95
    }
  ]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    // Initialize with welcome message
    if (messages.length === 0) {
      setMessages([{
        id: 'welcome',
        type: 'system',
        content: 'Welcome to Cognitive Chat! I can connect you with specialized AI agents using different cognitive patterns. How can I assist you today?',
        timestamp: new Date(),
        metadata: {
          confidence: 1.0,
          processingTime: 0
        }
      }]);
    }
  }, [messages.length]);

  const determineOptimalAgent = useCallback((message: string): CognitiveAgent => {
    // Simple keyword-based agent selection
    const keywords = message.toLowerCase();

    if (keywords.includes('code') || keywords.includes('program') || keywords.includes('debug')) {
      return cognitiveAgents.find(a => a.type === 'coder') || cognitiveAgents[0];
    }

    if (keywords.includes('analyze') || keywords.includes('research') || keywords.includes('data')) {
      return cognitiveAgents.find(a => a.type === 'researcher') || cognitiveAgents[0];
    }

    if (keywords.includes('plan') || keywords.includes('coordinate') || keywords.includes('manage')) {
      return cognitiveAgents.find(a => a.type === 'coordinator') || cognitiveAgents[0];
    }

    // Default to highest confidence agent
    return cognitiveAgents.reduce((prev, current) =>
      current.confidence > prev.confidence ? current : prev
    );
  }, [cognitiveAgents]);

  const generateAgentResponse = useCallback(async (message: string, agent: CognitiveAgent): Promise<string> => {
    // Simulate AI processing
    await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));

    const responses = {
      researcher: [
        "Based on my analysis, here's what I found...",
        "Let me research this for you. According to current data...",
        "I've examined the available information and can conclude...",
        "My convergent analysis suggests..."
      ],
      coder: [
        "Here's a creative solution approach...",
        "Let me think outside the box on this...",
        "I see multiple implementation paths we could explore...",
        "Using divergent thinking, I propose..."
      ],
      coordinator: [
        "From a systems perspective, we need to consider...",
        "Let me coordinate the various aspects of this problem...",
        "Taking a holistic view, the optimal approach would be...",
        "Systems thinking tells us..."
      ]
    };

    const agentResponses = responses[agent.type as keyof typeof responses] || responses.researcher;
    const baseResponse = agentResponses[Math.floor(Math.random() * agentResponses.length)];

    return `${baseResponse} "${message}" - I'm processing this using ${agent.pattern} cognitive patterns. My specializations in ${agent.specializations.join(', ')} help me provide focused insights.`;
  }, []);

  const sendMessage = useCallback(async () => {
    if (!input.trim() || isTyping) return;

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      type: 'user',
      content: input.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    try {
      if (chatMode === 'cognitive' || (chatMode === 'single' && autoAssign)) {
        const optimalAgent = selectedAgent
          ? cognitiveAgents.find(a => a.id === selectedAgent) || cognitiveAgents[0]
          : determineOptimalAgent(input);

        // Show agent thinking
        const thinkingMessage: ChatMessage = {
          id: `thinking-${Date.now()}`,
          type: 'system',
          content: `${optimalAgent.name} is thinking using ${optimalAgent.pattern} patterns...`,
          timestamp: new Date(),
          agentId: optimalAgent.id,
          agentType: optimalAgent.type,
          cognitivePattern: optimalAgent.pattern
        };

        setMessages(prev => [...prev, thinkingMessage]);

        const startTime = Date.now();
        const response = await generateAgentResponse(input, optimalAgent);
        const processingTime = Date.now() - startTime;

        // Remove thinking message and add response
        setMessages(prev => {
          const filtered = prev.filter(m => m.id !== thinkingMessage.id);
          return [...filtered, {
            id: `agent-${Date.now()}`,
            type: 'agent',
            content: response,
            timestamp: new Date(),
            agentId: optimalAgent.id,
            agentType: optimalAgent.type,
            cognitivePattern: optimalAgent.pattern,
            metadata: {
              confidence: optimalAgent.confidence,
              processingTime,
              reasoning: ['Pattern-based analysis', 'Specialization match', 'Context adaptation']
            }
          }];
        });
      } else if (chatMode === 'swarm') {
        // Swarm mode - multiple agents respond
        const swarmResponses = await Promise.all(
          cognitiveAgents.slice(0, 2).map(async agent => {
            const response = await generateAgentResponse(input, agent);
            return {
              agent,
              response,
              processingTime: 1000 + Math.random() * 1500
            };
          })
        );

        for (const { agent, response, processingTime } of swarmResponses) {
          setMessages(prev => [...prev, {
            id: `swarm-${agent.id}-${Date.now()}`,
            type: 'agent',
            content: response,
            timestamp: new Date(),
            agentId: agent.id,
            agentType: agent.type,
            cognitivePattern: agent.pattern,
            metadata: {
              confidence: agent.confidence,
              processingTime,
              reasoning: ['Swarm collaboration', 'Parallel processing']
            }
          }]);
          await new Promise(resolve => setTimeout(resolve, 500));
        }
      }

      // Send to neural context
      await neural.sendMessage({
        type: 'user',
        content: input.trim()
      });

    } catch (error) {
      setMessages(prev => [...prev, {
        id: `error-${Date.now()}`,
        type: 'system',
        content: `Error: ${error instanceof Error ? error.message : 'Unknown error occurred'}`,
        timestamp: new Date()
      }]);
    } finally {
      setIsTyping(false);
    }
  }, [input, isTyping, chatMode, selectedAgent, autoAssign, cognitiveAgents, determineOptimalAgent, generateAgentResponse, neural]);

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    neural.clearChat();
  };

  const getAgentAvatar = (agentType?: string, pattern?: string) => {
    if (pattern && cognitivePatterns[pattern as keyof typeof cognitivePatterns]) {
      return cognitivePatterns[pattern as keyof typeof cognitivePatterns].icon;
    }
    return agentType === 'user' ? '👤' : '🤖';
  };

  const getMessageStyle = (message: ChatMessage) => {
    if (message.type === 'user') return 'ml-auto neural-bg-primary';
    if (message.type === 'system') return 'neural-bg-tertiary';
    if (message.cognitivePattern) {
      const pattern = cognitivePatterns[message.cognitivePattern as keyof typeof cognitivePatterns];
      return pattern ? `neural-bg-secondary border-l-4 border-${pattern.color}` : 'neural-bg-secondary';
    }
    return 'neural-bg-secondary';
  };

  return (
    <div className="neural-theme h-full neural-flex neural-flex-col">
      {/* Header */}
      <div className="neural-card-header neural-flex neural-flex-between items-center">
        <div>
          <h2 className="neural-heading neural-heading-md">Cognitive Chat</h2>
          <p className="neural-text-muted">Intelligent conversation with AI agents</p>
        </div>
        <div className="neural-flex items-center gap-4">
          <div className="neural-flex items-center gap-2">
            <label className="neural-text-secondary text-sm">Mode:</label>
            <select
              value={chatMode}
              onChange={(e) => setChatMode(e.target.value as any)}
              className="neural-input text-sm py-1"
            >
              <option value="cognitive">Cognitive</option>
              <option value="single">Single Agent</option>
              <option value="swarm">Swarm</option>
            </select>
          </div>
          <button onClick={clearChat} className="neural-btn neural-btn-ghost neural-btn-sm">
            Clear
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="neural-flex neural-flex-1 overflow-hidden">
        {/* Chat Area */}
        <div className="neural-flex-1 neural-flex neural-flex-col">
          {/* Messages */}
          <div className="neural-flex-1 overflow-auto neural-scrollbar p-4 neural-space-y-4">
            {messages.map(message => (
              <div
                key={message.id}
                className={`neural-flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-2xl neural-card neural-card-body p-4 ${getMessageStyle(message)}`}
                >
                  {/* Message Header */}
                  {message.type !== 'user' && (
                    <div className="neural-flex items-center gap-2 mb-2">
                      <span className="text-lg">{getAgentAvatar(message.agentType, message.cognitivePattern)}</span>
                      <div className="neural-flex items-center gap-2">
                        {message.agentId && (
                          <span className="neural-text-accent font-medium text-sm">
                            {cognitiveAgents.find(a => a.id === message.agentId)?.name || 'Agent'}
                          </span>
                        )}
                        {message.cognitivePattern && (
                          <span className={`neural-badge ${cognitivePatterns[message.cognitivePattern as keyof typeof cognitivePatterns]?.color || 'neural-badge-primary'} text-xs`}>
                            {message.cognitivePattern}
                          </span>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Message Content */}
                  <div className="neural-text-primary mb-2">{message.content}</div>

                  {/* Message Footer */}
                  <div className="neural-flex neural-flex-between items-center">
                    <span className="neural-text-muted text-xs">
                      {message.timestamp.toLocaleTimeString()}
                    </span>
                    {message.metadata && (
                      <div className="neural-flex items-center gap-2">
                        {message.metadata.confidence && (
                          <span className="neural-text-muted text-xs">
                            {Math.round(message.metadata.confidence * 100)}% confidence
                          </span>
                        )}
                        {message.metadata.processingTime && (
                          <span className="neural-text-muted text-xs">
                            {message.metadata.processingTime}ms
                          </span>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Reasoning */}
                  {message.metadata?.reasoning && (
                    <details className="mt-2">
                      <summary className="neural-text-muted text-xs cursor-pointer">
                        Reasoning Process
                      </summary>
                      <ul className="mt-1 neural-text-muted text-xs">
                        {message.metadata.reasoning.map((step, index) => (
                          <li key={index}>• {step}</li>
                        ))}
                      </ul>
                    </details>
                  )}
                </div>
              </div>
            ))}
            {isTyping && (
              <div className="neural-flex justify-start">
                <div className="neural-card neural-card-body p-4 neural-bg-secondary">
                  <div className="neural-flex items-center gap-2">
                    <div className="neural-spinner"></div>
                    <span className="neural-text-muted">AI is thinking...</span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="neural-card-body border-t border-neural-border">
            <div className="neural-flex gap-3">
              <div className="neural-flex-1">
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask me anything... (Enter to send, Shift+Enter for new line)"
                  className="neural-input resize-none"
                  rows={2}
                  disabled={isTyping}
                />
              </div>
              <button
                onClick={sendMessage}
                disabled={!input.trim() || isTyping}
                className="neural-btn neural-btn-primary px-6"
              >
                Send
              </button>
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <div className="w-80 neural-card">
          <div className="neural-card-header">
            <h3 className="neural-heading neural-heading-sm">Cognitive Agents</h3>
          </div>
          <div className="neural-card-body neural-space-y-4">
            {/* Agent Selection */}
            {chatMode === 'single' && (
              <div>
                <div className="neural-flex items-center gap-2 mb-2">
                  <input
                    type="checkbox"
                    checked={autoAssign}
                    onChange={(e) => setAutoAssign(e.target.checked)}
                    className="neural-checkbox"
                  />
                  <label className="neural-text-secondary text-sm">Auto-assign optimal agent</label>
                </div>
                {!autoAssign && (
                  <select
                    value={selectedAgent || ''}
                    onChange={(e) => setSelectedAgent(e.target.value || null)}
                    className="neural-input text-sm"
                  >
                    <option value="">Select agent...</option>
                    {cognitiveAgents.map(agent => (
                      <option key={agent.id} value={agent.id}>
                        {agent.name} ({agent.type})
                      </option>
                    ))}
                  </select>
                )}
              </div>
            )}

            {/* Active Agents */}
            <div>
              <h4 className="neural-text-secondary font-semibold mb-2">Active Agents</h4>
              <div className="neural-space-y-3">
                {cognitiveAgents.map(agent => (
                  <div
                    key={agent.id}
                    className={`neural-card neural-card-body p-3 cursor-pointer transition-all ${
                      selectedAgent === agent.id ? 'neural-glow-primary' : ''
                    }`}
                    onClick={() => setSelectedAgent(agent.id)}
                  >
                    <div className="neural-flex items-center gap-2 mb-2">
                      <span className="text-lg">
                        {cognitivePatterns[agent.pattern as keyof typeof cognitivePatterns]?.icon}
                      </span>
                      <div className="neural-flex-1">
                        <div className="neural-text-primary font-medium text-sm">
                          {agent.name}
                        </div>
                        <div className="neural-text-muted text-xs">
                          {agent.type}
                        </div>
                      </div>
                      <div className={`neural-status neural-status-${agent.status === 'active' ? 'active' : 'pending'}`}>
                        <div className="neural-status-dot"></div>
                      </div>
                    </div>

                    <div className="neural-flex items-center gap-2 mb-2">
                      <span className={`neural-badge ${cognitivePatterns[agent.pattern as keyof typeof cognitivePatterns]?.color} text-xs`}>
                        {agent.pattern}
                      </span>
                      <span className="neural-text-muted text-xs">
                        {Math.round(agent.confidence * 100)}%
                      </span>
                    </div>

                    <div className="text-xs neural-text-muted">
                      {agent.specializations.slice(0, 2).join(', ')}
                      {agent.specializations.length > 2 && '...'}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Cognitive Patterns Guide */}
            <div>
              <h4 className="neural-text-secondary font-semibold mb-2">Cognitive Patterns</h4>
              <div className="neural-space-y-2">
                {Object.entries(cognitivePatterns).map(([pattern, config]) => (
                  <div key={pattern} className="neural-flex items-center gap-2">
                    <span>{config.icon}</span>
                    <div className="neural-flex-1">
                      <div className="neural-text-primary text-sm capitalize">{pattern}</div>
                      <div className="neural-text-muted text-xs">{config.description}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CognitiveChat;