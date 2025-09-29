/**
 * Neural Command Center - Main interface for neural AI capabilities
 */

import React, { useState, useEffect, useRef } from 'react';
import { useNeural } from '../contexts/NeuralContext';
import '../styles/neural-theme.css';

interface CommandSuggestion {
  command: string;
  description: string;
  category: 'swarm' | 'agent' | 'workflow' | 'neural' | 'analysis';
  args?: string[];
}

const commandSuggestions: CommandSuggestion[] = [
  // Swarm commands
  { command: 'swarm init', description: 'Initialize a new swarm with specified topology', category: 'swarm', args: ['topology'] },
  { command: 'swarm status', description: 'Get current swarm status and metrics', category: 'swarm' },
  { command: 'swarm scale', description: 'Scale swarm to target number of agents', category: 'swarm', args: ['count'] },
  { command: 'swarm destroy', description: 'Destroy current swarm and cleanup', category: 'swarm' },

  // Agent commands
  { command: 'agent spawn', description: 'Spawn a new agent with specified type', category: 'agent', args: ['type'] },
  { command: 'agent list', description: 'List all active agents', category: 'agent' },
  { command: 'agent inspect', description: 'Get detailed agent information', category: 'agent', args: ['id'] },
  { command: 'agent remove', description: 'Remove an agent from the swarm', category: 'agent', args: ['id'] },

  // Workflow commands
  { command: 'workflow create', description: 'Create a new workflow from template', category: 'workflow', args: ['name'] },
  { command: 'workflow list', description: 'List all workflows', category: 'workflow' },
  { command: 'workflow execute', description: 'Execute a workflow', category: 'workflow', args: ['id'] },
  { command: 'workflow stop', description: 'Stop a running workflow', category: 'workflow', args: ['id'] },

  // Neural commands
  { command: 'neural train', description: 'Train a neural network model', category: 'neural', args: ['config'] },
  { command: 'neural predict', description: 'Run inference on a model', category: 'neural', args: ['model', 'input'] },
  { command: 'neural models', description: 'List available neural models', category: 'neural' },
  { command: 'neural patterns', description: 'Analyze cognitive patterns', category: 'neural' },

  // Analysis commands
  { command: 'analyze performance', description: 'Analyze swarm performance metrics', category: 'analysis' },
  { command: 'analyze bottlenecks', description: 'Identify system bottlenecks', category: 'analysis' },
  { command: 'analyze patterns', description: 'Analyze cognitive and behavioral patterns', category: 'analysis' },
  { command: 'benchmark run', description: 'Run performance benchmarks', category: 'analysis' }
];

const NeuralCommand: React.FC = () => {
  const neural = useNeural();
  const [command, setCommand] = useState('');
  const [output, setOutput] = useState<string[]>([]);
  const [suggestions, setSuggestions] = useState<CommandSuggestion[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedSuggestion, setSelectedSuggestion] = useState(0);
  const [isExecuting, setIsExecuting] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const outputRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [output]);

  useEffect(() => {
    if (command.trim()) {
      const filtered = commandSuggestions.filter(cmd =>
        cmd.command.toLowerCase().includes(command.toLowerCase()) ||
        cmd.description.toLowerCase().includes(command.toLowerCase())
      );
      setSuggestions(filtered.slice(0, 8));
      setShowSuggestions(filtered.length > 0);
      setSelectedSuggestion(0);
    } else {
      setShowSuggestions(false);
    }
  }, [command]);

  const executeCommand = async (cmd: string) => {
    setIsExecuting(true);
    const timestamp = new Date().toLocaleTimeString();
    setOutput(prev => [...prev, `[${timestamp}] > ${cmd}`]);

    try {
      const result = await neural.executeCommand(cmd);
      setOutput(prev => [...prev, formatCommandResult(result)]);
    } catch (error) {
      setOutput(prev => [...prev, `Error: ${error instanceof Error ? error.message : 'Unknown error'}`]);
    } finally {
      setIsExecuting(false);
    }
  };

  const formatCommandResult = (result: any): string => {
    if (typeof result === 'string') return result;
    if (typeof result === 'object') {
      return JSON.stringify(result, null, 2);
    }
    return String(result);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (command.trim() && !isExecuting) {
      executeCommand(command.trim());
      setCommand('');
      setShowSuggestions(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (showSuggestions) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedSuggestion(prev => Math.min(prev + 1, suggestions.length - 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedSuggestion(prev => Math.max(prev - 1, 0));
      } else if (e.key === 'Tab') {
        e.preventDefault();
        if (suggestions[selectedSuggestion]) {
          setCommand(suggestions[selectedSuggestion].command);
          setShowSuggestions(false);
        }
      }
    }

    if (e.key === 'Escape') {
      setShowSuggestions(false);
    }
  };

  const selectSuggestion = (suggestion: CommandSuggestion) => {
    setCommand(suggestion.command);
    setShowSuggestions(false);
    inputRef.current?.focus();
  };

  const clearOutput = () => {
    setOutput([]);
  };

  const getCategoryColor = (category: CommandSuggestion['category']) => {
    const colors = {
      swarm: 'neural-badge-primary',
      agent: 'neural-badge-success',
      workflow: 'neural-badge-secondary',
      neural: 'neural-badge-accent',
      analysis: 'neural-badge-warning'
    };
    return colors[category] || 'neural-badge-primary';
  };

  return (
    <div className="neural-theme h-screen flex flex-col">
      {/* Header */}
      <div className="neural-card-header flex justify-between items-center">
        <div>
          <h1 className="neural-heading neural-heading-lg">Neural Command Center</h1>
          <p className="neural-text-muted">Advanced AI swarm orchestration and control</p>
        </div>
        <div className="neural-flex neural-flex-center gap-4">
          <div className={`neural-status ${neural.isConnected ? 'neural-status-active' : 'neural-status-error'}`}>
            <div className="neural-status-dot"></div>
            <span className="text-sm">
              {neural.isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          <div className="neural-badge neural-badge-primary">
            {neural.agents.length} Agents
          </div>
          <div className="neural-badge neural-badge-success">
            {neural.workflows.filter(w => w.status === 'running').length} Active
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 neural-flex neural-flex-col p-6 gap-6">
        {/* Quick Stats */}
        <div className="neural-grid neural-grid-4 gap-4">
          <div className="neural-card neural-card-body p-4">
            <div className="neural-flex neural-flex-between items-center">
              <div>
                <p className="neural-text-muted text-sm">Total Tasks</p>
                <p className="neural-heading neural-heading-md">
                  {neural.agents.reduce((sum, agent) => sum + agent.performance.tasksCompleted, 0)}
                </p>
              </div>
              <div className="neural-badge neural-badge-primary">
                +{neural.agents.filter(a => a.status === 'busy').length}
              </div>
            </div>
          </div>

          <div className="neural-card neural-card-body p-4">
            <div className="neural-flex neural-flex-between items-center">
              <div>
                <p className="neural-text-muted text-sm">Success Rate</p>
                <p className="neural-heading neural-heading-md">
                  {neural.agents.length > 0
                    ? Math.round(neural.agents.reduce((sum, agent) => sum + agent.performance.successRate, 0) / neural.agents.length * 100)
                    : 0}%
                </p>
              </div>
              <div className="neural-badge neural-badge-success">
                Active
              </div>
            </div>
          </div>

          <div className="neural-card neural-card-body p-4">
            <div className="neural-flex neural-flex-between items-center">
              <div>
                <p className="neural-text-muted text-sm">Avg Response</p>
                <p className="neural-heading neural-heading-md">
                  {neural.agents.length > 0
                    ? Math.round(neural.agents.reduce((sum, agent) => sum + agent.performance.averageResponseTime, 0) / neural.agents.length)
                    : 0}ms
                </p>
              </div>
              <div className="neural-badge neural-badge-accent">
                Fast
              </div>
            </div>
          </div>

          <div className="neural-card neural-card-body p-4">
            <div className="neural-flex neural-flex-between items-center">
              <div>
                <p className="neural-text-muted text-sm">Memory Usage</p>
                <p className="neural-heading neural-heading-md">
                  {neural.memory.length}
                </p>
              </div>
              <div className="neural-badge neural-badge-warning">
                Growing
              </div>
            </div>
          </div>
        </div>

        {/* Command Interface */}
        <div className="neural-flex neural-flex-1 gap-6">
          {/* Command Input */}
          <div className="flex-1 neural-flex neural-flex-col">
            <div className="neural-card neural-flex-1 neural-flex neural-flex-col">
              <div className="neural-card-header neural-flex neural-flex-between items-center">
                <h2 className="neural-heading neural-heading-sm">Command Terminal</h2>
                <button
                  onClick={clearOutput}
                  className="neural-btn neural-btn-ghost neural-btn-sm"
                >
                  Clear
                </button>
              </div>

              <div className="neural-card-body neural-flex-1 neural-flex neural-flex-col">
                {/* Output Area */}
                <div
                  ref={outputRef}
                  className="flex-1 neural-bg-tertiary p-4 rounded-lg mb-4 neural-scrollbar overflow-auto font-mono text-sm"
                  style={{ minHeight: '300px', maxHeight: '500px' }}
                >
                  {output.length === 0 ? (
                    <div className="neural-text-muted">
                      Welcome to Neural Command Center. Type a command to get started.
                      <br />
                      Try: <span className="neural-text-accent">swarm status</span> or <span className="neural-text-accent">help</span>
                    </div>
                  ) : (
                    output.map((line, index) => (
                      <div key={index} className="mb-1">
                        {line}
                      </div>
                    ))
                  )}
                  {isExecuting && (
                    <div className="neural-flex items-center gap-2 neural-text-accent">
                      <div className="neural-spinner"></div>
                      Executing...
                    </div>
                  )}
                </div>

                {/* Command Input */}
                <form onSubmit={handleSubmit} className="relative">
                  <div className="neural-flex gap-2">
                    <div className="flex-1 relative">
                      <input
                        ref={inputRef}
                        type="text"
                        value={command}
                        onChange={(e) => setCommand(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Enter neural command..."
                        className="neural-input font-mono"
                        disabled={isExecuting}
                      />

                      {/* Suggestions Dropdown */}
                      {showSuggestions && (
                        <div className="absolute top-full left-0 right-0 mt-1 neural-card neural-card-body p-2 z-50 max-h-64 overflow-auto">
                          {suggestions.map((suggestion, index) => (
                            <div
                              key={suggestion.command}
                              onClick={() => selectSuggestion(suggestion)}
                              className={`p-2 rounded cursor-pointer transition-colors ${
                                index === selectedSuggestion
                                  ? 'neural-bg-primary'
                                  : 'hover:neural-bg-tertiary'
                              }`}
                            >
                              <div className="neural-flex neural-flex-between items-start">
                                <div className="flex-1">
                                  <div className="neural-flex items-center gap-2">
                                    <code className="neural-text-accent font-mono text-sm">
                                      {suggestion.command}
                                    </code>
                                    <span className={`neural-badge ${getCategoryColor(suggestion.category)} text-xs`}>
                                      {suggestion.category}
                                    </span>
                                  </div>
                                  <p className="neural-text-muted text-sm mt-1">
                                    {suggestion.description}
                                  </p>
                                  {suggestion.args && (
                                    <div className="mt-1">
                                      <span className="neural-text-muted text-xs">Args: </span>
                                      {suggestion.args.map(arg => (
                                        <code key={arg} className="neural-text-accent text-xs mr-2">
                                          &lt;{arg}&gt;
                                        </code>
                                      ))}
                                    </div>
                                  )}
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                    <button
                      type="submit"
                      disabled={!command.trim() || isExecuting}
                      className="neural-btn neural-btn-primary px-6"
                    >
                      {isExecuting ? 'Executing...' : 'Execute'}
                    </button>
                  </div>
                </form>
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="w-80">
            <div className="neural-card">
              <div className="neural-card-header">
                <h3 className="neural-heading neural-heading-sm">Quick Actions</h3>
              </div>
              <div className="neural-card-body neural-space-y-4">
                <div>
                  <h4 className="neural-text-secondary font-semibold mb-2">Swarm Management</h4>
                  <div className="neural-space-y-2">
                    <button
                      onClick={() => setCommand('swarm status')}
                      className="w-full neural-btn neural-btn-outline text-left"
                    >
                      Check Status
                    </button>
                    <button
                      onClick={() => setCommand('swarm init mesh')}
                      className="w-full neural-btn neural-btn-outline text-left"
                    >
                      Initialize Mesh
                    </button>
                    <button
                      onClick={() => setCommand('agent spawn researcher')}
                      className="w-full neural-btn neural-btn-outline text-left"
                    >
                      Spawn Agent
                    </button>
                  </div>
                </div>

                <div>
                  <h4 className="neural-text-secondary font-semibold mb-2">Analysis</h4>
                  <div className="neural-space-y-2">
                    <button
                      onClick={() => setCommand('analyze performance')}
                      className="w-full neural-btn neural-btn-outline text-left"
                    >
                      Performance
                    </button>
                    <button
                      onClick={() => setCommand('analyze patterns')}
                      className="w-full neural-btn neural-btn-outline text-left"
                    >
                      Patterns
                    </button>
                    <button
                      onClick={() => setCommand('benchmark run')}
                      className="w-full neural-btn neural-btn-outline text-left"
                    >
                      Benchmark
                    </button>
                  </div>
                </div>

                <div>
                  <h4 className="neural-text-secondary font-semibold mb-2">Command History</h4>
                  <div className="neural-bg-tertiary rounded p-3 max-h-32 overflow-auto neural-scrollbar">
                    {neural.commandHistory.slice(-5).map((cmd, index) => (
                      <div
                        key={index}
                        onClick={() => setCommand(cmd)}
                        className="text-sm neural-text-muted cursor-pointer hover:neural-text-accent p-1 rounded hover:neural-bg-surface transition-colors"
                      >
                        {cmd}
                      </div>
                    ))}
                    {neural.commandHistory.length === 0 && (
                      <div className="neural-text-muted text-sm">No recent commands</div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NeuralCommand;