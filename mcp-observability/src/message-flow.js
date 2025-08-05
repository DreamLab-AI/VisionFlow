import { v4 as uuidv4 } from 'uuid';
import { createLogger } from './logger.js';

const logger = createLogger('MessageFlowTracker');

export class MessageFlowTracker {
  constructor() {
    this.messages = [];
    this.messageHistory = new Map(); // agentId -> recent messages
    this.flowPatterns = new Map(); // pattern analysis
    this.maxHistorySize = 1000;
    this.patternWindow = 300; // 5 minutes
  }
  
  // Track a new message
  trackMessage({ from, to, type = 'coordination', priority = 1, content = {} }) {
    const message = {
      id: uuidv4(),
      timestamp: new Date(),
      from_agent: from,
      to_agent: Array.isArray(to) ? to : [to],
      message_type: type,
      priority,
      content,
      latency_ms: 0, // Will be updated when acknowledged
      acknowledged: false,
      springForce: this.calculateSpringForce(priority, type)
    };
    
    this.messages.push(message);
    
    // Update agent history
    this.updateAgentHistory(from, message);
    this.to_agent.forEach(agent => this.updateAgentHistory(agent, message));
    
    // Analyze patterns
    this.analyzeFlowPatterns(message);
    
    // Cleanup old messages
    if (this.messages.length > this.maxHistorySize) {
      this.messages = this.messages.slice(-this.maxHistorySize);
    }
    
    logger.debug(`Tracked message: ${from} -> ${to} (${type})`);
    
    return message;
  }
  
  // Acknowledge message receipt
  acknowledgeMessage(messageId, latency = null) {
    const message = this.messages.find(m => m.id === messageId);
    if (message) {
      message.acknowledged = true;
      message.latency_ms = latency || (Date.now() - message.timestamp.getTime());
      logger.debug(`Message ${messageId} acknowledged: ${message.latency_ms}ms`);
    }
  }
  
  // Update agent message history
  updateAgentHistory(agentId, message) {
    if (!this.messageHistory.has(agentId)) {
      this.messageHistory.set(agentId, []);
    }
    
    const history = this.messageHistory.get(agentId);
    history.push({
      messageId: message.id,
      timestamp: message.timestamp,
      direction: message.from_agent === agentId ? 'sent' : 'received',
      type: message.message_type
    });
    
    // Keep only recent history
    if (history.length > 100) {
      history.shift();
    }
  }
  
  // Analyze message flow patterns
  analyzeFlowPatterns(message) {
    const patternKey = `${message.from_agent}-${message.message_type}`;
    
    if (!this.flowPatterns.has(patternKey)) {
      this.flowPatterns.set(patternKey, {
        count: 0,
        avgLatency: 0,
        destinations: new Map()
      });
    }
    
    const pattern = this.flowPatterns.get(patternKey);
    pattern.count++;
    
    // Update destination frequency
    message.to_agent.forEach(dest => {
      const destCount = pattern.destinations.get(dest) || 0;
      pattern.destinations.set(dest, destCount + 1);
    });
  }
  
  // Calculate spring force based on message properties
  calculateSpringForce(priority, type) {
    const typeWeights = {
      coordination: 1.2,
      task_assignment: 1.5,
      status_update: 0.8,
      error_report: 2.0,
      data_transfer: 1.0
    };
    
    const weight = typeWeights[type] || 1.0;
    return priority * weight * 0.1; // Scale to physics units
  }
  
  // Get message flow for time window
  getMessageFlow(timeWindowSeconds = 300, agentFilter = null) {
    const cutoff = new Date(Date.now() - timeWindowSeconds * 1000);
    
    let messages = this.messages.filter(m => m.timestamp > cutoff);
    
    if (agentFilter && agentFilter.length > 0) {
      messages = messages.filter(m => 
        agentFilter.includes(m.from_agent) || 
        m.to_agent.some(to => agentFilter.includes(to))
      );
    }
    
    return messages.map(m => ({
      id: m.id,
      timestamp: m.timestamp.toISOString(),
      from_agent: m.from_agent,
      to_agent: m.to_agent,
      message_type: m.message_type,
      priority: m.priority,
      latency_ms: m.latency_ms,
      acknowledged: m.acknowledged,
      springForce: m.springForce
    }));
  }
  
  // Get communication statistics
  getCommunicationStats(agentId = null) {
    const relevantMessages = agentId 
      ? this.messages.filter(m => 
          m.from_agent === agentId || m.to_agent.includes(agentId))
      : this.messages;
    
    const sentMessages = relevantMessages.filter(m => 
      !agentId || m.from_agent === agentId);
    const receivedMessages = relevantMessages.filter(m => 
      !agentId || m.to_agent.includes(agentId));
    
    const avgLatency = relevantMessages
      .filter(m => m.acknowledged)
      .reduce((sum, m) => sum + m.latency_ms, 0) / 
      (relevantMessages.filter(m => m.acknowledged).length || 1);
    
    return {
      totalMessages: relevantMessages.length,
      sentCount: sentMessages.length,
      receivedCount: receivedMessages.length,
      avgLatency: Math.round(avgLatency),
      messageTypes: this.getMessageTypeDistribution(relevantMessages),
      peakHour: this.findPeakHour(relevantMessages),
      topCommunicators: this.getTopCommunicators(relevantMessages)
    };
  }
  
  // Get message type distribution
  getMessageTypeDistribution(messages) {
    const distribution = {};
    
    messages.forEach(m => {
      distribution[m.message_type] = (distribution[m.message_type] || 0) + 1;
    });
    
    return distribution;
  }
  
  // Find peak communication hour
  findPeakHour(messages) {
    const hourCounts = {};
    
    messages.forEach(m => {
      const hour = m.timestamp.getHours();
      hourCounts[hour] = (hourCounts[hour] || 0) + 1;
    });
    
    let peakHour = 0;
    let peakCount = 0;
    
    Object.entries(hourCounts).forEach(([hour, count]) => {
      if (count > peakCount) {
        peakCount = count;
        peakHour = parseInt(hour);
      }
    });
    
    return { hour: peakHour, count: peakCount };
  }
  
  // Get top communicating agent pairs
  getTopCommunicators(messages) {
    const pairCounts = {};
    
    messages.forEach(m => {
      m.to_agent.forEach(to => {
        const pair = [m.from_agent, to].sort().join('-');
        pairCounts[pair] = (pairCounts[pair] || 0) + 1;
      });
    });
    
    return Object.entries(pairCounts)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5)
      .map(([pair, count]) => {
        const [agent1, agent2] = pair.split('-');
        return { agent1, agent2, messageCount: count };
      });
  }
  
  // Detect communication bottlenecks
  detectBottlenecks() {
    const agentLoad = new Map();
    
    // Calculate message load per agent
    this.messages.forEach(m => {
      // Outgoing messages
      const outgoing = agentLoad.get(m.from_agent) || { sent: 0, received: 0 };
      outgoing.sent++;
      agentLoad.set(m.from_agent, outgoing);
      
      // Incoming messages
      m.to_agent.forEach(to => {
        const incoming = agentLoad.get(to) || { sent: 0, received: 0 };
        incoming.received++;
        agentLoad.set(to, incoming);
      });
    });
    
    // Find agents with high load
    const bottlenecks = [];
    const avgLoad = Array.from(agentLoad.values())
      .reduce((sum, load) => sum + load.sent + load.received, 0) / 
      (agentLoad.size * 2);
    
    agentLoad.forEach((load, agentId) => {
      const totalLoad = load.sent + load.received;
      if (totalLoad > avgLoad * 2) {
        bottlenecks.push({
          agentId,
          load: totalLoad,
          sent: load.sent,
          received: load.received,
          severity: totalLoad / avgLoad
        });
      }
    });
    
    return bottlenecks.sort((a, b) => b.severity - a.severity);
  }
  
  // Get coordination patterns
  getCoordinationPatterns() {
    const patterns = [];
    
    // Detect broadcast patterns
    const broadcasts = this.messages.filter(m => m.to_agent.length > 3);
    if (broadcasts.length > 0) {
      patterns.push({
        type: 'broadcast',
        count: broadcasts.length,
        agents: [...new Set(broadcasts.map(m => m.from_agent))]
      });
    }
    
    // Detect pipeline patterns (sequential messaging)
    const pipelines = this.detectPipelinePatterns();
    if (pipelines.length > 0) {
      patterns.push({
        type: 'pipeline',
        count: pipelines.length,
        chains: pipelines
      });
    }
    
    // Detect hub patterns (central coordinator)
    const hubs = this.detectHubPatterns();
    if (hubs.length > 0) {
      patterns.push({
        type: 'hub',
        count: hubs.length,
        hubs: hubs
      });
    }
    
    return patterns;
  }
  
  // Detect pipeline communication patterns
  detectPipelinePatterns() {
    const pipelines = [];
    const sequenceWindow = 5000; // 5 seconds
    
    // Look for A->B->C patterns
    this.messages.forEach((msg1, idx) => {
      const subsequentMessages = this.messages.slice(idx + 1)
        .filter(msg2 => 
          msg2.timestamp - msg1.timestamp < sequenceWindow &&
          msg1.to_agent.includes(msg2.from_agent)
        );
      
      subsequentMessages.forEach(msg2 => {
        const chain = [msg1.from_agent, msg2.from_agent, ...msg2.to_agent];
        if (chain.length >= 3) {
          pipelines.push({
            agents: chain,
            duration: msg2.timestamp - msg1.timestamp
          });
        }
      });
    });
    
    return pipelines;
  }
  
  // Detect hub communication patterns
  detectHubPatterns() {
    const agentConnections = new Map();
    
    // Count unique connections per agent
    this.messages.forEach(m => {
      // From connections
      const fromConns = agentConnections.get(m.from_agent) || new Set();
      m.to_agent.forEach(to => fromConns.add(to));
      agentConnections.set(m.from_agent, fromConns);
      
      // To connections
      m.to_agent.forEach(to => {
        const toConns = agentConnections.get(to) || new Set();
        toConns.add(m.from_agent);
        agentConnections.set(to, toConns);
      });
    });
    
    // Find agents with many connections
    const hubs = [];
    const avgConnections = Array.from(agentConnections.values())
      .reduce((sum, conns) => sum + conns.size, 0) / agentConnections.size;
    
    agentConnections.forEach((connections, agentId) => {
      if (connections.size > avgConnections * 2) {
        hubs.push({
          agentId,
          connectionCount: connections.size,
          connectedAgents: Array.from(connections)
        });
      }
    });
    
    return hubs.sort((a, b) => b.connectionCount - a.connectionCount);
  }
}