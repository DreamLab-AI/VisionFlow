import { createLogger } from './logger.js';

const logger = createLogger('PerformanceMonitor');

export class PerformanceMonitor {
  constructor() {
    this.metrics = {
      systemMetrics: {
        timestamp: new Date(),
        activeAgents: 0,
        totalAgents: 0,
        messageRate: 0,
        avgLatency: 0,
        errorRate: 0,
        networkHealth: 1.0,
        cpuUsage: 0,
        memoryUsage: 0,
        gpuUsage: null
      },
      agentMetrics: new Map(),
      swarmMetrics: new Map(),
      historicalData: []
    };
    
    this.collectionInterval = 1000; // 1 second
    this.historyLimit = 3600; // 1 hour of data
    this.lastCollectionTime = Date.now();
  }
  
  // Collect performance metrics
  collect() {
    const now = Date.now();
    const deltaTime = (now - this.lastCollectionTime) / 1000;
    this.lastCollectionTime = now;
    
    // Update system metrics
    this.updateSystemMetrics();
    
    // Store historical data
    this.metrics.historicalData.push({
      timestamp: new Date(),
      ...this.metrics.systemMetrics
    });
    
    // Cleanup old data
    if (this.metrics.historicalData.length > this.historyLimit) {
      this.metrics.historicalData.shift();
    }
    
    logger.debug(`Collected metrics: ${this.metrics.systemMetrics.activeAgents} active agents`);
  }
  
  // Update system-wide metrics
  updateSystemMetrics() {
    const metrics = this.metrics.systemMetrics;
    
    // Get process metrics
    const memUsage = process.memoryUsage();
    metrics.memoryUsage = memUsage.heapUsed / (1024 * 1024); // MB
    
    // CPU usage (simplified)
    const cpuUsage = process.cpuUsage();
    metrics.cpuUsage = (cpuUsage.user + cpuUsage.system) / 1000000; // Convert to seconds
    
    // Calculate message rate from recent history
    if (this.metrics.historicalData.length > 0) {
      const recentMetrics = this.metrics.historicalData.slice(-60); // Last minute
      const totalMessages = recentMetrics.reduce((sum, m) => sum + (m.messageCount || 0), 0);
      metrics.messageRate = totalMessages / 60; // Messages per second
    }
    
    // Network health based on error rate
    metrics.networkHealth = Math.max(0, 1 - metrics.errorRate);
    
    metrics.timestamp = new Date();
  }
  
  // Update agent-specific metrics
  updateAgentMetrics(agentId, metrics) {
    const agentMetrics = this.metrics.agentMetrics.get(agentId) || {
      performanceHistory: [],
      averages: {}
    };
    
    agentMetrics.performanceHistory.push({
      timestamp: new Date(),
      ...metrics
    });
    
    // Keep only recent history
    if (agentMetrics.performanceHistory.length > 300) { // 5 minutes
      agentMetrics.performanceHistory.shift();
    }
    
    // Calculate averages
    agentMetrics.averages = this.calculateAverages(agentMetrics.performanceHistory);
    
    this.metrics.agentMetrics.set(agentId, agentMetrics);
  }
  
  // Update swarm-specific metrics
  updateSwarmMetrics(swarmId, metrics) {
    const swarmMetrics = this.metrics.swarmMetrics.get(swarmId) || {
      history: [],
      trends: {}
    };
    
    swarmMetrics.history.push({
      timestamp: new Date(),
      ...metrics
    });
    
    // Keep only recent history
    if (swarmMetrics.history.length > 300) {
      swarmMetrics.history.shift();
    }
    
    // Calculate trends
    swarmMetrics.trends = this.calculateTrends(swarmMetrics.history);
    
    this.metrics.swarmMetrics.set(swarmId, swarmMetrics);
  }
  
  // Calculate averages from historical data
  calculateAverages(history) {
    if (history.length === 0) return {};
    
    const sums = {};
    const counts = {};
    
    history.forEach(data => {
      Object.entries(data).forEach(([key, value]) => {
        if (typeof value === 'number') {
          sums[key] = (sums[key] || 0) + value;
          counts[key] = (counts[key] || 0) + 1;
        }
      });
    });
    
    const averages = {};
    Object.keys(sums).forEach(key => {
      averages[key] = sums[key] / counts[key];
    });
    
    return averages;
  }
  
  // Calculate trends from historical data
  calculateTrends(history) {
    if (history.length < 2) return {};
    
    const trends = {};
    const recent = history.slice(-10);
    const older = history.slice(-20, -10);
    
    if (older.length === 0) return {};
    
    const recentAvg = this.calculateAverages(recent);
    const olderAvg = this.calculateAverages(older);
    
    Object.keys(recentAvg).forEach(key => {
      if (olderAvg[key]) {
        const change = (recentAvg[key] - olderAvg[key]) / olderAvg[key];
        trends[key] = {
          direction: change > 0 ? 'up' : change < 0 ? 'down' : 'stable',
          changePercent: Math.abs(change * 100)
        };
      }
    });
    
    return trends;
  }
  
  // Get current system metrics
  getSystemMetrics() {
    return { ...this.metrics.systemMetrics };
  }
  
  // Get performance report
  getPerformanceReport(timeWindow = 300) { // 5 minutes default
    const cutoff = Date.now() - timeWindow * 1000;
    const recentData = this.metrics.historicalData.filter(d => 
      d.timestamp.getTime() > cutoff
    );
    
    const report = {
      summary: {
        avgActiveAgents: this.calculateAverage(recentData, 'activeAgents'),
        avgMessageRate: this.calculateAverage(recentData, 'messageRate'),
        avgLatency: this.calculateAverage(recentData, 'avgLatency'),
        avgNetworkHealth: this.calculateAverage(recentData, 'networkHealth'),
        peakActiveAgents: Math.max(...recentData.map(d => d.activeAgents || 0)),
        peakMessageRate: Math.max(...recentData.map(d => d.messageRate || 0))
      },
      trends: this.calculateTrends(recentData),
      agentPerformance: this.getTopPerformingAgents(),
      bottlenecks: this.detectPerformanceBottlenecks(),
      recommendations: this.generateRecommendations()
    };
    
    return report;
  }
  
  // Calculate average of a specific metric
  calculateAverage(data, metric) {
    const values = data.map(d => d[metric] || 0).filter(v => v !== null);
    return values.length > 0 ? values.reduce((a, b) => a + b) / values.length : 0;
  }
  
  // Get top performing agents
  getTopPerformingAgents() {
    const agentScores = [];
    
    this.metrics.agentMetrics.forEach((metrics, agentId) => {
      const score = this.calculateAgentScore(metrics.averages);
      agentScores.push({ agentId, score, metrics: metrics.averages });
    });
    
    return agentScores
      .sort((a, b) => b.score - a.score)
      .slice(0, 10);
  }
  
  // Calculate agent performance score
  calculateAgentScore(averages) {
    const weights = {
      successRate: 0.3,
      tasksCompleted: 0.2,
      avgResponseTime: -0.2, // Lower is better
      resourceUtilization: -0.1, // Lower is better
      messageCount: 0.2
    };
    
    let score = 0;
    Object.entries(weights).forEach(([metric, weight]) => {
      if (averages[metric] !== undefined) {
        score += averages[metric] * weight;
      }
    });
    
    return score;
  }
  
  // Detect performance bottlenecks
  detectPerformanceBottlenecks() {
    const bottlenecks = [];
    
    // High latency detection
    if (this.metrics.systemMetrics.avgLatency > 100) {
      bottlenecks.push({
        type: 'latency',
        severity: 'high',
        value: this.metrics.systemMetrics.avgLatency,
        threshold: 100,
        recommendation: 'Optimize message routing or increase processing capacity'
      });
    }
    
    // High error rate detection
    if (this.metrics.systemMetrics.errorRate > 0.05) {
      bottlenecks.push({
        type: 'errors',
        severity: 'medium',
        value: this.metrics.systemMetrics.errorRate,
        threshold: 0.05,
        recommendation: 'Investigate error sources and improve error handling'
      });
    }
    
    // Memory usage detection
    if (this.metrics.systemMetrics.memoryUsage > 500) {
      bottlenecks.push({
        type: 'memory',
        severity: 'medium',
        value: this.metrics.systemMetrics.memoryUsage,
        threshold: 500,
        recommendation: 'Consider optimizing data structures or increasing memory'
      });
    }
    
    // Agent overload detection
    const overloadedAgents = [];
    this.metrics.agentMetrics.forEach((metrics, agentId) => {
      if (metrics.averages.resourceUtilization > 0.8) {
        overloadedAgents.push(agentId);
      }
    });
    
    if (overloadedAgents.length > 0) {
      bottlenecks.push({
        type: 'agent_overload',
        severity: 'high',
        agents: overloadedAgents,
        recommendation: 'Redistribute workload or spawn additional agents'
      });
    }
    
    return bottlenecks;
  }
  
  // Generate performance recommendations
  generateRecommendations() {
    const recommendations = [];
    const metrics = this.metrics.systemMetrics;
    
    // Scaling recommendations
    if (metrics.messageRate > 100 && metrics.avgLatency > 50) {
      recommendations.push({
        category: 'scaling',
        priority: 'high',
        action: 'Increase agent count',
        reason: 'High message rate with increasing latency',
        expectedImprovement: 'Reduce latency by 30-50%'
      });
    }
    
    // Physics optimization
    const recentData = this.metrics.historicalData.slice(-60);
    const avgAgents = this.calculateAverage(recentData, 'activeAgents');
    
    if (avgAgents > 50) {
      recommendations.push({
        category: 'physics',
        priority: 'medium',
        action: 'Optimize spring physics parameters',
        reason: 'Large agent count may benefit from adjusted physics',
        expectedImprovement: 'Improve visualization performance by 20%'
      });
    }
    
    // Communication optimization
    if (metrics.errorRate > 0.01) {
      recommendations.push({
        category: 'communication',
        priority: 'high',
        action: 'Implement retry logic and circuit breakers',
        reason: 'Error rate above acceptable threshold',
        expectedImprovement: 'Reduce error rate by 80%'
      });
    }
    
    // Resource optimization
    if (metrics.memoryUsage > 300) {
      recommendations.push({
        category: 'resources',
        priority: 'medium',
        action: 'Implement data pruning and compression',
        reason: 'Memory usage trending upward',
        expectedImprovement: 'Reduce memory usage by 40%'
      });
    }
    
    return recommendations;
  }
  
  // Analyze performance for specific metrics
  analyzePerformance(metrics, aggregation = 'avg') {
    const analysis = {};
    
    metrics.forEach(metric => {
      const values = this.metrics.historicalData.map(d => d[metric] || 0);
      
      switch (aggregation) {
        case 'avg':
          analysis[metric] = values.reduce((a, b) => a + b, 0) / values.length;
          break;
        case 'sum':
          analysis[metric] = values.reduce((a, b) => a + b, 0);
          break;
        case 'max':
          analysis[metric] = Math.max(...values);
          break;
        case 'min':
          analysis[metric] = Math.min(...values.filter(v => v > 0));
          break;
      }
    });
    
    return analysis;
  }
}