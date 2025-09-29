/**
 * Neural Insights - AI-powered insights and analytics dashboard
 */

import React, { useState, useEffect, useCallback } from 'react';
import { useNeural } from '../contexts/NeuralContext';
import '../styles/neural-theme.css';

interface InsightData {
  id: string;
  type: 'pattern' | 'optimization' | 'anomaly' | 'prediction' | 'performance';
  title: string;
  description: string;
  confidence: number;
  impact: 'low' | 'medium' | 'high' | 'critical';
  category: 'swarm' | 'agents' | 'workflows' | 'performance' | 'memory';
  timestamp: Date;
  data: any;
  actionable: boolean;
  actions?: Array<{
    label: string;
    type: 'primary' | 'secondary' | 'warning';
    action: () => void;
  }>;
}

interface MetricCard {
  id: string;
  title: string;
  value: string | number;
  change: number;
  trend: 'up' | 'down' | 'stable';
  unit?: string;
  format?: 'number' | 'percentage' | 'duration' | 'bytes';
}

const NeuralInsights: React.FC = () => {
  const neural = useNeural();
  const [insights, setInsights] = useState<InsightData[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedImpact, setSelectedImpact] = useState<string>('all');
  const [timeRange, setTimeRange] = useState<string>('24h');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Generate mock insights based on neural state
  const generateInsights = useCallback((): InsightData[] => {
    const currentTime = new Date();
    const mockInsights: InsightData[] = [];

    // Agent Performance Insights
    if (neural.agents.length > 0) {
      const avgPerformance = neural.agents.reduce((sum, agent) =>
        sum + agent.performance.successRate, 0) / neural.agents.length;

      if (avgPerformance < 0.8) {
        mockInsights.push({
          id: 'perf-low',
          type: 'performance',
          title: 'Low Agent Performance Detected',
          description: `Average success rate is ${Math.round(avgPerformance * 100)}%. Consider retraining or optimizing underperforming agents.`,
          confidence: 0.85,
          impact: 'medium',
          category: 'agents',
          timestamp: currentTime,
          data: { avgPerformance, affectedAgents: neural.agents.filter(a => a.performance.successRate < 0.8).length },
          actionable: true,
          actions: [
            { label: 'Optimize Agents', type: 'primary', action: () => console.log('Optimizing agents...') },
            { label: 'View Details', type: 'secondary', action: () => console.log('Viewing details...') }
          ]
        });
      }

      // Memory Usage Pattern
      const totalMemory = neural.memory.length;
      if (totalMemory > 1000) {
        mockInsights.push({
          id: 'memory-high',
          type: 'pattern',
          title: 'High Memory Usage Pattern',
          description: `Neural memory contains ${totalMemory} entries. Consider implementing memory cleanup strategies.`,
          confidence: 0.92,
          impact: 'medium',
          category: 'memory',
          timestamp: currentTime,
          data: { memoryCount: totalMemory },
          actionable: true,
          actions: [
            { label: 'Clean Memory', type: 'warning', action: () => console.log('Cleaning memory...') }
          ]
        });
      }

      // Cognitive Pattern Distribution
      const patterns = neural.agents.reduce((acc, agent) => {
        acc[agent.cognitivePattern] = (acc[agent.cognitivePattern] || 0) + 1;
        return acc;
      }, {} as Record<string, number>);

      const dominantPattern = Object.entries(patterns).reduce((a, b) =>
        patterns[a[0]] > patterns[b[0]] ? a : b
      )[0];

      mockInsights.push({
        id: 'pattern-dist',
        type: 'pattern',
        title: 'Cognitive Pattern Analysis',
        description: `${dominantPattern} pattern dominates (${Math.round((patterns[dominantPattern] / neural.agents.length) * 100)}%). Consider diversifying cognitive approaches.`,
        confidence: 0.78,
        impact: 'low',
        category: 'agents',
        timestamp: currentTime,
        data: { patterns, dominantPattern },
        actionable: true
      });
    }

    // Swarm Topology Insights
    if (neural.topology.nodes.length > 0) {
      const activeNodes = neural.topology.nodes.filter(n => n.status === 'online').length;
      const efficiency = activeNodes / neural.topology.nodes.length;

      if (efficiency < 0.9) {
        mockInsights.push({
          id: 'topology-efficiency',
          type: 'optimization',
          title: 'Suboptimal Swarm Topology',
          description: `Only ${Math.round(efficiency * 100)}% of nodes are active. Network efficiency could be improved.`,
          confidence: 0.88,
          impact: 'high',
          category: 'swarm',
          timestamp: currentTime,
          data: { efficiency, activeNodes, totalNodes: neural.topology.nodes.length },
          actionable: true,
          actions: [
            { label: 'Optimize Topology', type: 'primary', action: () => console.log('Optimizing topology...') },
            { label: 'Restart Nodes', type: 'warning', action: () => console.log('Restarting nodes...') }
          ]
        });
      }
    }

    // Workflow Efficiency
    const runningWorkflows = neural.workflows.filter(w => w.status === 'running');
    if (runningWorkflows.length > 0) {
      mockInsights.push({
        id: 'workflow-efficiency',
        type: 'performance',
        title: 'Workflow Optimization Opportunity',
        description: `${runningWorkflows.length} workflows are currently running. Monitor for bottlenecks and optimization opportunities.`,
        confidence: 0.72,
        impact: 'medium',
        category: 'workflows',
        timestamp: currentTime,
        data: { runningWorkflows: runningWorkflows.length },
        actionable: true
      });
    }

    // Predictive Insights
    mockInsights.push({
      id: 'prediction-load',
      type: 'prediction',
      title: 'Predicted Load Increase',
      description: 'Based on historical patterns, expect 25% increase in task volume over the next 2 hours.',
      confidence: 0.67,
      impact: 'medium',
      category: 'performance',
      timestamp: currentTime,
      data: { predictedIncrease: 0.25, timeframe: '2 hours' },
      actionable: true,
      actions: [
        { label: 'Scale Proactively', type: 'primary', action: () => console.log('Scaling swarm...') }
      ]
    });

    // Anomaly Detection
    if (neural.agents.some(a => a.performance.errorRate > 0.1)) {
      mockInsights.push({
        id: 'anomaly-errors',
        type: 'anomaly',
        title: 'Error Rate Anomaly Detected',
        description: 'Some agents showing unusually high error rates. Investigation recommended.',
        confidence: 0.91,
        impact: 'high',
        category: 'agents',
        timestamp: currentTime,
        data: { affectedAgents: neural.agents.filter(a => a.performance.errorRate > 0.1).length },
        actionable: true,
        actions: [
          { label: 'Investigate', type: 'warning', action: () => console.log('Investigating errors...') },
          { label: 'Reset Agents', type: 'warning', action: () => console.log('Resetting agents...') }
        ]
      });
    }

    return mockInsights;
  }, [neural]);

  // Generate metric cards
  const generateMetrics = useCallback((): MetricCard[] => {
    return [
      {
        id: 'total-agents',
        title: 'Active Agents',
        value: neural.agents.filter(a => a.status === 'active').length,
        change: 12,
        trend: 'up',
        format: 'number'
      },
      {
        id: 'success-rate',
        title: 'Success Rate',
        value: neural.agents.length > 0
          ? Math.round(neural.agents.reduce((sum, a) => sum + a.performance.successRate, 0) / neural.agents.length * 100)
          : 0,
        change: 5.2,
        trend: 'up',
        unit: '%',
        format: 'percentage'
      },
      {
        id: 'avg-response',
        title: 'Avg Response Time',
        value: neural.agents.length > 0
          ? Math.round(neural.agents.reduce((sum, a) => sum + a.performance.averageResponseTime, 0) / neural.agents.length)
          : 0,
        change: -8.1,
        trend: 'down',
        unit: 'ms',
        format: 'duration'
      },
      {
        id: 'memory-usage',
        title: 'Memory Entries',
        value: neural.memory.length,
        change: 15.3,
        trend: 'up',
        format: 'number'
      },
      {
        id: 'network-efficiency',
        title: 'Network Efficiency',
        value: neural.topology.nodes.length > 0
          ? Math.round((neural.topology.nodes.filter(n => n.status === 'online').length / neural.topology.nodes.length) * 100)
          : 0,
        change: -2.1,
        trend: 'down',
        unit: '%',
        format: 'percentage'
      },
      {
        id: 'active-workflows',
        title: 'Active Workflows',
        value: neural.workflows.filter(w => w.status === 'running').length,
        change: 0,
        trend: 'stable',
        format: 'number'
      }
    ];
  }, [neural]);

  // Auto-refresh insights
  useEffect(() => {
    const refreshInsights = () => {
      setIsAnalyzing(true);
      setTimeout(() => {
        setInsights(generateInsights());
        setIsAnalyzing(false);
      }, 1000);
    };

    refreshInsights();

    if (autoRefresh) {
      const interval = setInterval(refreshInsights, 30000); // 30 seconds
      return () => clearInterval(interval);
    }
  }, [autoRefresh, generateInsights]);

  const filteredInsights = insights.filter(insight => {
    if (selectedCategory !== 'all' && insight.category !== selectedCategory) return false;
    if (selectedImpact !== 'all' && insight.impact !== selectedImpact) return false;
    return true;
  });

  const getImpactColor = (impact: string) => {
    const colors = {
      low: 'neural-badge-success',
      medium: 'neural-badge-warning',
      high: 'neural-badge-error',
      critical: 'neural-badge-error'
    };
    return colors[impact as keyof typeof colors] || 'neural-badge-primary';
  };

  const getTypeIcon = (type: string) => {
    const icons = {
      pattern: '🔍',
      optimization: '⚡',
      anomaly: '⚠️',
      prediction: '🔮',
      performance: '📊'
    };
    return icons[type as keyof typeof icons] || '💡';
  };

  const getTrendIcon = (trend: string) => {
    const icons = {
      up: '📈',
      down: '📉',
      stable: '➡️'
    };
    return icons[trend as keyof typeof icons] || '➡️';
  };

  const formatValue = (card: MetricCard) => {
    if (card.format === 'percentage') return `${card.value}%`;
    if (card.format === 'duration') return `${card.value}ms`;
    if (card.format === 'bytes') return `${card.value}MB`;
    return card.value.toString();
  };

  const metrics = generateMetrics();

  return (
    <div className="neural-theme h-full neural-flex neural-flex-col">
      {/* Header */}
      <div className="neural-card-header neural-flex neural-flex-between items-center">
        <div>
          <h2 className="neural-heading neural-heading-md">Neural Insights</h2>
          <p className="neural-text-muted">AI-powered analytics and optimization recommendations</p>
        </div>
        <div className="neural-flex items-center gap-4">
          <div className="neural-flex items-center gap-2">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="neural-checkbox"
            />
            <label className="neural-text-secondary text-sm">Auto-refresh</label>
          </div>
          <button
            onClick={() => {
              setIsAnalyzing(true);
              setTimeout(() => {
                setInsights(generateInsights());
                setIsAnalyzing(false);
              }, 1000);
            }}
            disabled={isAnalyzing}
            className="neural-btn neural-btn-primary neural-btn-sm"
          >
            {isAnalyzing ? 'Analyzing...' : 'Refresh'}
          </button>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="p-6">
        <div className="neural-grid neural-grid-3 gap-4 mb-6">
          {metrics.map(metric => (
            <div key={metric.id} className="neural-card neural-card-body">
              <div className="neural-flex neural-flex-between items-start mb-2">
                <div>
                  <p className="neural-text-muted text-sm">{metric.title}</p>
                  <p className="neural-heading neural-heading-lg">
                    {formatValue(metric)}
                  </p>
                </div>
                <span className="text-lg">{getTrendIcon(metric.trend)}</span>
              </div>
              <div className="neural-flex items-center gap-2">
                <span className={`text-sm ${
                  metric.change > 0 ? 'neural-text-success' :
                  metric.change < 0 ? 'neural-text-error' : 'neural-text-muted'
                }`}>
                  {metric.change > 0 ? '+' : ''}{metric.change}%
                </span>
                <span className="neural-text-muted text-sm">vs last period</span>
              </div>
            </div>
          ))}
        </div>

        {/* Filters */}
        <div className="neural-card neural-card-body mb-6">
          <div className="neural-flex neural-flex-wrap gap-4">
            <div>
              <label className="neural-text-secondary text-sm mb-1 block">Category</label>
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="neural-input neural-select"
              >
                <option value="all">All Categories</option>
                <option value="swarm">Swarm</option>
                <option value="agents">Agents</option>
                <option value="workflows">Workflows</option>
                <option value="performance">Performance</option>
                <option value="memory">Memory</option>
              </select>
            </div>
            <div>
              <label className="neural-text-secondary text-sm mb-1 block">Impact</label>
              <select
                value={selectedImpact}
                onChange={(e) => setSelectedImpact(e.target.value)}
                className="neural-input neural-select"
              >
                <option value="all">All Impact Levels</option>
                <option value="critical">Critical</option>
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
              </select>
            </div>
            <div>
              <label className="neural-text-secondary text-sm mb-1 block">Time Range</label>
              <select
                value={timeRange}
                onChange={(e) => setTimeRange(e.target.value)}
                className="neural-input neural-select"
              >
                <option value="1h">Last Hour</option>
                <option value="24h">Last 24 Hours</option>
                <option value="7d">Last 7 Days</option>
                <option value="30d">Last 30 Days</option>
              </select>
            </div>
          </div>
        </div>

        {/* Insights List */}
        <div className="neural-space-y-4">
          {isAnalyzing ? (
            <div className="neural-card neural-card-body neural-flex neural-flex-center p-8">
              <div className="neural-flex items-center gap-3">
                <div className="neural-spinner"></div>
                <span className="neural-text-muted">Analyzing neural patterns...</span>
              </div>
            </div>
          ) : filteredInsights.length === 0 ? (
            <div className="neural-card neural-card-body neural-flex neural-flex-center p-8">
              <div className="text-center">
                <span className="text-4xl mb-4 block">🎯</span>
                <h3 className="neural-heading neural-heading-sm mb-2">No Insights Found</h3>
                <p className="neural-text-muted">
                  No insights match your current filters. Try adjusting the criteria or check back later.
                </p>
              </div>
            </div>
          ) : (
            filteredInsights.map(insight => (
              <div key={insight.id} className="neural-card">
                <div className="neural-card-body">
                  <div className="neural-flex neural-flex-between items-start mb-3">
                    <div className="neural-flex items-center gap-3">
                      <span className="text-2xl">{getTypeIcon(insight.type)}</span>
                      <div>
                        <h3 className="neural-heading neural-heading-sm">{insight.title}</h3>
                        <div className="neural-flex items-center gap-2 mt-1">
                          <span className={`neural-badge ${getImpactColor(insight.impact)} text-xs`}>
                            {insight.impact}
                          </span>
                          <span className="neural-badge neural-badge-secondary text-xs">
                            {insight.category}
                          </span>
                          <span className="neural-text-muted text-xs">
                            {Math.round(insight.confidence * 100)}% confidence
                          </span>
                        </div>
                      </div>
                    </div>
                    <span className="neural-text-muted text-sm">
                      {insight.timestamp.toLocaleTimeString()}
                    </span>
                  </div>

                  <p className="neural-text-secondary mb-4">{insight.description}</p>

                  {insight.data && (
                    <details className="mb-4">
                      <summary className="neural-text-muted text-sm cursor-pointer">
                        View Data
                      </summary>
                      <pre className="neural-bg-tertiary p-3 rounded mt-2 text-sm overflow-auto">
                        {JSON.stringify(insight.data, null, 2)}
                      </pre>
                    </details>
                  )}

                  {insight.actions && insight.actions.length > 0 && (
                    <div className="neural-flex gap-2">
                      {insight.actions.map((action, index) => (
                        <button
                          key={index}
                          onClick={action.action}
                          className={`neural-btn neural-btn-${action.type} neural-btn-sm`}
                        >
                          {action.label}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default NeuralInsights;