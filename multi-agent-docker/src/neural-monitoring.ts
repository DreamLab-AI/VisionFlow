/**
 * Neural Monitoring System
 * Advanced system performance tracking and neural processing analytics
 */

import { EventEmitter } from 'events';
import pino from 'pino';
import { v4 as uuidv4 } from 'uuid';
import * as si from 'systeminformation';

interface MonitoringConfig {
  enabled: boolean;
  metrics: string[];
  interval: number;
  retention: {
    raw: number; // seconds
    aggregated: number; // seconds
  };
  alerts: {
    enabled: boolean;
    thresholds: {
      cpu: number;
      memory: number;
      gpu: number;
      neural: number;
    };
  };
}

interface SystemMetrics {
  timestamp: Date;
  system: {
    uptime: number;
    load: number[];
    cpu: {
      usage: number;
      cores: number;
      speed: number;
      temperature?: number;
    };
    memory: {
      total: number;
      used: number;
      free: number;
      percentage: number;
    };
    disk: {
      total: number;
      used: number;
      free: number;
      percentage: number;
      readSpeed: number;
      writeSpeed: number;
    };
    network: {
      interfaces: Array<{
        name: string;
        rx: number;
        tx: number;
        speed?: number;
      }>;
      totalRx: number;
      totalTx: number;
    };
    gpu?: Array<{
      usage: number;
      memory: {
        total: number;
        used: number;
        percentage: number;
      };
      temperature?: number;
      powerDraw?: number;
    }>;
  };
}

interface NeuralMetrics {
  timestamp: Date;
  processing: {
    totalRequests: number;
    successfulRequests: number;
    failedRequests: number;
    averageProcessingTime: number;
    throughput: number; // requests per second
    queueSize: number;
  };
  models: Array<{
    name: string;
    usage: number;
    requests: number;
    averageLatency: number;
    errorRate: number;
  }>;
  resources: {
    cpuUsage: number;
    memoryUsage: number;
    gpuUsage?: number;
    contextSwitches: number;
  };
  quality: {
    averageConfidence: number;
    accuracyScore: number;
    diversityIndex: number;
  };
}

interface ApplicationMetrics {
  timestamp: Date;
  components: {
    mcpBridge: {
      connected: boolean;
      bridges: number;
      messagesProcessed: number;
      averageLatency: number;
      errorRate: number;
    };
    webSocket: {
      activeConnections: number;
      totalMessages: number;
      averageLatency: number;
      subscriptions: number;
    };
    apiGateway: {
      totalRequests: number;
      successRate: number;
      averageResponseTime: number;
      cacheHitRate: number;
    };
    resourceManager: {
      allocations: number;
      optimizations: number;
      gpuUtilization: number;
      memoryEfficiency: number;
    };
  };
}

interface Alert {
  id: string;
  timestamp: Date;
  type: 'warning' | 'error' | 'critical';
  component: string;
  metric: string;
  value: number;
  threshold: number;
  message: string;
  resolved?: boolean;
  resolvedAt?: Date;
}

interface MetricsHistory {
  system: SystemMetrics[];
  neural: NeuralMetrics[];
  application: ApplicationMetrics[];
  alerts: Alert[];
}

class NeuralMonitoring extends EventEmitter {
  private config: MonitoringConfig;
  private logger: pino.Logger;
  private monitoring: boolean;
  private intervals: Map<string, NodeJS.Timeout>;
  private metricsHistory: MetricsHistory;
  private currentMetrics: {
    system?: SystemMetrics;
    neural?: NeuralMetrics;
    application?: ApplicationMetrics;
  };
  private alertsHistory: Alert[];
  private lastCleanup: Date;

  constructor(config: MonitoringConfig, logger: pino.Logger) {
    super();
    this.config = config;
    this.logger = logger.child({ component: 'neural-monitoring' });
    this.monitoring = false;
    this.intervals = new Map();
    this.currentMetrics = {};
    this.alertsHistory = [];
    this.lastCleanup = new Date();

    this.metricsHistory = {
      system: [],
      neural: [],
      application: [],
      alerts: []
    };

    this.initializeMonitoring();
  }

  private initializeMonitoring(): void {
    if (!this.config.enabled) {
      this.logger.info('Monitoring disabled by configuration');
      return;
    }

    this.logger.info('Initializing neural monitoring system', {
      metrics: this.config.metrics,
      interval: this.config.interval,
      alerts: this.config.alerts.enabled
    });
  }

  public async start(): Promise<void> {
    if (this.monitoring || !this.config.enabled) {
      return;
    }

    this.monitoring = true;
    this.logger.info('Starting neural monitoring system');

    try {
      // Start system metrics collection
      if (this.config.metrics.includes('system')) {
        this.intervals.set('system', setInterval(async () => {
          await this.collectSystemMetrics();
        }, this.config.interval));
      }

      // Start neural metrics collection
      if (this.config.metrics.includes('neural')) {
        this.intervals.set('neural', setInterval(async () => {
          await this.collectNeuralMetrics();
        }, this.config.interval));
      }

      // Start application metrics collection
      if (this.config.metrics.includes('application')) {
        this.intervals.set('application', setInterval(async () => {
          await this.collectApplicationMetrics();
        }, this.config.interval));
      }

      // Start cleanup process
      this.intervals.set('cleanup', setInterval(() => {
        this.cleanupOldMetrics();
      }, 60000)); // Cleanup every minute

      // Start alert processing
      if (this.config.alerts.enabled) {
        this.intervals.set('alerts', setInterval(() => {
          this.processAlerts();
        }, this.config.interval / 2)); // Check alerts more frequently
      }

      this.logger.info('Neural monitoring system started successfully');
      this.emit('monitoring-started');
    } catch (error) {
      this.logger.error('Failed to start monitoring system', { error });
      throw error;
    }
  }

  public async stop(): Promise<void> {
    if (!this.monitoring) return;

    this.monitoring = false;
    this.logger.info('Stopping neural monitoring system');

    // Clear all intervals
    for (const [name, interval] of this.intervals) {
      clearInterval(interval);
    }
    this.intervals.clear();

    this.logger.info('Neural monitoring system stopped');
    this.emit('monitoring-stopped');
  }

  private async collectSystemMetrics(): Promise<void> {
    try {
      const [cpuData, memData, diskData, networkData, gpuData, loadData] = await Promise.all([
        si.cpu(),
        si.mem(),
        si.fsSize(),
        si.networkStats(),
        si.graphics(),
        si.currentLoad()
      ]);

      const systemMetrics: SystemMetrics = {
        timestamp: new Date(),
        system: {
          uptime: process.uptime(),
          load: loadData.cpus?.map(cpu => cpu.load) || [],
          cpu: {
            usage: loadData.currentLoad,
            cores: cpuData.cores,
            speed: cpuData.speed,
            temperature: loadData.cpus?.[0]?.temperature
          },
          memory: {
            total: memData.total,
            used: memData.used,
            free: memData.free,
            percentage: (memData.used / memData.total) * 100
          },
          disk: {
            total: diskData.reduce((acc, disk) => acc + disk.size, 0),
            used: diskData.reduce((acc, disk) => acc + disk.used, 0),
            free: diskData.reduce((acc, disk) => acc + disk.available, 0),
            percentage: diskData.reduce((acc, disk) => acc + (disk.use || 0), 0) / diskData.length,
            readSpeed: 0, // Would need more complex tracking
            writeSpeed: 0
          },
          network: {
            interfaces: networkData.map(iface => ({
              name: iface.iface,
              rx: iface.rx_bytes,
              tx: iface.tx_bytes,
              speed: iface.speed
            })),
            totalRx: networkData.reduce((acc, iface) => acc + iface.rx_bytes, 0),
            totalTx: networkData.reduce((acc, iface) => acc + iface.tx_bytes, 0)
          },
          gpu: gpuData.controllers?.map(gpu => ({
            usage: gpu.utilizationGpu || 0,
            memory: {
              total: gpu.memoryTotal || 0,
              used: gpu.memoryUsed || 0,
              percentage: gpu.memoryTotal ? ((gpu.memoryUsed || 0) / gpu.memoryTotal) * 100 : 0
            },
            temperature: gpu.temperatureGpu,
            powerDraw: gpu.powerDraw
          }))
        }
      };

      this.currentMetrics.system = systemMetrics;
      this.metricsHistory.system.push(systemMetrics);

      this.emit('system-metrics-collected', systemMetrics);
    } catch (error) {
      this.logger.error('Error collecting system metrics', { error });
    }
  }

  private async collectNeuralMetrics(): Promise<void> {
    try {
      // This would be integrated with the actual neural processing components
      // For now, creating a structure that represents what would be collected
      const neuralMetrics: NeuralMetrics = {
        timestamp: new Date(),
        processing: {
          totalRequests: this.getTotalNeuralRequests(),
          successfulRequests: this.getSuccessfulNeuralRequests(),
          failedRequests: this.getFailedNeuralRequests(),
          averageProcessingTime: this.getAverageProcessingTime(),
          throughput: this.calculateThroughput(),
          queueSize: this.getNeuralQueueSize()
        },
        models: this.getModelMetrics(),
        resources: {
          cpuUsage: this.getNeuralCPUUsage(),
          memoryUsage: this.getNeuralMemoryUsage(),
          gpuUsage: this.getNeuralGPUUsage(),
          contextSwitches: this.getContextSwitches()
        },
        quality: {
          averageConfidence: this.getAverageConfidence(),
          accuracyScore: this.getAccuracyScore(),
          diversityIndex: this.getDiversityIndex()
        }
      };

      this.currentMetrics.neural = neuralMetrics;
      this.metricsHistory.neural.push(neuralMetrics);

      this.emit('neural-metrics-collected', neuralMetrics);
    } catch (error) {
      this.logger.error('Error collecting neural metrics', { error });
    }
  }

  private async collectApplicationMetrics(): Promise<void> {
    try {
      const applicationMetrics: ApplicationMetrics = {
        timestamp: new Date(),
        components: {
          mcpBridge: {
            connected: true, // Would get from actual MCP bridge
            bridges: 3,
            messagesProcessed: this.getMCPMessagesProcessed(),
            averageLatency: this.getMCPAverageLatency(),
            errorRate: this.getMCPErrorRate()
          },
          webSocket: {
            activeConnections: this.getActiveWebSocketConnections(),
            totalMessages: this.getTotalWebSocketMessages(),
            averageLatency: this.getWebSocketAverageLatency(),
            subscriptions: this.getWebSocketSubscriptions()
          },
          apiGateway: {
            totalRequests: this.getAPITotalRequests(),
            successRate: this.getAPISuccessRate(),
            averageResponseTime: this.getAPIAverageResponseTime(),
            cacheHitRate: this.getAPICacheHitRate()
          },
          resourceManager: {
            allocations: this.getResourceAllocations(),
            optimizations: this.getResourceOptimizations(),
            gpuUtilization: this.getGPUUtilization(),
            memoryEfficiency: this.getMemoryEfficiency()
          }
        }
      };

      this.currentMetrics.application = applicationMetrics;
      this.metricsHistory.application.push(applicationMetrics);

      this.emit('application-metrics-collected', applicationMetrics);
    } catch (error) {
      this.logger.error('Error collecting application metrics', { error });
    }
  }

  private processAlerts(): void {
    if (!this.config.alerts.enabled) return;

    try {
      const alerts: Alert[] = [];
      const now = new Date();

      // Check system metrics alerts
      if (this.currentMetrics.system) {
        const system = this.currentMetrics.system.system;

        if (system.cpu.usage > this.config.alerts.thresholds.cpu) {
          alerts.push({
            id: uuidv4(),
            timestamp: now,
            type: 'warning',
            component: 'system',
            metric: 'cpu',
            value: system.cpu.usage,
            threshold: this.config.alerts.thresholds.cpu,
            message: `High CPU usage detected: ${system.cpu.usage.toFixed(2)}%`
          });
        }

        if (system.memory.percentage > this.config.alerts.thresholds.memory) {
          alerts.push({
            id: uuidv4(),
            timestamp: now,
            type: 'warning',
            component: 'system',
            metric: 'memory',
            value: system.memory.percentage,
            threshold: this.config.alerts.thresholds.memory,
            message: `High memory usage detected: ${system.memory.percentage.toFixed(2)}%`
          });
        }

        if (system.gpu) {
          for (let i = 0; i < system.gpu.length; i++) {
            const gpu = system.gpu[i];
            if (gpu.memory.percentage > this.config.alerts.thresholds.gpu) {
              alerts.push({
                id: uuidv4(),
                timestamp: now,
                type: 'warning',
                component: 'system',
                metric: 'gpu',
                value: gpu.memory.percentage,
                threshold: this.config.alerts.thresholds.gpu,
                message: `High GPU ${i} memory usage detected: ${gpu.memory.percentage.toFixed(2)}%`
              });
            }
          }
        }
      }

      // Check neural metrics alerts
      if (this.currentMetrics.neural) {
        const neural = this.currentMetrics.neural;

        const errorRate = neural.processing.failedRequests / neural.processing.totalRequests * 100;
        if (errorRate > this.config.alerts.thresholds.neural) {
          alerts.push({
            id: uuidv4(),
            timestamp: now,
            type: 'error',
            component: 'neural',
            metric: 'error_rate',
            value: errorRate,
            threshold: this.config.alerts.thresholds.neural,
            message: `High neural processing error rate detected: ${errorRate.toFixed(2)}%`
          });
        }
      }

      // Process new alerts
      for (const alert of alerts) {
        this.alertsHistory.push(alert);
        this.metricsHistory.alerts.push(alert);

        this.logger.warn('Alert generated', {
          id: alert.id,
          type: alert.type,
          component: alert.component,
          metric: alert.metric,
          value: alert.value,
          threshold: alert.threshold
        });

        this.emit('alert-generated', alert);
      }
    } catch (error) {
      this.logger.error('Error processing alerts', { error });
    }
  }

  private cleanupOldMetrics(): void {
    const now = Date.now();
    const rawRetention = this.config.retention.raw * 1000;
    const aggregatedRetention = this.config.retention.aggregated * 1000;

    // Clean up system metrics
    this.metricsHistory.system = this.metricsHistory.system.filter(
      metric => now - metric.timestamp.getTime() < rawRetention
    );

    // Clean up neural metrics
    this.metricsHistory.neural = this.metricsHistory.neural.filter(
      metric => now - metric.timestamp.getTime() < rawRetention
    );

    // Clean up application metrics
    this.metricsHistory.application = this.metricsHistory.application.filter(
      metric => now - metric.timestamp.getTime() < rawRetention
    );

    // Clean up old alerts
    this.metricsHistory.alerts = this.metricsHistory.alerts.filter(
      alert => now - alert.timestamp.getTime() < aggregatedRetention
    );

    this.lastCleanup = new Date();
  }

  // Placeholder methods for metrics collection (would be implemented with actual data sources)
  private getTotalNeuralRequests(): number { return Math.floor(Math.random() * 1000); }
  private getSuccessfulNeuralRequests(): number { return Math.floor(Math.random() * 950); }
  private getFailedNeuralRequests(): number { return Math.floor(Math.random() * 50); }
  private getAverageProcessingTime(): number { return Math.random() * 1000; }
  private calculateThroughput(): number { return Math.random() * 100; }
  private getNeuralQueueSize(): number { return Math.floor(Math.random() * 10); }
  private getNeuralCPUUsage(): number { return Math.random() * 100; }
  private getNeuralMemoryUsage(): number { return Math.random() * 100; }
  private getNeuralGPUUsage(): number { return Math.random() * 100; }
  private getContextSwitches(): number { return Math.floor(Math.random() * 1000); }
  private getAverageConfidence(): number { return 0.8 + Math.random() * 0.2; }
  private getAccuracyScore(): number { return 0.85 + Math.random() * 0.15; }
  private getDiversityIndex(): number { return Math.random(); }

  private getModelMetrics(): NeuralMetrics['models'] {
    return [
      {
        name: 'gpt-4',
        usage: Math.random() * 100,
        requests: Math.floor(Math.random() * 500),
        averageLatency: Math.random() * 1000,
        errorRate: Math.random() * 5
      },
      {
        name: 'claude-3',
        usage: Math.random() * 100,
        requests: Math.floor(Math.random() * 300),
        averageLatency: Math.random() * 800,
        errorRate: Math.random() * 3
      }
    ];
  }

  private getMCPMessagesProcessed(): number { return Math.floor(Math.random() * 500); }
  private getMCPAverageLatency(): number { return Math.random() * 100; }
  private getMCPErrorRate(): number { return Math.random() * 5; }
  private getActiveWebSocketConnections(): number { return Math.floor(Math.random() * 50); }
  private getTotalWebSocketMessages(): number { return Math.floor(Math.random() * 1000); }
  private getWebSocketAverageLatency(): number { return Math.random() * 50; }
  private getWebSocketSubscriptions(): number { return Math.floor(Math.random() * 20); }
  private getAPITotalRequests(): number { return Math.floor(Math.random() * 2000); }
  private getAPISuccessRate(): number { return 95 + Math.random() * 5; }
  private getAPIAverageResponseTime(): number { return Math.random() * 200; }
  private getAPICacheHitRate(): number { return 70 + Math.random() * 30; }
  private getResourceAllocations(): number { return Math.floor(Math.random() * 20); }
  private getResourceOptimizations(): number { return Math.floor(Math.random() * 10); }
  private getGPUUtilization(): number { return Math.random() * 100; }
  private getMemoryEfficiency(): number { return 80 + Math.random() * 20; }

  public getMetrics(): any {
    return {
      current: this.currentMetrics,
      history: {
        system: this.metricsHistory.system.slice(-100), // Last 100 entries
        neural: this.metricsHistory.neural.slice(-100),
        application: this.metricsHistory.application.slice(-100),
        alerts: this.metricsHistory.alerts.slice(-50) // Last 50 alerts
      },
      summary: this.generateSummary()
    };
  }

  private generateSummary(): any {
    const recentSystem = this.metricsHistory.system.slice(-10);
    const recentNeural = this.metricsHistory.neural.slice(-10);
    const recentApplication = this.metricsHistory.application.slice(-10);
    const recentAlerts = this.metricsHistory.alerts.slice(-10);

    return {
      averages: {
        cpu: recentSystem.reduce((acc, m) => acc + m.system.cpu.usage, 0) / recentSystem.length || 0,
        memory: recentSystem.reduce((acc, m) => acc + m.system.memory.percentage, 0) / recentSystem.length || 0,
        neuralThroughput: recentNeural.reduce((acc, m) => acc + m.processing.throughput, 0) / recentNeural.length || 0
      },
      trends: {
        systemLoad: this.calculateTrend(recentSystem.map(m => m.system.cpu.usage)),
        memoryUsage: this.calculateTrend(recentSystem.map(m => m.system.memory.percentage)),
        neuralProcessing: this.calculateTrend(recentNeural.map(m => m.processing.averageProcessingTime))
      },
      alerts: {
        total: recentAlerts.length,
        warnings: recentAlerts.filter(a => a.type === 'warning').length,
        errors: recentAlerts.filter(a => a.type === 'error').length,
        critical: recentAlerts.filter(a => a.type === 'critical').length
      }
    };
  }

  private calculateTrend(values: number[]): 'increasing' | 'decreasing' | 'stable' {
    if (values.length < 2) return 'stable';

    const first = values.slice(0, Math.floor(values.length / 2));
    const second = values.slice(Math.floor(values.length / 2));

    const firstAvg = first.reduce((a, b) => a + b, 0) / first.length;
    const secondAvg = second.reduce((a, b) => a + b, 0) / second.length;

    const difference = ((secondAvg - firstAvg) / firstAvg) * 100;

    if (difference > 5) return 'increasing';
    if (difference < -5) return 'decreasing';
    return 'stable';
  }

  public getActiveAlerts(): Alert[] {
    return this.alertsHistory.filter(alert => !alert.resolved);
  }

  public resolveAlert(alertId: string): void {
    const alert = this.alertsHistory.find(a => a.id === alertId);
    if (alert) {
      alert.resolved = true;
      alert.resolvedAt = new Date();
      this.emit('alert-resolved', alert);
    }
  }

  public isActive(): boolean {
    return this.monitoring;
  }

  public getStatus(): any {
    return {
      monitoring: this.monitoring,
      config: this.config,
      lastCleanup: this.lastCleanup,
      metricsCount: {
        system: this.metricsHistory.system.length,
        neural: this.metricsHistory.neural.length,
        application: this.metricsHistory.application.length,
        alerts: this.metricsHistory.alerts.length
      },
      intervals: Array.from(this.intervals.keys())
    };
  }
}

export {
  NeuralMonitoring,
  MonitoringConfig,
  SystemMetrics,
  NeuralMetrics,
  ApplicationMetrics,
  Alert,
  MetricsHistory
};