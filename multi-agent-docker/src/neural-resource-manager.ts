/**
 * Neural Resource Manager
 * GPU/CPU optimization and resource allocation for neural processing
 */

import { EventEmitter } from 'events';
import * as si from 'systeminformation';
import { GPU } from 'gpu-js';
import pino from 'pino';

interface ResourceMetrics {
  cpu: {
    usage: number;
    cores: number;
    temperature?: number;
    load: number[];
  };
  memory: {
    used: number;
    available: number;
    total: number;
    percentage: number;
  };
  gpu: {
    usage: number;
    memory: {
      used: number;
      total: number;
      percentage: number;
    };
    temperature?: number;
    powerDraw?: number;
  }[];
  disk: {
    usage: number;
    available: number;
    total: number;
  };
  network: {
    bytesIn: number;
    bytesOut: number;
    packetsIn: number;
    packetsOut: number;
  };
}

interface ResourceAllocation {
  processId: string;
  type: 'neural' | 'mcp' | 'websocket' | 'api';
  resources: {
    cpuCores?: number[];
    memoryMB?: number;
    gpuIndex?: number;
    gpuMemoryMB?: number;
  };
  priority: 'low' | 'medium' | 'high' | 'critical';
  startTime: Date;
  estimatedDuration?: number;
}

interface OptimizationStrategy {
  name: string;
  description: string;
  enabled: boolean;
  thresholds: {
    cpuThreshold: number;
    memoryThreshold: number;
    gpuThreshold: number;
  };
  actions: {
    scaleDown: boolean;
    redistributeLoad: boolean;
    useGPU: boolean;
    cacheClear: boolean;
  };
}

class NeuralResourceManager extends EventEmitter {
  private logger: pino.Logger;
  private gpu: GPU;
  private currentMetrics: ResourceMetrics;
  private allocations: Map<string, ResourceAllocation>;
  private strategies: Map<string, OptimizationStrategy>;
  private monitoring: boolean;
  private monitoringInterval: NodeJS.Timeout | null;
  private lastOptimization: Date;

  constructor(logger: pino.Logger) {
    super();
    this.logger = logger.child({ component: 'neural-resource-manager' });
    this.allocations = new Map();
    this.strategies = new Map();
    this.monitoring = false;
    this.monitoringInterval = null;
    this.lastOptimization = new Date();

    this.currentMetrics = {
      cpu: { usage: 0, cores: 0, load: [] },
      memory: { used: 0, available: 0, total: 0, percentage: 0 },
      gpu: [],
      disk: { usage: 0, available: 0, total: 0 },
      network: { bytesIn: 0, bytesOut: 0, packetsIn: 0, packetsOut: 0 }
    };

    this.initializeGPU();
    this.initializeStrategies();
    this.startMonitoring();
  }

  private async initializeGPU(): Promise<void> {
    try {
      this.gpu = new GPU({
        mode: 'gpu'
      });

      // Test GPU availability
      const testKernel = this.gpu.createKernel(function() {
        return 1;
      }).setOutput([1]);

      testKernel();
      this.logger.info('GPU initialized successfully');
    } catch (error) {
      this.logger.warn('GPU initialization failed, falling back to CPU', { error });
      this.gpu = new GPU({
        mode: 'cpu'
      });
    }
  }

  private initializeStrategies(): void {
    // Conservative strategy
    this.strategies.set('conservative', {
      name: 'Conservative',
      description: 'Minimal resource usage with safety margins',
      enabled: true,
      thresholds: {
        cpuThreshold: 70,
        memoryThreshold: 75,
        gpuThreshold: 80
      },
      actions: {
        scaleDown: true,
        redistributeLoad: false,
        useGPU: false,
        cacheClear: true
      }
    });

    // Balanced strategy
    this.strategies.set('balanced', {
      name: 'Balanced',
      description: 'Optimal balance between performance and resource usage',
      enabled: true,
      thresholds: {
        cpuThreshold: 80,
        memoryThreshold: 85,
        gpuThreshold: 90
      },
      actions: {
        scaleDown: true,
        redistributeLoad: true,
        useGPU: true,
        cacheClear: false
      }
    });

    // Aggressive strategy
    this.strategies.set('aggressive', {
      name: 'Aggressive',
      description: 'Maximum performance utilization',
      enabled: false,
      thresholds: {
        cpuThreshold: 95,
        memoryThreshold: 95,
        gpuThreshold: 95
      },
      actions: {
        scaleDown: false,
        redistributeLoad: true,
        useGPU: true,
        cacheClear: false
      }
    });

    this.logger.info('Resource optimization strategies initialized');
  }

  private async startMonitoring(): Promise<void> {
    if (this.monitoring) return;

    this.monitoring = true;
    this.monitoringInterval = setInterval(async () => {
      await this.updateMetrics();
      await this.optimizeResources();
    }, 5000); // Update every 5 seconds

    this.logger.info('Resource monitoring started');
  }

  private async updateMetrics(): Promise<void> {
    try {
      const [cpuData, memData, gpuData, diskData, networkData] = await Promise.all([
        si.currentLoad(),
        si.mem(),
        si.graphics(),
        si.fsSize(),
        si.networkStats()
      ]);

      // Update CPU metrics
      this.currentMetrics.cpu = {
        usage: cpuData.currentLoad,
        cores: cpuData.cpus?.length || 0,
        temperature: cpuData.cpus?.[0]?.temperature,
        load: cpuData.cpus?.map(cpu => cpu.load) || []
      };

      // Update Memory metrics
      this.currentMetrics.memory = {
        used: memData.used,
        available: memData.available,
        total: memData.total,
        percentage: (memData.used / memData.total) * 100
      };

      // Update GPU metrics
      this.currentMetrics.gpu = gpuData.controllers?.map(gpu => ({
        usage: gpu.utilizationGpu || 0,
        memory: {
          used: gpu.memoryUsed || 0,
          total: gpu.memoryTotal || 0,
          percentage: gpu.memoryTotal ? ((gpu.memoryUsed || 0) / gpu.memoryTotal) * 100 : 0
        },
        temperature: gpu.temperatureGpu,
        powerDraw: gpu.powerDraw
      })) || [];

      // Update Disk metrics
      const primaryDisk = diskData[0];
      if (primaryDisk) {
        this.currentMetrics.disk = {
          usage: primaryDisk.used,
          available: primaryDisk.available,
          total: primaryDisk.size
        };
      }

      // Update Network metrics
      const primaryNetwork = networkData[0];
      if (primaryNetwork) {
        this.currentMetrics.network = {
          bytesIn: primaryNetwork.rx_bytes,
          bytesOut: primaryNetwork.tx_bytes,
          packetsIn: primaryNetwork.rx_packets,
          packetsOut: primaryNetwork.tx_packets
        };
      }

      this.emit('metrics-updated', this.currentMetrics);
    } catch (error) {
      this.logger.error('Error updating metrics', { error });
    }
  }

  private async optimizeResources(): Promise<void> {
    try {
      const enabledStrategies = Array.from(this.strategies.values()).filter(s => s.enabled);

      for (const strategy of enabledStrategies) {
        await this.applyOptimizationStrategy(strategy);
      }

      this.lastOptimization = new Date();
    } catch (error) {
      this.logger.error('Error optimizing resources', { error });
    }
  }

  private async applyOptimizationStrategy(strategy: OptimizationStrategy): Promise<void> {
    const metrics = this.currentMetrics;
    let optimizationNeeded = false;
    const actions: string[] = [];

    // Check CPU threshold
    if (metrics.cpu.usage > strategy.thresholds.cpuThreshold) {
      optimizationNeeded = true;
      actions.push('cpu-optimization');

      if (strategy.actions.scaleDown) {
        await this.scaleDownCPUIntensiveProcesses();
      }
      if (strategy.actions.redistributeLoad) {
        await this.redistributeCPULoad();
      }
    }

    // Check Memory threshold
    if (metrics.memory.percentage > strategy.thresholds.memoryThreshold) {
      optimizationNeeded = true;
      actions.push('memory-optimization');

      if (strategy.actions.cacheClear) {
        await this.clearCaches();
      }
      if (strategy.actions.scaleDown) {
        await this.scaleDownMemoryIntensiveProcesses();
      }
    }

    // Check GPU threshold
    for (let i = 0; i < metrics.gpu.length; i++) {
      const gpu = metrics.gpu[i];
      if (gpu.memory.percentage > strategy.thresholds.gpuThreshold) {
        optimizationNeeded = true;
        actions.push(`gpu-${i}-optimization`);

        if (strategy.actions.useGPU && this.canOffloadToGPU()) {
          await this.offloadToGPU(i);
        }
      }
    }

    if (optimizationNeeded) {
      this.logger.info('Resource optimization applied', {
        strategy: strategy.name,
        actions,
        metrics: {
          cpu: metrics.cpu.usage,
          memory: metrics.memory.percentage,
          gpu: metrics.gpu.map(g => g.memory.percentage)
        }
      });

      this.emit('optimization-applied', {
        strategy: strategy.name,
        actions,
        timestamp: new Date()
      });
    }
  }

  private async scaleDownCPUIntensiveProcesses(): Promise<void> {
    const cpuIntensiveAllocations = Array.from(this.allocations.values())
      .filter(alloc => alloc.type === 'neural' && alloc.priority !== 'critical')
      .sort((a, b) => {
        const priorityOrder = { low: 0, medium: 1, high: 2, critical: 3 };
        return priorityOrder[a.priority] - priorityOrder[b.priority];
      });

    for (const allocation of cpuIntensiveAllocations.slice(0, 2)) {
      await this.reduceAllocation(allocation.processId, 'cpu');
    }
  }

  private async redistributeCPULoad(): Promise<void> {
    const availableCores = Array.from({ length: this.currentMetrics.cpu.cores }, (_, i) => i);
    const loadPerCore = this.currentMetrics.cpu.load;

    // Find least loaded cores
    const sortedCores = availableCores
      .map(core => ({ core, load: loadPerCore[core] || 0 }))
      .sort((a, b) => a.load - b.load);

    const lowLoadCores = sortedCores.slice(0, Math.ceil(availableCores.length / 2));

    // Redistribute allocations to low-load cores
    for (const allocation of this.allocations.values()) {
      if (allocation.resources.cpuCores && allocation.priority !== 'critical') {
        allocation.resources.cpuCores = lowLoadCores
          .slice(0, allocation.resources.cpuCores.length)
          .map(c => c.core);
      }
    }
  }

  private async scaleDownMemoryIntensiveProcesses(): Promise<void> {
    const memoryIntensiveAllocations = Array.from(this.allocations.values())
      .filter(alloc => alloc.resources.memoryMB && alloc.resources.memoryMB > 1000)
      .sort((a, b) => (b.resources.memoryMB || 0) - (a.resources.memoryMB || 0));

    for (const allocation of memoryIntensiveAllocations.slice(0, 3)) {
      await this.reduceAllocation(allocation.processId, 'memory');
    }
  }

  private async clearCaches(): Promise<void> {
    // Emit cache clear event for components to handle
    this.emit('clear-caches');
    this.logger.info('Cache clear signal sent to components');
  }

  private canOffloadToGPU(): boolean {
    return this.gpu && this.currentMetrics.gpu.length > 0;
  }

  private async offloadToGPU(gpuIndex: number): Promise<void> {
    // Find CPU-bound neural processes that can be moved to GPU
    const neuralAllocations = Array.from(this.allocations.values())
      .filter(alloc => alloc.type === 'neural' && !alloc.resources.gpuIndex);

    for (const allocation of neuralAllocations.slice(0, 1)) {
      allocation.resources.gpuIndex = gpuIndex;
      allocation.resources.gpuMemoryMB = Math.min(1024,
        this.currentMetrics.gpu[gpuIndex].memory.total * 0.3);

      this.logger.info('Process offloaded to GPU', {
        processId: allocation.processId,
        gpuIndex,
        gpuMemoryMB: allocation.resources.gpuMemoryMB
      });
    }
  }

  private async reduceAllocation(processId: string, resourceType: 'cpu' | 'memory'): Promise<void> {
    const allocation = this.allocations.get(processId);
    if (!allocation) return;

    switch (resourceType) {
      case 'cpu':
        if (allocation.resources.cpuCores && allocation.resources.cpuCores.length > 1) {
          allocation.resources.cpuCores = allocation.resources.cpuCores.slice(0, -1);
        }
        break;
      case 'memory':
        if (allocation.resources.memoryMB && allocation.resources.memoryMB > 512) {
          allocation.resources.memoryMB = Math.max(512, allocation.resources.memoryMB * 0.8);
        }
        break;
    }

    this.emit('allocation-reduced', { processId, resourceType, allocation });
    this.logger.info('Resource allocation reduced', { processId, resourceType });
  }

  public allocateResources(processId: string, type: ResourceAllocation['type'], requirements: any): ResourceAllocation {
    const allocation: ResourceAllocation = {
      processId,
      type,
      resources: this.calculateOptimalAllocation(requirements),
      priority: requirements.priority || 'medium',
      startTime: new Date(),
      estimatedDuration: requirements.estimatedDuration
    };

    this.allocations.set(processId, allocation);

    this.logger.info('Resources allocated', { processId, type, resources: allocation.resources });
    this.emit('resource-allocated', allocation);

    return allocation;
  }

  private calculateOptimalAllocation(requirements: any): ResourceAllocation['resources'] {
    const metrics = this.currentMetrics;
    const allocation: ResourceAllocation['resources'] = {};

    // CPU allocation
    if (requirements.cpu) {
      const availableCores = metrics.cpu.cores;
      const requestedCores = Math.min(requirements.cpu.cores || 2, availableCores);

      // Select least loaded cores
      const loadPerCore = metrics.cpu.load;
      const coresByLoad = Array.from({ length: availableCores }, (_, i) => ({
        core: i,
        load: loadPerCore[i] || 0
      })).sort((a, b) => a.load - b.load);

      allocation.cpuCores = coresByLoad.slice(0, requestedCores).map(c => c.core);
    }

    // Memory allocation
    if (requirements.memory) {
      const availableMemory = metrics.memory.available;
      const requestedMemory = Math.min(
        requirements.memory.mb || 1024,
        availableMemory * 0.8 // Don't allocate more than 80% of available
      );
      allocation.memoryMB = requestedMemory;
    }

    // GPU allocation
    if (requirements.gpu && this.canOffloadToGPU()) {
      const bestGPU = this.findBestAvailableGPU();
      if (bestGPU !== -1) {
        allocation.gpuIndex = bestGPU;
        allocation.gpuMemoryMB = Math.min(
          requirements.gpu.memoryMB || 1024,
          metrics.gpu[bestGPU].memory.total * 0.5
        );
      }
    }

    return allocation;
  }

  private findBestAvailableGPU(): number {
    if (this.currentMetrics.gpu.length === 0) return -1;

    // Find GPU with lowest memory usage
    let bestGPU = -1;
    let lowestUsage = 100;

    for (let i = 0; i < this.currentMetrics.gpu.length; i++) {
      const gpu = this.currentMetrics.gpu[i];
      if (gpu.memory.percentage < lowestUsage) {
        lowestUsage = gpu.memory.percentage;
        bestGPU = i;
      }
    }

    return bestGPU;
  }

  public deallocateResources(processId: string): void {
    const allocation = this.allocations.get(processId);
    if (!allocation) {
      this.logger.warn('Attempted to deallocate non-existent process', { processId });
      return;
    }

    this.allocations.delete(processId);

    this.logger.info('Resources deallocated', { processId, allocation });
    this.emit('resource-deallocated', { processId, allocation });
  }

  public getCurrentUsage(): ResourceMetrics {
    return { ...this.currentMetrics };
  }

  public getAllocations(): ResourceAllocation[] {
    return Array.from(this.allocations.values());
  }

  public getOptimizationStrategies(): OptimizationStrategy[] {
    return Array.from(this.strategies.values());
  }

  public updateStrategy(name: string, updates: Partial<OptimizationStrategy>): void {
    const strategy = this.strategies.get(name);
    if (!strategy) {
      this.logger.warn('Attempted to update non-existent strategy', { name });
      return;
    }

    Object.assign(strategy, updates);
    this.logger.info('Optimization strategy updated', { name, updates });
  }

  public getRecommendations(): any {
    const metrics = this.currentMetrics;
    const recommendations = [];

    // CPU recommendations
    if (metrics.cpu.usage > 80) {
      recommendations.push({
        type: 'cpu',
        severity: 'high',
        message: 'High CPU usage detected. Consider scaling down processes or adding more CPU cores.',
        actions: ['scale-down', 'redistribute-load']
      });
    }

    // Memory recommendations
    if (metrics.memory.percentage > 85) {
      recommendations.push({
        type: 'memory',
        severity: 'high',
        message: 'High memory usage detected. Consider clearing caches or adding more RAM.',
        actions: ['clear-cache', 'scale-down-memory']
      });
    }

    // GPU recommendations
    metrics.gpu.forEach((gpu, index) => {
      if (gpu.memory.percentage > 90) {
        recommendations.push({
          type: 'gpu',
          severity: 'critical',
          message: `GPU ${index} memory almost full. Consider offloading some processes to CPU.`,
          actions: ['offload-to-cpu', 'clear-gpu-cache']
        });
      }
    });

    return recommendations;
  }

  public createGPUKernel(kernelFunction: Function, output: number[]): Function {
    if (!this.gpu) {
      throw new Error('GPU not available');
    }

    return this.gpu.createKernel(kernelFunction).setOutput(output);
  }

  public getStatus(): any {
    return {
      monitoring: this.monitoring,
      allocations: this.allocations.size,
      strategies: this.strategies.size,
      lastOptimization: this.lastOptimization,
      gpu: {
        available: this.canOffloadToGPU(),
        count: this.currentMetrics.gpu.length
      }
    };
  }

  public stop(): void {
    this.monitoring = false;

    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }

    this.allocations.clear();
    this.logger.info('Neural resource manager stopped');
  }
}

export { NeuralResourceManager, ResourceMetrics, ResourceAllocation, OptimizationStrategy };