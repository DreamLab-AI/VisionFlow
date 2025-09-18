import { createLogger } from '../../../utils/logger';

const logger = createLogger('PollingPerformance');

export interface PollingMetrics {
  pollCount: number;
  successCount: number;
  errorCount: number;
  averageDuration: number;
  minDuration: number;
  maxDuration: number;
  dataChangeCount: number;
  lastPollTime: number;
  uptimeMs: number;
}

export class PollingPerformanceMonitor {
  private metrics: PollingMetrics = {
    pollCount: 0,
    successCount: 0,
    errorCount: 0,
    averageDuration: 0,
    minDuration: Infinity,
    maxDuration: 0,
    dataChangeCount: 0,
    lastPollTime: 0,
    uptimeMs: 0
  };
  
  private startTime: number = Date.now();
  private durations: number[] = [];
  private maxDurationHistory = 100;

  /**
   * Record a successful poll
   */
  recordPoll(duration: number, dataChanged: boolean): void {
    this.metrics.pollCount++;
    this.metrics.successCount++;
    this.metrics.lastPollTime = Date.now();
    
    if (dataChanged) {
      this.metrics.dataChangeCount++;
    }
    
    // Update duration statistics
    this.durations.push(duration);
    if (this.durations.length > this.maxDurationHistory) {
      this.durations.shift();
    }
    
    this.metrics.minDuration = Math.min(this.metrics.minDuration, duration);
    this.metrics.maxDuration = Math.max(this.metrics.maxDuration, duration);
    this.metrics.averageDuration = this.durations.reduce((a, b) => a + b, 0) / this.durations.length;
    
    // Log performance warnings
    if (duration > 1000) {
      logger.warn(`Slow poll detected: ${duration}ms`);
    }
  }

  /**
   * Record a failed poll
   */
  recordError(): void {
    this.metrics.pollCount++;
    this.metrics.errorCount++;
  }

  /**
   * Get current metrics
   */
  getMetrics(): PollingMetrics {
    return {
      ...this.metrics,
      uptimeMs: Date.now() - this.startTime
    };
  }

  /**
   * Get success rate
   */
  getSuccessRate(): number {
    if (this.metrics.pollCount === 0) return 1;
    return this.metrics.successCount / this.metrics.pollCount;
  }

  /**
   * Get data freshness (time since last successful poll)
   */
  getDataFreshness(): number {
    if (this.metrics.lastPollTime === 0) return Infinity;
    return Date.now() - this.metrics.lastPollTime;
  }

  /**
   * Reset metrics
   */
  reset(): void {
    this.metrics = {
      pollCount: 0,
      successCount: 0,
      errorCount: 0,
      averageDuration: 0,
      minDuration: Infinity,
      maxDuration: 0,
      dataChangeCount: 0,
      lastPollTime: 0,
      uptimeMs: 0
    };
    this.durations = [];
    this.startTime = Date.now();
  }

  /**
   * Get performance summary
   */
  getSummary(): string {
    const successRate = (this.getSuccessRate() * 100).toFixed(1);
    const changeRate = this.metrics.pollCount > 0 
      ? ((this.metrics.dataChangeCount / this.metrics.pollCount) * 100).toFixed(1)
      : '0.0';
    
    return `Polls: ${this.metrics.pollCount} | Success: ${successRate}% | Changes: ${changeRate}% | Avg: ${this.metrics.averageDuration.toFixed(0)}ms`;
  }
}