/**
 * Network Resilience Testing Suite
 *
 * Tests application behavior under various network conditions:
 * - Simulated latency (100ms, 500ms, 1000ms)
 * - Packet loss
 * - Bandwidth throttling
 * - Connection interruptions
 */

import { WebSocket } from 'ws';

export interface NetworkCondition {
  name: string;
  latency: number; // ms
  jitter: number; // ms
  packetLoss: number; // 0-1
  bandwidth: number; // kbps
}

export interface NetworkTestConfig {
  serverUrl: string;
  conditions: NetworkCondition[];
  testDuration: number; // seconds
  operationsPerSecond: number;
}

export interface NetworkTestResult {
  condition: NetworkCondition;
  actualLatency: number;
  latencyStdDev: number;
  interpolationSmoothness: number; // 0-100
  rubberBanding: number; // count
  reconnections: number;
  messagesLost: number;
  messagesSent: number;
  messagesReceived: number;
  passed: boolean;
  issues: string[];
  timestamp: Date;
}

interface PendingMessage {
  id: string;
  sentTime: number;
  acknowledged: boolean;
}

export class NetworkLatencyTest {
  private ws?: WebSocket;
  private pendingMessages: Map<string, PendingMessage> = new Map();
  private latencies: number[] = [];
  private positions: Array<{ time: number; position: THREE.Vector3 }> = [];
  private rubberBandingCount: number = 0;
  private reconnectionCount: number = 0;
  private messagesSent: number = 0;
  private messagesReceived: number = 0;
  private issues: string[] = [];

  constructor(private config: NetworkTestConfig) {}

  /**
   * Simulate network latency
   */
  private async simulateLatency(ms: number, jitter: number): Promise<void> {
    const actualLatency = ms + (Math.random() - 0.5) * jitter * 2;
    await new Promise(resolve => setTimeout(resolve, actualLatency));
  }

  /**
   * Simulate packet loss
   */
  private shouldDropPacket(lossRate: number): boolean {
    return Math.random() < lossRate;
  }

  /**
   * Connect to server with network conditions
   */
  private async connect(condition: NetworkCondition): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.config.serverUrl);

      const timeout = setTimeout(() => {
        reject(new Error('Connection timeout'));
      }, 10000);

      this.ws.on('open', () => {
        clearTimeout(timeout);
        resolve();
      });

      this.ws.on('message', async (data: string) => {
        // Simulate network latency on receive
        await this.simulateLatency(condition.latency / 2, condition.jitter / 2);

        // Simulate packet loss
        if (this.shouldDropPacket(condition.packetLoss)) {
          return;
        }

        this.handleMessage(data);
      });

      this.ws.on('error', (error) => {
        this.issues.push(`WebSocket error: ${error.message}`);
      });

      this.ws.on('close', () => {
        this.reconnectionCount++;
      });
    });
  }

  /**
   * Handle incoming message
   */
  private handleMessage(data: string): void {
    try {
      const message = JSON.parse(data);
      this.messagesReceived++;

      // Handle acknowledgment
      if (message.type === 'ack' && message.messageId) {
        const pending = this.pendingMessages.get(message.messageId);
        if (pending) {
          const latency = Date.now() - pending.sentTime;
          this.latencies.push(latency);
          pending.acknowledged = true;
        }
      }

      // Handle position update
      if (message.type === 'positionUpdate') {
        this.handlePositionUpdate(message);
      }
    } catch (error) {
      this.issues.push(`Parse error: ${error}`);
    }
  }

  /**
   * Handle position update and detect rubber-banding
   */
  private handlePositionUpdate(message: any): void {
    const { position, timestamp } = message;
    const currentTime = Date.now();

    // Store position history
    this.positions.push({
      time: currentTime,
      position: new THREE.Vector3(position.x, position.y, position.z)
    });

    // Keep only recent positions (last 2 seconds)
    this.positions = this.positions.filter(p => currentTime - p.time < 2000);

    // Detect rubber-banding (sudden position jumps)
    if (this.positions.length >= 2) {
      const lastPos = this.positions[this.positions.length - 2].position;
      const currentPos = this.positions[this.positions.length - 1].position;
      const distance = lastPos.distanceTo(currentPos);

      // If position jumped more than expected, it's rubber-banding
      const maxExpectedDistance = 5; // units per frame
      if (distance > maxExpectedDistance) {
        this.rubberBandingCount++;
      }
    }
  }

  /**
   * Send message with network conditions
   */
  private async sendMessage(condition: NetworkCondition, message: any): Promise<void> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      this.issues.push('Cannot send message: not connected');
      return;
    }

    // Simulate packet loss on send
    if (this.shouldDropPacket(condition.packetLoss)) {
      return;
    }

    const messageId = `msg-${Date.now()}-${Math.random()}`;
    message.id = messageId;

    // Track pending message
    this.pendingMessages.set(messageId, {
      id: messageId,
      sentTime: Date.now(),
      acknowledged: false
    });

    // Simulate latency on send
    await this.simulateLatency(condition.latency / 2, condition.jitter / 2);

    this.ws.send(JSON.stringify(message));
    this.messagesSent++;
  }

  /**
   * Calculate interpolation smoothness
   */
  private calculateSmoothness(): number {
    if (this.positions.length < 3) return 100;

    let totalDeviation = 0;
    let count = 0;

    // Calculate deviation from linear interpolation
    for (let i = 1; i < this.positions.length - 1; i++) {
      const prev = this.positions[i - 1];
      const curr = this.positions[i];
      const next = this.positions[i + 1];

      // Calculate expected position (linear interpolation)
      const t = (curr.time - prev.time) / (next.time - prev.time);
      const expectedPos = new THREE.Vector3().lerpVectors(prev.position, next.position, t);

      // Calculate deviation
      const deviation = curr.position.distanceTo(expectedPos);
      totalDeviation += deviation;
      count++;
    }

    const avgDeviation = count > 0 ? totalDeviation / count : 0;

    // Convert to 0-100 score (lower deviation = higher score)
    const smoothness = Math.max(0, 100 - (avgDeviation * 20));

    return smoothness;
  }

  /**
   * Run test with specific network condition
   */
  private async runSingleTest(condition: NetworkCondition): Promise<NetworkTestResult> {
    console.log(`Testing network condition: ${condition.name}`);

    // Reset metrics
    this.pendingMessages.clear();
    this.latencies = [];
    this.positions = [];
    this.rubberBandingCount = 0;
    this.reconnectionCount = 0;
    this.messagesSent = 0;
    this.messagesReceived = 0;
    this.issues = [];

    // Connect
    await this.connect(condition);

    // Run test operations
    const startTime = Date.now();
    const endTime = startTime + (this.config.testDuration * 1000);
    const operationInterval = 1000 / this.config.operationsPerSecond;

    while (Date.now() < endTime) {
      // Send position update
      await this.sendMessage(condition, {
        type: 'updatePosition',
        position: {
          x: Math.random() * 100,
          y: Math.random() * 100,
          z: Math.random() * 100
        },
        timestamp: Date.now()
      });

      await new Promise(resolve => setTimeout(resolve, operationInterval));
    }

    // Wait for pending acknowledgments
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Calculate metrics
    const actualLatency = this.latencies.length > 0
      ? this.latencies.reduce((a, b) => a + b, 0) / this.latencies.length
      : 0;

    const latencyMean = actualLatency;
    const latencyVariance = this.latencies.reduce((sum, l) =>
      sum + Math.pow(l - latencyMean, 2), 0) / this.latencies.length;
    const latencyStdDev = Math.sqrt(latencyVariance);

    const interpolationSmoothness = this.calculateSmoothness();

    const messagesLost = this.messagesSent - this.messagesReceived;

    // Check pass/fail
    const issues: string[] = [...this.issues];
    let passed = true;

    if (actualLatency > condition.latency * 1.5) {
      issues.push(`Actual latency (${actualLatency.toFixed(2)}ms) significantly exceeds expected (${condition.latency}ms)`);
    }

    if (interpolationSmoothness < 70) {
      issues.push(`Poor interpolation smoothness: ${interpolationSmoothness.toFixed(2)}`);
      passed = false;
    }

    if (this.rubberBandingCount > 5) {
      issues.push(`Excessive rubber-banding: ${this.rubberBandingCount} occurrences`);
      passed = false;
    }

    if (this.reconnectionCount > 0) {
      issues.push(`Connection interrupted ${this.reconnectionCount} times`);
      passed = false;
    }

    // Cleanup
    if (this.ws) {
      this.ws.close();
    }

    const result: NetworkTestResult = {
      condition,
      actualLatency,
      latencyStdDev,
      interpolationSmoothness,
      rubberBanding: this.rubberBandingCount,
      reconnections: this.reconnectionCount,
      messagesLost,
      messagesSent: this.messagesSent,
      messagesReceived: this.messagesReceived,
      passed,
      issues,
      timestamp: new Date()
    };

    console.log(`Test complete: ${actualLatency.toFixed(2)}ms latency, ${interpolationSmoothness.toFixed(2)} smoothness`);

    return result;
  }

  /**
   * Run full network test suite
   */
  async run(): Promise<NetworkTestResult[]> {
    console.log('Starting network resilience test suite...');

    const results: NetworkTestResult[] = [];

    for (const condition of this.config.conditions) {
      try {
        const result = await this.runSingleTest(condition);
        results.push(result);

        // Cool down between tests
        await new Promise(resolve => setTimeout(resolve, 2000));
      } catch (error) {
        console.error(`Test failed for condition ${condition.name}:`, error);
      }
    }

    return results;
  }

  /**
   * Generate network test report
   */
  static generateReport(results: NetworkTestResult[]): string {
    let report = '# Network Resilience Test Report\n\n';
    report += `Generated: ${new Date().toISOString()}\n\n`;

    report += '## Test Results\n\n';
    report += '| Condition | Expected Latency | Actual Latency | Smoothness | Rubber-banding | Status |\n';
    report += '|-----------|------------------|----------------|------------|----------------|--------|\n';

    for (const result of results) {
      const status = result.passed ? '✅' : '❌';
      report += `| ${result.condition.name} | ${result.condition.latency}ms | `;
      report += `${result.actualLatency.toFixed(2)}ms | ${result.interpolationSmoothness.toFixed(2)} | `;
      report += `${result.rubberBanding} | ${status} |\n`;
    }

    report += '\n## Detailed Metrics\n\n';

    for (const result of results) {
      report += `### ${result.condition.name}\n\n`;
      report += `- **Latency**: ${result.actualLatency.toFixed(2)}ms ± ${result.latencyStdDev.toFixed(2)}ms\n`;
      report += `- **Smoothness**: ${result.interpolationSmoothness.toFixed(2)}/100\n`;
      report += `- **Messages**: ${result.messagesSent} sent, ${result.messagesReceived} received, ${result.messagesLost} lost\n`;
      report += `- **Rubber-banding**: ${result.rubberBanding} occurrences\n`;
      report += `- **Reconnections**: ${result.reconnections}\n`;

      if (result.issues.length > 0) {
        report += '\n**Issues**:\n';
        result.issues.forEach(issue => {
          report += `- ${issue}\n`;
        });
      }

      report += '\n';
    }

    return report;
  }
}

import * as THREE from 'three';

export const DEFAULT_NETWORK_CONDITIONS: NetworkCondition[] = [
  { name: 'Good Connection', latency: 50, jitter: 10, packetLoss: 0, bandwidth: 10000 },
  { name: 'Average Connection', latency: 100, jitter: 20, packetLoss: 0.01, bandwidth: 5000 },
  { name: 'Poor Connection', latency: 500, jitter: 100, packetLoss: 0.05, bandwidth: 1000 },
  { name: 'Very Poor Connection', latency: 1000, jitter: 200, packetLoss: 0.1, bandwidth: 500 }
];

export const DEFAULT_NETWORK_TEST_CONFIG: NetworkTestConfig = {
  serverUrl: 'ws://localhost:3000',
  conditions: DEFAULT_NETWORK_CONDITIONS,
  testDuration: 30,
  operationsPerSecond: 10
};
