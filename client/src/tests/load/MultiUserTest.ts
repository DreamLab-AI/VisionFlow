/**
 * Multi-User Load Testing Suite
 *
 * Simulates concurrent users interacting with the graph to test:
 * - WebSocket connection handling
 * - Position convergence time
 * - Conflict resolution
 * - Server scalability
 */

import { WebSocket } from 'ws';
import { EventEmitter } from 'events';

export interface LoadTestConfig {
  serverUrl: string;
  userCounts: number[];
  testDuration: number; // seconds
  operationsPerSecond: number;
  nodeCount: number;
}

export interface LoadTestResult {
  userCount: number;
  successfulConnections: number;
  failedConnections: number;
  avgLatency: number;
  p99Latency: number;
  avgConvergenceTime: number;
  conflictsDetected: number;
  conflictsResolved: number;
  messagesPerSecond: number;
  errors: string[];
  timestamp: Date;
}

interface UserSession {
  id: string;
  ws: WebSocket;
  connected: boolean;
  latencies: number[];
  lastUpdateTime: number;
  nodeStates: Map<string, any>;
}

export class MultiUserLoadTest extends EventEmitter {
  private sessions: UserSession[] = [];
  private latencies: number[] = [];
  private convergenceTimes: number[] = [];
  private conflicts: number = 0;
  private conflictsResolved: number = 0;
  private errors: string[] = [];
  private messageCount: number = 0;

  constructor(private config: LoadTestConfig) {
    super();
  }

  /**
   * Create a simulated user session
   */
  private async createUserSession(userId: string): Promise<UserSession> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this.config.serverUrl);
      const session: UserSession = {
        id: userId,
        ws,
        connected: false,
        latencies: [],
        lastUpdateTime: Date.now(),
        nodeStates: new Map()
      };

      const timeout = setTimeout(() => {
        reject(new Error(`Connection timeout for user ${userId}`));
      }, 10000);

      ws.on('open', () => {
        clearTimeout(timeout);
        session.connected = true;

        // Send authentication/join message
        ws.send(JSON.stringify({
          type: 'join',
          userId,
          timestamp: Date.now()
        }));

        resolve(session);
      });

      ws.on('message', (data: string) => {
        this.handleMessage(session, data);
      });

      ws.on('error', (error) => {
        this.errors.push(`User ${userId}: ${error.message}`);
      });

      ws.on('close', () => {
        session.connected = false;
      });
    });
  }

  /**
   * Handle incoming message from server
   */
  private handleMessage(session: UserSession, data: string): void {
    try {
      const message = JSON.parse(data);
      this.messageCount++;

      // Calculate latency
      if (message.timestamp) {
        const latency = Date.now() - message.timestamp;
        session.latencies.push(latency);
        this.latencies.push(latency);
      }

      // Handle different message types
      switch (message.type) {
        case 'nodeUpdate':
          this.handleNodeUpdate(session, message);
          break;
        case 'conflict':
          this.conflicts++;
          break;
        case 'conflictResolved':
          this.conflictsResolved++;
          break;
      }
    } catch (error) {
      this.errors.push(`Parse error: ${error}`);
    }
  }

  /**
   * Handle node update and check for convergence
   */
  private handleNodeUpdate(session: UserSession, message: any): void {
    const { nodeId, position, version, timestamp } = message;

    const previousState = session.nodeStates.get(nodeId);

    if (previousState && previousState.version !== version) {
      // State changed - measure convergence time
      const convergenceTime = Date.now() - previousState.updateTime;
      this.convergenceTimes.push(convergenceTime);
    }

    session.nodeStates.set(nodeId, {
      position,
      version,
      updateTime: Date.now()
    });
  }

  /**
   * Simulate user operations
   */
  private async simulateUserOperations(session: UserSession, duration: number): Promise<void> {
    const endTime = Date.now() + (duration * 1000);
    const operationInterval = 1000 / this.config.operationsPerSecond;

    while (Date.now() < endTime && session.connected) {
      // Randomly select operation type
      const operation = Math.random();

      if (operation < 0.5) {
        // Move node
        this.sendNodeUpdate(session);
      } else if (operation < 0.8) {
        // Select node
        this.sendNodeSelection(session);
      } else {
        // Create edge
        this.sendEdgeCreation(session);
      }

      await new Promise(resolve => setTimeout(resolve, operationInterval));
    }
  }

  /**
   * Send node position update
   */
  private sendNodeUpdate(session: UserSession): void {
    if (!session.connected) return;

    const nodeId = `node-${Math.floor(Math.random() * this.config.nodeCount)}`;

    session.ws.send(JSON.stringify({
      type: 'updateNode',
      userId: session.id,
      nodeId,
      position: {
        x: (Math.random() - 0.5) * 200,
        y: (Math.random() - 0.5) * 200,
        z: (Math.random() - 0.5) * 200
      },
      timestamp: Date.now()
    }));
  }

  /**
   * Send node selection event
   */
  private sendNodeSelection(session: UserSession): void {
    if (!session.connected) return;

    const nodeId = `node-${Math.floor(Math.random() * this.config.nodeCount)}`;

    session.ws.send(JSON.stringify({
      type: 'selectNode',
      userId: session.id,
      nodeId,
      timestamp: Date.now()
    }));
  }

  /**
   * Send edge creation request
   */
  private sendEdgeCreation(session: UserSession): void {
    if (!session.connected) return;

    const sourceId = `node-${Math.floor(Math.random() * this.config.nodeCount)}`;
    const targetId = `node-${Math.floor(Math.random() * this.config.nodeCount)}`;

    if (sourceId === targetId) return;

    session.ws.send(JSON.stringify({
      type: 'createEdge',
      userId: session.id,
      sourceId,
      targetId,
      timestamp: Date.now()
    }));
  }

  /**
   * Run load test with specific user count
   */
  private async runSingleLoadTest(userCount: number): Promise<LoadTestResult> {
    console.log(`Starting load test with ${userCount} concurrent users...`);

    // Reset metrics
    this.sessions = [];
    this.latencies = [];
    this.convergenceTimes = [];
    this.conflicts = 0;
    this.conflictsResolved = 0;
    this.errors = [];
    this.messageCount = 0;

    const startTime = Date.now();

    // Create user sessions
    const connectionPromises: Promise<UserSession>[] = [];
    for (let i = 0; i < userCount; i++) {
      connectionPromises.push(
        this.createUserSession(`user-${i}`)
          .catch(error => {
            this.errors.push(error.message);
            return null as any;
          })
      );
    }

    const sessions = await Promise.all(connectionPromises);
    this.sessions = sessions.filter(s => s !== null);

    const successfulConnections = this.sessions.length;
    const failedConnections = userCount - successfulConnections;

    console.log(`Connected ${successfulConnections}/${userCount} users`);

    if (successfulConnections === 0) {
      throw new Error('All connections failed');
    }

    // Start user operations
    const operationPromises = this.sessions.map(session =>
      this.simulateUserOperations(session, this.config.testDuration)
    );

    await Promise.all(operationPromises);

    const testDuration = (Date.now() - startTime) / 1000;

    // Calculate metrics
    const avgLatency = this.latencies.length > 0
      ? this.latencies.reduce((a, b) => a + b, 0) / this.latencies.length
      : 0;

    const sortedLatencies = [...this.latencies].sort((a, b) => a - b);
    const p99Index = Math.floor(sortedLatencies.length * 0.99);
    const p99Latency = sortedLatencies.length > 0 ? sortedLatencies[p99Index] : 0;

    const avgConvergenceTime = this.convergenceTimes.length > 0
      ? this.convergenceTimes.reduce((a, b) => a + b, 0) / this.convergenceTimes.length
      : 0;

    const messagesPerSecond = this.messageCount / testDuration;

    // Cleanup
    this.sessions.forEach(session => {
      if (session.ws.readyState === WebSocket.OPEN) {
        session.ws.close();
      }
    });

    const result: LoadTestResult = {
      userCount,
      successfulConnections,
      failedConnections,
      avgLatency,
      p99Latency,
      avgConvergenceTime,
      conflictsDetected: this.conflicts,
      conflictsResolved: this.conflictsResolved,
      messagesPerSecond,
      errors: [...this.errors],
      timestamp: new Date()
    };

    console.log(`Load test complete: ${messagesPerSecond.toFixed(2)} msg/s, latency: ${avgLatency.toFixed(2)}ms`);

    return result;
  }

  /**
   * Run full load test suite
   */
  async run(): Promise<LoadTestResult[]> {
    console.log('Starting multi-user load test suite...');

    const results: LoadTestResult[] = [];

    for (const userCount of this.config.userCounts) {
      try {
        const result = await this.runSingleLoadTest(userCount);
        results.push(result);

        // Cool down between tests
        await new Promise(resolve => setTimeout(resolve, 5000));
      } catch (error) {
        console.error(`Load test failed for ${userCount} users:`, error);
        this.errors.push(`Test failed: ${error}`);
      }
    }

    return results;
  }

  /**
   * Generate load test report
   */
  static generateReport(results: LoadTestResult[]): string {
    let report = '# Multi-User Load Test Report\n\n';
    report += `Generated: ${new Date().toISOString()}\n\n`;

    report += '## Connection Statistics\n\n';
    report += '| Users | Connected | Failed | Success Rate |\n';
    report += '|-------|-----------|--------|-------------|\n';

    for (const result of results) {
      const successRate = ((result.successfulConnections / result.userCount) * 100).toFixed(1);
      report += `| ${result.userCount} | ${result.successfulConnections} | `;
      report += `${result.failedConnections} | ${successRate}% |\n`;
    }

    report += '\n## Performance Metrics\n\n';
    report += '| Users | Avg Latency | P99 Latency | Convergence Time | Messages/sec |\n';
    report += '|-------|-------------|-------------|------------------|-------------|\n';

    for (const result of results) {
      report += `| ${result.userCount} | ${result.avgLatency.toFixed(2)}ms | `;
      report += `${result.p99Latency.toFixed(2)}ms | ${result.avgConvergenceTime.toFixed(2)}ms | `;
      report += `${result.messagesPerSecond.toFixed(2)} |\n`;
    }

    report += '\n## Conflict Resolution\n\n';
    report += '| Users | Conflicts Detected | Conflicts Resolved | Resolution Rate |\n';
    report += '|-------|-------------------|-------------------|----------------|\n';

    for (const result of results) {
      const resolutionRate = result.conflictsDetected > 0
        ? ((result.conflictsResolved / result.conflictsDetected) * 100).toFixed(1)
        : 'N/A';
      report += `| ${result.userCount} | ${result.conflictsDetected} | `;
      report += `${result.conflictsResolved} | ${resolutionRate}% |\n`;
    }

    // Issues
    const issuesFound: string[] = [];
    results.forEach(r => {
      if (r.failedConnections > 0) {
        issuesFound.push(`${r.userCount} users: ${r.failedConnections} connection failures`);
      }
      if (r.avgLatency > 200) {
        issuesFound.push(`${r.userCount} users: High latency (${r.avgLatency.toFixed(2)}ms)`);
      }
      if (r.avgConvergenceTime > 1000) {
        issuesFound.push(`${r.userCount} users: Slow convergence (${r.avgConvergenceTime.toFixed(2)}ms)`);
      }
    });

    if (issuesFound.length > 0) {
      report += '\n## ⚠️ Issues Detected\n\n';
      issuesFound.forEach(issue => {
        report += `- ${issue}\n`;
      });
    } else {
      report += '\n## ✅ All Tests Passed\n\n';
      report += 'All load tests completed successfully.\n';
    }

    return report;
  }
}

export const DEFAULT_LOAD_TEST_CONFIG: LoadTestConfig = {
  serverUrl: 'ws://localhost:3000',
  userCounts: [10, 50, 100],
  testDuration: 30,
  operationsPerSecond: 5,
  nodeCount: 1000
};
