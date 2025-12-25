/**
 * Graph Performance Benchmark Suite
 *
 * Automated FPS measurement and performance profiling for large graphs.
 * Tests rendering performance at various node counts with detailed metrics.
 */

import * as THREE from 'three';
import { NodeManager } from '../../core/NodeManager';
import { EdgeManager } from '../../core/EdgeManager';
import { GraphRenderer } from '../../rendering/GraphRenderer';
import { Node, Edge } from '../../types/graph';

export interface BenchmarkResult {
  nodeCount: number;
  edgeCount: number;
  avgFps: number;
  minFps: number;
  maxFps: number;
  avgFrameTime: number;
  p99FrameTime: number;
  gcPauses: number;
  memoryUsage: number;
  renderTime: number;
  updateTime: number;
  timestamp: Date;
}

export interface BenchmarkConfig {
  duration: number; // Test duration in seconds
  nodeCounts: number[];
  edgeDensity: number; // Edges per node
  warmupFrames: number;
  collectGCMetrics: boolean;
}

export class GraphBenchmark {
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  private nodeManager!: NodeManager;
  private edgeManager!: EdgeManager;
  private graphRenderer!: GraphRenderer;

  private frameTimes: number[] = [];
  private renderTimes: number[] = [];
  private updateTimes: number[] = [];
  private gcPauseCount: number = 0;
  private lastFrameTime: number = 0;

  private performanceObserver?: PerformanceObserver;

  constructor(private config: BenchmarkConfig) {}

  /**
   * Initialize Three.js scene and components for testing
   */
  private async initialize(): Promise<void> {
    // Create offscreen canvas for headless testing
    const canvas = document.createElement('canvas');
    canvas.width = 1920;
    canvas.height = 1080;

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: false,
      powerPreference: 'high-performance'
    });
    this.renderer.setSize(1920, 1080);
    this.renderer.setPixelRatio(1);

    this.scene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera(75, 1920 / 1080, 0.1, 10000);
    this.camera.position.z = 100;

    this.nodeManager = new NodeManager();
    this.edgeManager = new EdgeManager();
    this.graphRenderer = new GraphRenderer(this.scene, this.nodeManager, this.edgeManager);

    await this.graphRenderer.initialize();
  }

  /**
   * Generate synthetic graph for testing
   */
  private generateTestGraph(nodeCount: number): void {
    const nodes: Node[] = [];
    const edges: Edge[] = [];

    // Generate nodes in 3D space
    for (let i = 0; i < nodeCount; i++) {
      const angle = (i / nodeCount) * Math.PI * 2;
      const radius = Math.sqrt(i) * 10;
      const height = (Math.random() - 0.5) * 50;

      nodes.push({
        id: `node-${i}`,
        label: `Node ${i}`,
        position: {
          x: Math.cos(angle) * radius,
          y: height,
          z: Math.sin(angle) * radius
        },
        metadata: {
          type: i % 3 === 0 ? 'primary' : 'secondary',
          weight: Math.random()
        }
      });
    }

    // Generate edges based on density
    const edgesPerNode = Math.floor(this.config.edgeDensity);
    for (let i = 0; i < nodeCount; i++) {
      for (let j = 0; j < edgesPerNode; j++) {
        const targetIndex = (i + j + 1) % nodeCount;
        edges.push({
          id: `edge-${i}-${targetIndex}`,
          source: `node-${i}`,
          target: `node-${targetIndex}`,
          metadata: {
            weight: Math.random()
          }
        });
      }
    }

    // Load graph into managers
    this.nodeManager.clear();
    this.edgeManager.clear();

    nodes.forEach(node => this.nodeManager.addNode(node));
    edges.forEach(edge => this.edgeManager.addEdge(edge));

    this.graphRenderer.refresh();
  }

  /**
   * Setup performance monitoring
   */
  private setupPerformanceMonitoring(): void {
    if (!this.config.collectGCMetrics) return;

    // Monitor GC pauses via performance timeline
    this.performanceObserver = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.entryType === 'measure' && entry.name.includes('gc')) {
          this.gcPauseCount++;
        }
      }
    });

    try {
      this.performanceObserver.observe({ entryTypes: ['measure'] });
    } catch (e) {
      console.warn('GC monitoring not available');
    }
  }

  /**
   * Run benchmark for specific node count
   */
  private async runSingleBenchmark(nodeCount: number): Promise<BenchmarkResult> {
    console.log(`Starting benchmark with ${nodeCount} nodes...`);

    // Reset metrics
    this.frameTimes = [];
    this.renderTimes = [];
    this.updateTimes = [];
    this.gcPauseCount = 0;

    // Generate test graph
    this.generateTestGraph(nodeCount);

    // Warmup
    for (let i = 0; i < this.config.warmupFrames; i++) {
      this.graphRenderer.update(16.67);
      this.renderer.render(this.scene, this.camera);
    }

    // Run benchmark
    const startTime = performance.now();
    const endTime = startTime + (this.config.duration * 1000);
    let frameCount = 0;

    this.lastFrameTime = startTime;

    while (performance.now() < endTime) {
      const frameStart = performance.now();

      // Update phase
      const updateStart = performance.now();
      this.graphRenderer.update(frameStart - this.lastFrameTime);
      const updateEnd = performance.now();
      this.updateTimes.push(updateEnd - updateStart);

      // Render phase
      const renderStart = performance.now();
      this.renderer.render(this.scene, this.camera);
      const renderEnd = performance.now();
      this.renderTimes.push(renderEnd - renderStart);

      const frameEnd = performance.now();
      const frameTime = frameEnd - frameStart;
      this.frameTimes.push(frameTime);

      this.lastFrameTime = frameStart;
      frameCount++;

      // Yield to event loop
      await new Promise(resolve => setTimeout(resolve, 0));
    }

    const totalTime = performance.now() - startTime;

    // Calculate metrics
    const avgFps = (frameCount / totalTime) * 1000;
    const fps = this.frameTimes.map(t => 1000 / t);
    const minFps = Math.min(...fps);
    const maxFps = Math.max(...fps);
    const avgFrameTime = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;

    // Calculate p99
    const sortedFrameTimes = [...this.frameTimes].sort((a, b) => a - b);
    const p99Index = Math.floor(sortedFrameTimes.length * 0.99);
    const p99FrameTime = sortedFrameTimes[p99Index];

    // Memory usage
    const memoryUsage = (performance as any).memory?.usedJSHeapSize || 0;

    const avgRenderTime = this.renderTimes.reduce((a, b) => a + b, 0) / this.renderTimes.length;
    const avgUpdateTime = this.updateTimes.reduce((a, b) => a + b, 0) / this.updateTimes.length;

    const result: BenchmarkResult = {
      nodeCount,
      edgeCount: this.edgeManager.getEdges().length,
      avgFps,
      minFps,
      maxFps,
      avgFrameTime,
      p99FrameTime,
      gcPauses: this.gcPauseCount,
      memoryUsage,
      renderTime: avgRenderTime,
      updateTime: avgUpdateTime,
      timestamp: new Date()
    };

    console.log(`Benchmark complete: ${avgFps.toFixed(2)} FPS (min: ${minFps.toFixed(2)}, max: ${maxFps.toFixed(2)})`);

    return result;
  }

  /**
   * Run full benchmark suite
   */
  async run(): Promise<BenchmarkResult[]> {
    console.log('Initializing benchmark suite...');

    await this.initialize();
    this.setupPerformanceMonitoring();

    const results: BenchmarkResult[] = [];

    for (const nodeCount of this.config.nodeCounts) {
      try {
        const result = await this.runSingleBenchmark(nodeCount);
        results.push(result);

        // Cool down between tests
        await new Promise(resolve => setTimeout(resolve, 1000));
      } catch (error) {
        console.error(`Benchmark failed for ${nodeCount} nodes:`, error);
      }
    }

    this.cleanup();

    return results;
  }

  /**
   * Cleanup resources
   */
  private cleanup(): void {
    this.performanceObserver?.disconnect();
    this.graphRenderer.dispose();
    this.renderer.dispose();
  }

  /**
   * Generate benchmark report
   */
  static generateReport(results: BenchmarkResult[]): string {
    let report = '# Graph Performance Benchmark Report\n\n';
    report += `Generated: ${new Date().toISOString()}\n\n`;

    report += '## Results Summary\n\n';
    report += '| Nodes | Edges | Avg FPS | Min FPS | P99 Frame Time | Memory (MB) | Render (ms) | Update (ms) |\n';
    report += '|-------|-------|---------|---------|----------------|-------------|-------------|-------------|\n';

    for (const result of results) {
      const memoryMB = (result.memoryUsage / 1024 / 1024).toFixed(2);
      report += `| ${result.nodeCount} | ${result.edgeCount} | `;
      report += `${result.avgFps.toFixed(2)} | ${result.minFps.toFixed(2)} | `;
      report += `${result.p99FrameTime.toFixed(2)}ms | ${memoryMB} | `;
      report += `${result.renderTime.toFixed(2)} | ${result.updateTime.toFixed(2)} |\n`;
    }

    report += '\n## Performance Analysis\n\n';

    // Check for performance issues
    const failedTests = results.filter(r => r.avgFps < 60);
    if (failedTests.length > 0) {
      report += '### ⚠️ Performance Issues Detected\n\n';
      failedTests.forEach(r => {
        report += `- **${r.nodeCount} nodes**: ${r.avgFps.toFixed(2)} FPS (target: 60 FPS)\n`;
      });
      report += '\n';
    } else {
      report += '### ✅ All Tests Passed\n\n';
      report += 'All configurations maintained 60+ FPS.\n\n';
    }

    // GC analysis
    const gcIssues = results.filter(r => r.gcPauses > 10);
    if (gcIssues.length > 0) {
      report += '### ⚠️ Garbage Collection Issues\n\n';
      gcIssues.forEach(r => {
        report += `- **${r.nodeCount} nodes**: ${r.gcPauses} GC pauses detected\n`;
      });
      report += '\n';
    }

    return report;
  }
}

/**
 * Default benchmark configuration
 */
export const DEFAULT_BENCHMARK_CONFIG: BenchmarkConfig = {
  duration: 10,
  nodeCounts: [100, 500, 1000, 5000],
  edgeDensity: 3,
  warmupFrames: 60,
  collectGCMetrics: true
};

/**
 * Run benchmark suite with default config
 */
export async function runDefaultBenchmark(): Promise<void> {
  const benchmark = new GraphBenchmark(DEFAULT_BENCHMARK_CONFIG);
  const results = await benchmark.run();

  const report = GraphBenchmark.generateReport(results);
  console.log(report);

  // Save results
  const resultsJson = JSON.stringify(results, null, 2);
  console.log('\n## Raw Results (JSON)\n');
  console.log(resultsJson);
}
