

import * as THREE from 'three';
import { createLogger } from './loggerConfig';

const logger = createLogger('DualGraphPerformanceMonitor');

export interface GraphPerformanceMetrics {
  nodeCount: number;
  edgeCount: number;
  updateTime: number;
  renderTime: number;
  physicsTime: number;
  instancedRendering: boolean;
  visibleNodes: number;
  culledNodes: number;
}

export interface PerformanceMetrics {
  fps: number;
  frameTime: number;
  frameTimeMin: number;
  frameTimeMax: number;
  memory: {
    used: number;
    limit: number;
    percent: number;
  };
  webgl: {
    drawCalls: number;
    triangles: number;
    points: number;
    lines: number;
    programs: number;
    textures: number;
    geometries: number;
  };
  graphMetrics: {
    logseq: GraphPerformanceMetrics;
    visionflow: GraphPerformanceMetrics;
  };
  workerMetrics: {
    physicsWorker: {
      messagesSent: number;
      messagesReceived: number;
      avgResponseTime: number;
    };
  };
}

class DualGraphPerformanceMonitor {
  private metrics: PerformanceMetrics;
  private frameCount = 0;
  private frameStartTime = 0;
  private fpsUpdateInterval = 500; 
  private lastFpsUpdate = 0;
  private frameTimeSamples: number[] = [];
  private maxSamples = 60;
  
  
  private gl: WebGLRenderingContext | WebGL2RenderingContext | null = null;
  private extDisjointTimerQuery: any = null;
  
  
  private performanceMarks = new Map<string, number>();
  
  
  private workerMessageTimes = new Map<string, number>();
  
  constructor() {
    this.metrics = this.initializeMetrics();
  }

  private initializeMetrics(): PerformanceMetrics {
    return {
      fps: 0,
      frameTime: 0,
      frameTimeMin: Infinity,
      frameTimeMax: 0,
      memory: {
        used: 0,
        limit: 0,
        percent: 0
      },
      webgl: {
        drawCalls: 0,
        triangles: 0,
        points: 0,
        lines: 0,
        programs: 0,
        textures: 0,
        geometries: 0
      },
      graphMetrics: {
        logseq: {
          nodeCount: 0,
          edgeCount: 0,
          updateTime: 0,
          renderTime: 0,
          physicsTime: 0,
          instancedRendering: true,
          visibleNodes: 0,
          culledNodes: 0
        },
        visionflow: {
          nodeCount: 0,
          edgeCount: 0,
          updateTime: 0,
          renderTime: 0,
          physicsTime: 0,
          instancedRendering: false,
          visibleNodes: 0,
          culledNodes: 0
        }
      },
      workerMetrics: {
        physicsWorker: {
          messagesSent: 0,
          messagesReceived: 0,
          avgResponseTime: 0
        }
      }
    };
  }

  
  public initializeWebGL(renderer: THREE.WebGLRenderer) {
    this.gl = renderer.getContext();
    
    
    if (this.gl) {
      this.extDisjointTimerQuery = 
        this.gl.getExtension('EXT_disjoint_timer_query_webgl2') ||
        this.gl.getExtension('EXT_disjoint_timer_query');
    }
    
    
    renderer.info.autoReset = false;
    
    logger.info('WebGL monitoring initialized', {
      hasTimerQuery: !!this.extDisjointTimerQuery,
      maxTextureSize: this.gl?.getParameter(this.gl.MAX_TEXTURE_SIZE),
      maxVertexUniforms: this.gl?.getParameter(this.gl.MAX_VERTEX_UNIFORM_VECTORS),
      maxFragmentUniforms: this.gl?.getParameter(this.gl.MAX_FRAGMENT_UNIFORM_VECTORS)
    });
  }

  
  public mark(name: string) {
    this.performanceMarks.set(name, performance.now());
  }

  
  public measure(name: string): number {
    const startTime = this.performanceMarks.get(name);
    if (!startTime) return 0;
    
    const duration = performance.now() - startTime;
    this.performanceMarks.delete(name);
    return duration;
  }

  
  public beginFrame() {
    this.frameStartTime = performance.now();
    this.mark('frame');
  }

  
  public endFrame(renderer?: THREE.WebGLRenderer) {
    const frameTime = this.measure('frame');
    
    
    this.frameTimeSamples.push(frameTime);
    if (this.frameTimeSamples.length > this.maxSamples) {
      this.frameTimeSamples.shift();
    }
    
    
    const avgFrameTime = this.frameTimeSamples.reduce((a, b) => a + b, 0) / this.frameTimeSamples.length;
    this.metrics.frameTime = Math.round(avgFrameTime * 100) / 100;
    this.metrics.frameTimeMin = Math.min(...this.frameTimeSamples);
    this.metrics.frameTimeMax = Math.max(...this.frameTimeSamples);
    
    
    this.frameCount++;
    const now = performance.now();
    if (now - this.lastFpsUpdate >= this.fpsUpdateInterval) {
      const elapsed = now - this.lastFpsUpdate;
      this.metrics.fps = Math.round((this.frameCount / elapsed) * 1000);
      this.frameCount = 0;
      this.lastFpsUpdate = now;
      
      
      this.updateMemoryMetrics();
      if (renderer) {
        this.updateWebGLStats(renderer);
      }
    }
  }

  
  private updateMemoryMetrics() {
    if ('memory' in performance) {
      const memory = (performance as any).memory;
      this.metrics.memory = {
        used: Math.round(memory.usedJSHeapSize / 1048576), 
        limit: Math.round(memory.jsHeapSizeLimit / 1048576), 
        percent: Math.round((memory.usedJSHeapSize / memory.jsHeapSizeLimit) * 100)
      };
    }
  }

  
  public updateWebGLStats(renderer: THREE.WebGLRenderer) {
    const info = renderer.info;
    
    this.metrics.webgl = {
      drawCalls: info.render.calls,
      triangles: info.render.triangles,
      points: info.render.points,
      lines: info.render.lines,
      programs: info.programs?.length || 0,
      textures: info.memory.textures,
      geometries: info.memory.geometries
    };
    
    
    info.reset();
  }

  
  public updateGraphMetrics(graphType: 'logseq' | 'visionflow', metrics: Partial<GraphPerformanceMetrics>) {
    Object.assign(this.metrics.graphMetrics[graphType], metrics);
  }

  
  public trackWorkerMessage(workerId: string, type: 'sent' | 'received') {
    const key = `${workerId}_${type}`;
    
    if (type === 'sent') {
      this.workerMessageTimes.set(key, performance.now());
      this.metrics.workerMetrics.physicsWorker.messagesSent++;
    } else {
      const sentTime = this.workerMessageTimes.get(`${workerId}_sent`);
      if (sentTime) {
        const responseTime = performance.now() - sentTime;
        const metrics = this.metrics.workerMetrics.physicsWorker;
        metrics.messagesReceived++;
        
        
        const total = metrics.avgResponseTime * (metrics.messagesReceived - 1) + responseTime;
        metrics.avgResponseTime = total / metrics.messagesReceived;
        
        this.workerMessageTimes.delete(`${workerId}_sent`);
      }
    }
  }

  
  public getMetrics(): Readonly<PerformanceMetrics> {
    return { ...this.metrics };
  }

  
  public generateReport(): string {
    const m = this.metrics;
    const report = [
      '=== Dual Graph Performance Report ===',
      `FPS: ${m.fps} | Frame: ${m.frameTime}ms (${m.frameTimeMin.toFixed(1)}-${m.frameTimeMax.toFixed(1)}ms)`,
      `Memory: ${m.memory.used}MB / ${m.memory.limit}MB (${m.memory.percent}%)`,
      '',
      '--- WebGL Stats ---',
      `Draw Calls: ${m.webgl.drawCalls} | Programs: ${m.webgl.programs}`,
      `Triangles: ${m.webgl.triangles} | Points: ${m.webgl.points} | Lines: ${m.webgl.lines}`,
      `Textures: ${m.webgl.textures} | Geometries: ${m.webgl.geometries}`,
      '',
      '--- Logseq Graph ---',
      `Nodes: ${m.graphMetrics.logseq.nodeCount} (${m.graphMetrics.logseq.visibleNodes} visible, ${m.graphMetrics.logseq.culledNodes} culled)`,
      `Edges: ${m.graphMetrics.logseq.edgeCount}`,
      `Instanced: ${m.graphMetrics.logseq.instancedRendering ? '✅' : '❌'}`,
      `Update: ${m.graphMetrics.logseq.updateTime.toFixed(1)}ms | Render: ${m.graphMetrics.logseq.renderTime.toFixed(1)}ms | Physics: ${m.graphMetrics.logseq.physicsTime.toFixed(1)}ms`,
      '',
      '--- VisionFlow Graph ---',
      `Nodes: ${m.graphMetrics.visionflow.nodeCount} (${m.graphMetrics.visionflow.visibleNodes} visible, ${m.graphMetrics.visionflow.culledNodes} culled)`,
      `Edges: ${m.graphMetrics.visionflow.edgeCount}`,
      `Instanced: ${m.graphMetrics.visionflow.instancedRendering ? '✅' : '❌'}`,
      `Update: ${m.graphMetrics.visionflow.updateTime.toFixed(1)}ms | Render: ${m.graphMetrics.visionflow.renderTime.toFixed(1)}ms | Physics: ${m.graphMetrics.visionflow.physicsTime.toFixed(1)}ms`,
      '',
      '--- Worker Performance ---',
      `Physics Worker: ${m.workerMetrics.physicsWorker.messagesReceived} messages, ${m.workerMetrics.physicsWorker.avgResponseTime.toFixed(1)}ms avg response`,
      '',
      '--- Recommendations ---'
    ];

    
    const recommendations = this.getPerformanceRecommendations();
    report.push(...recommendations);

    return report.join('\n');
  }

  
  private getPerformanceRecommendations(): string[] {
    const recommendations: string[] = [];
    const m = this.metrics;

    
    if (m.fps < 30) {
      recommendations.push('⚠️ Low FPS detected:');
      
      if (!m.graphMetrics.visionflow.instancedRendering && m.graphMetrics.visionflow.nodeCount > 20) {
        recommendations.push('  - Enable instanced rendering for VisionFlow (currently using individual meshes)');
      }
      
      if (m.frameTimeMax > 33) {
        recommendations.push('  - Frame time spikes detected, consider profiling with Chrome DevTools');
      }
      
      recommendations.push('  - Reduce particle effects and ambient animations');
      recommendations.push('  - Implement Level of Detail (LOD) for distant nodes');
    }

    
    if (m.memory.percent > 80) {
      recommendations.push('⚠️ High memory usage:');
      recommendations.push('  - Dispose unused geometries and materials');
      recommendations.push('  - Implement node culling for off-screen elements');
      recommendations.push('  - Consider using BufferGeometry.dispose() on hidden graphs');
    }

    
    if (m.webgl.drawCalls > 300) {
      recommendations.push('⚠️ High draw call count:');
      
      const totalNodes = m.graphMetrics.logseq.nodeCount + m.graphMetrics.visionflow.nodeCount;
      const instancedNodes = 
        (m.graphMetrics.logseq.instancedRendering ? m.graphMetrics.logseq.nodeCount : 0) +
        (m.graphMetrics.visionflow.instancedRendering ? m.graphMetrics.visionflow.nodeCount : 0);
      
      if (instancedNodes < totalNodes) {
        recommendations.push(`  - Only ${instancedNodes}/${totalNodes} nodes use instanced rendering`);
      }
      
      recommendations.push('  - Merge edge geometries where possible');
      recommendations.push('  - Use texture atlases for node icons/sprites');
    }

    
    const totalNodes = m.graphMetrics.logseq.nodeCount + m.graphMetrics.visionflow.nodeCount;
    if (totalNodes > 1000) {
      recommendations.push('⚠️ Large node count optimization needed:');
      recommendations.push('  - Implement spatial partitioning (octree/BVH)');
      recommendations.push('  - Add frustum culling with THREE.Frustum');
      recommendations.push('  - Consider SharedArrayBuffer for worker communication');
      
      const culledRatio = (m.graphMetrics.logseq.culledNodes + m.graphMetrics.visionflow.culledNodes) / totalNodes;
      if (culledRatio < 0.2) {
        recommendations.push('  - Low culling ratio, improve visibility testing');
      }
    }

    
    if (m.graphMetrics.logseq.physicsTime + m.graphMetrics.visionflow.physicsTime > 10) {
      recommendations.push('⚠️ Physics performance issues:');
      recommendations.push('  - Consider spatial hashing for collision detection');
      recommendations.push('  - Reduce physics update frequency for distant nodes');
      recommendations.push('  - Use fixed timestep for physics simulation');
    }

    
    if (m.workerMetrics.physicsWorker.avgResponseTime > 16) {
      recommendations.push('⚠️ Worker communication bottleneck:');
      recommendations.push('  - Consider SharedArrayBuffer for zero-copy communication');
      recommendations.push('  - Batch worker messages to reduce overhead');
      recommendations.push('  - Implement worker message prioritization');
    }

    if (recommendations.length === 0) {
      recommendations.push('✅ Performance is excellent!');
      recommendations.push(`  - Maintaining ${m.fps} FPS with ${totalNodes} total nodes`);
      recommendations.push(`  - Draw calls optimized at ${m.webgl.drawCalls}`);
      recommendations.push(`  - Memory usage healthy at ${m.memory.percent}%`);
    }

    return recommendations;
  }

  
  public logReport() {
    console.log(this.generateReport());
  }

  
  public getPerformanceScore(): number {
    const m = this.metrics;
    
    
    const fpsScore = Math.min(m.fps / 60, 1) * 30; 
    const frameTimeScore = Math.max(0, 1 - (m.frameTime / 16.67)) * 20; 
    const memoryScore = Math.max(0, 1 - (m.memory.percent / 100)) * 20; 
    const drawCallScore = Math.max(0, 1 - (m.webgl.drawCalls / 500)) * 20; 
    const workerScore = Math.max(0, 1 - (m.workerMetrics.physicsWorker.avgResponseTime / 16)) * 10; 
    
    return Math.round(fpsScore + frameTimeScore + memoryScore + drawCallScore + workerScore);
  }

  
  public reset() {
    this.metrics = this.initializeMetrics();
    this.frameCount = 0;
    this.frameTimeSamples = [];
    this.performanceMarks.clear();
    this.workerMessageTimes.clear();
  }

  
  public exportMetrics(): string {
    return JSON.stringify(this.metrics, null, 2);
  }

  
  public dispose() {
    this.gl = null;
    this.extDisjointTimerQuery = null;
    this.reset();
  }
}

// Singleton instance
export const dualGraphPerformanceMonitor = new DualGraphPerformanceMonitor();

// Export to window for debugging
if (typeof window !== 'undefined' && process.env.NODE_ENV === 'development') {
  (window as any).dualGraphPerformanceMonitor = dualGraphPerformanceMonitor;
}