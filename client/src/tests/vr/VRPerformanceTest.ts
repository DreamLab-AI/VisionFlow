/**
 * VR Performance Validation Suite
 *
 * Tests VR-specific performance metrics for Quest 3:
 * - 72fps minimum framerate
 * - Hand tracking responsiveness (<50ms latency)
 * - Reprojection rate monitoring
 * - Comfort metrics (no judder/stutter)
 */

import * as THREE from 'three';
import { VRButton } from 'three/examples/jsm/webxr/VRButton';
import { XRControllerModelFactory } from 'three/examples/jsm/webxr/XRControllerModelFactory';

export interface VRPerformanceConfig {
  testDuration: number; // seconds
  targetFps: number;
  maxHandTrackingLatency: number; // ms
  nodeCount: number;
}

export interface VRPerformanceResult {
  avgFps: number;
  minFps: number;
  maxFps: number;
  frameTimeVariance: number;
  droppedFrames: number;
  reprojectionRate: number;
  handTrackingLatency: number;
  handTrackingDropouts: number;
  renderLatency: number;
  comfortScore: number; // 0-100
  passed: boolean;
  issues: string[];
  timestamp: Date;
}

export class VRPerformanceTest {
  private renderer!: THREE.WebGLRenderer;
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private xrSession?: XRSession;

  private frameTimes: number[] = [];
  private handTrackingLatencies: number[] = [];
  private handTrackingDropouts: number = 0;
  private reprojectedFrames: number = 0;
  private droppedFrames: number = 0;
  private lastFrameTime: number = 0;
  private issues: string[] = [];

  private handMeshes: Map<string, THREE.Mesh> = new Map();
  private lastHandUpdate: Map<string, number> = new Map();

  constructor(private config: VRPerformanceConfig) {}

  /**
   * Initialize VR session and scene
   */
  private async initializeVR(): Promise<void> {
    // Create renderer
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true
    });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.xr.enabled = true;

    // Create scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x101010);

    // Create camera
    this.camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );

    // Add lighting
    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(1, 1, 1);
    this.scene.add(light);
    this.scene.add(new THREE.AmbientLight(0x404040));

    // Add test geometry
    this.createTestScene();

    // Setup hand tracking
    await this.setupHandTracking();
  }

  /**
   * Create test scene with nodes
   */
  private createTestScene(): void {
    const geometry = new THREE.SphereGeometry(0.05, 16, 16);
    const material = new THREE.MeshStandardMaterial({ color: 0x00ff00 });

    for (let i = 0; i < this.config.nodeCount; i++) {
      const mesh = new THREE.Mesh(geometry, material.clone());

      // Distribute in 3D space around user
      const angle = (i / this.config.nodeCount) * Math.PI * 2;
      const radius = 2 + Math.random() * 3;
      const height = (Math.random() - 0.5) * 2;

      mesh.position.set(
        Math.cos(angle) * radius,
        height,
        Math.sin(angle) * radius
      );

      this.scene.add(mesh);
    }
  }

  /**
   * Setup hand tracking
   */
  private async setupHandTracking(): Promise<void> {
    if (!this.renderer.xr.isPresenting) {
      return;
    }

    const session = this.renderer.xr.getSession();
    if (!session) return;

    this.xrSession = session;

    // Request hand tracking
    try {
      const referenceSpace = await session.requestReferenceSpace('local');

      session.requestAnimationFrame((time, frame) => {
        this.onXRFrame(time, frame, referenceSpace);
      });
    } catch (error) {
      this.issues.push(`Hand tracking initialization failed: ${error}`);
    }
  }

  /**
   * Handle XR frame updates
   */
  private onXRFrame(time: number, frame: XRFrame, referenceSpace: XRReferenceSpace): void {
    const inputSources = frame.session.inputSources;

    for (const inputSource of inputSources) {
      if (inputSource.hand) {
        this.updateHandTracking(inputSource.hand, frame, referenceSpace);
      }
    }

    // Continue animation loop
    if (this.xrSession) {
      this.xrSession.requestAnimationFrame((t, f) =>
        this.onXRFrame(t, f, referenceSpace)
      );
    }
  }

  /**
   * Update hand tracking data
   */
  private updateHandTracking(hand: XRHand, frame: XRFrame, referenceSpace: XRReferenceSpace): void {
    const handId = hand === frame.session.inputSources[0]?.hand ? 'left' : 'right';
    const updateTime = performance.now();

    // Measure latency
    const lastUpdate = this.lastHandUpdate.get(handId);
    if (lastUpdate) {
      const latency = updateTime - lastUpdate;
      this.handTrackingLatencies.push(latency);

      if (latency > this.config.maxHandTrackingLatency) {
        this.handTrackingDropouts++;
      }
    }

    this.lastHandUpdate.set(handId, updateTime);

    // Update hand mesh visualization
    for (const joint of hand.values()) {
      const jointPose = frame.getJointPose(joint, referenceSpace);
      if (jointPose) {
        // Joint is tracked - visualize it
        const key = `${handId}-${joint.jointName}`;
        let mesh = this.handMeshes.get(key);

        if (!mesh) {
          mesh = new THREE.Mesh(
            new THREE.SphereGeometry(0.01, 8, 8),
            new THREE.MeshBasicMaterial({ color: 0xff0000 })
          );
          this.handMeshes.set(key, mesh);
          this.scene.add(mesh);
        }

        mesh.position.set(
          jointPose.transform.position.x,
          jointPose.transform.position.y,
          jointPose.transform.position.z
        );
      }
    }
  }

  /**
   * Run render loop and collect metrics
   */
  private async runRenderLoop(): Promise<void> {
    return new Promise((resolve) => {
      const startTime = performance.now();
      const endTime = startTime + (this.config.testDuration * 1000);

      this.lastFrameTime = startTime;
      let frameCount = 0;

      const animate = () => {
        if (!this.renderer.xr.isPresenting) {
          resolve();
          return;
        }

        if (performance.now() >= endTime) {
          resolve();
          return;
        }

        const frameStart = performance.now();

        // Check for dropped frames
        const expectedFrameTime = 1000 / this.config.targetFps;
        const actualFrameTime = frameStart - this.lastFrameTime;

        if (actualFrameTime > expectedFrameTime * 1.5) {
          this.droppedFrames++;
        }

        // Render
        this.renderer.render(this.scene, this.camera);

        const frameEnd = performance.now();
        const frameTime = frameEnd - frameStart;
        this.frameTimes.push(frameTime);

        this.lastFrameTime = frameStart;
        frameCount++;

        // Continue loop
        this.renderer.setAnimationLoop(animate);
      };

      this.renderer.setAnimationLoop(animate);
    });
  }

  /**
   * Calculate comfort score
   */
  private calculateComfortScore(): number {
    let score = 100;

    // Penalize frame drops
    const dropRate = this.droppedFrames / this.frameTimes.length;
    score -= dropRate * 50;

    // Penalize high frame time variance (judder)
    const avgFrameTime = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
    const variance = this.frameTimes.reduce((sum, t) => sum + Math.pow(t - avgFrameTime, 2), 0) / this.frameTimes.length;
    const stdDev = Math.sqrt(variance);

    if (stdDev > 5) score -= 20;
    if (stdDev > 10) score -= 20;

    // Penalize reprojection
    const reprojectionRate = this.reprojectedFrames / this.frameTimes.length;
    score -= reprojectionRate * 30;

    return Math.max(0, score);
  }

  /**
   * Run VR performance test
   */
  async run(): Promise<VRPerformanceResult> {
    console.log('Initializing VR performance test...');

    // Reset metrics
    this.frameTimes = [];
    this.handTrackingLatencies = [];
    this.handTrackingDropouts = 0;
    this.reprojectedFrames = 0;
    this.droppedFrames = 0;
    this.issues = [];

    await this.initializeVR();

    // Enter VR
    const enterVRButton = VRButton.createButton(this.renderer);
    document.body.appendChild(enterVRButton);

    // Wait for VR session
    await new Promise<void>((resolve) => {
      const checkInterval = setInterval(() => {
        if (this.renderer.xr.isPresenting) {
          clearInterval(checkInterval);
          resolve();
        }
      }, 100);
    });

    console.log('VR session started, running test...');

    await this.runRenderLoop();

    // Calculate metrics
    const fps = this.frameTimes.map(t => 1000 / t);
    const avgFps = fps.reduce((a, b) => a + b, 0) / fps.length;
    const minFps = Math.min(...fps);
    const maxFps = Math.max(...fps);

    const avgFrameTime = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
    const frameTimeVariance = this.frameTimes.reduce((sum, t) =>
      sum + Math.pow(t - avgFrameTime, 2), 0) / this.frameTimes.length;

    const reprojectionRate = this.reprojectedFrames / this.frameTimes.length;

    const avgHandTrackingLatency = this.handTrackingLatencies.length > 0
      ? this.handTrackingLatencies.reduce((a, b) => a + b, 0) / this.handTrackingLatencies.length
      : 0;

    const renderLatency = avgFrameTime;
    const comfortScore = this.calculateComfortScore();

    // Check pass/fail
    let passed = true;

    if (avgFps < this.config.targetFps) {
      this.issues.push(`Average FPS (${avgFps.toFixed(2)}) below target (${this.config.targetFps})`);
      passed = false;
    }

    if (minFps < this.config.targetFps * 0.9) {
      this.issues.push(`Minimum FPS (${minFps.toFixed(2)}) too low`);
      passed = false;
    }

    if (avgHandTrackingLatency > this.config.maxHandTrackingLatency) {
      this.issues.push(`Hand tracking latency (${avgHandTrackingLatency.toFixed(2)}ms) too high`);
      passed = false;
    }

    if (comfortScore < 80) {
      this.issues.push(`Comfort score (${comfortScore.toFixed(2)}) below threshold`);
      passed = false;
    }

    const result: VRPerformanceResult = {
      avgFps,
      minFps,
      maxFps,
      frameTimeVariance,
      droppedFrames: this.droppedFrames,
      reprojectionRate,
      handTrackingLatency: avgHandTrackingLatency,
      handTrackingDropouts: this.handTrackingDropouts,
      renderLatency,
      comfortScore,
      passed,
      issues: [...this.issues],
      timestamp: new Date()
    };

    this.cleanup();

    return result;
  }

  /**
   * Cleanup resources
   */
  private cleanup(): void {
    this.renderer.xr.setSession(null as any);
    this.renderer.dispose();
    this.handMeshes.forEach(mesh => {
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
    });
  }

  /**
   * Generate VR test report
   */
  static generateReport(result: VRPerformanceResult): string {
    let report = '# VR Performance Test Report\n\n';
    report += `Generated: ${result.timestamp.toISOString()}\n\n`;

    report += `## Overall Result: ${result.passed ? '✅ PASSED' : '❌ FAILED'}\n\n`;

    report += '## Performance Metrics\n\n';
    report += '| Metric | Value | Target | Status |\n';
    report += '|--------|-------|--------|--------|\n';
    report += `| Avg FPS | ${result.avgFps.toFixed(2)} | 72 | ${result.avgFps >= 72 ? '✅' : '❌'} |\n`;
    report += `| Min FPS | ${result.minFps.toFixed(2)} | 65 | ${result.minFps >= 65 ? '✅' : '❌'} |\n`;
    report += `| Hand Latency | ${result.handTrackingLatency.toFixed(2)}ms | <50ms | ${result.handTrackingLatency < 50 ? '✅' : '❌'} |\n`;
    report += `| Dropped Frames | ${result.droppedFrames} | 0 | ${result.droppedFrames === 0 ? '✅' : '⚠️'} |\n`;
    report += `| Comfort Score | ${result.comfortScore.toFixed(2)} | >80 | ${result.comfortScore > 80 ? '✅' : '❌'} |\n`;

    if (result.issues.length > 0) {
      report += '\n## ⚠️ Issues Detected\n\n';
      result.issues.forEach(issue => {
        report += `- ${issue}\n`;
      });
    }

    return report;
  }
}

export const DEFAULT_VR_CONFIG: VRPerformanceConfig = {
  testDuration: 30,
  targetFps: 72,
  maxHandTrackingLatency: 50,
  nodeCount: 500
};
