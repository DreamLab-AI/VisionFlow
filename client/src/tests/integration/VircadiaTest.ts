/**
 * Vircadia Integration Testing Suite
 *
 * Verifies Vircadia features work correctly with Three.js renderer:
 * - Avatar synchronization
 * - Presence indicators
 * - Collaboration features
 * - Audio/video integration
 * - Domain coordination
 */

import { WebSocket } from 'ws';
import * as THREE from 'three';

export interface VircadiaTestConfig {
  domainUrl: string;
  username: string;
  testDuration: number; // seconds
}

export interface VircadiaTestResult {
  avatarSyncWorking: boolean;
  presenceAccurate: boolean;
  collaborationFunctional: boolean;
  audioWorking: boolean;
  domainConnectionStable: boolean;
  threeJsCompatible: boolean;
  issues: string[];
  details: {
    avatarUpdateLatency: number;
    presenceUpdateFrequency: number;
    collaborationEvents: number;
    audioPacketsReceived: number;
    reconnections: number;
  };
  timestamp: Date;
  passed: boolean;
}

interface Avatar {
  id: string;
  displayName: string;
  position: THREE.Vector3;
  rotation: THREE.Quaternion;
  lastUpdate: number;
}

interface PresenceInfo {
  userId: string;
  online: boolean;
  location: string;
  lastSeen: number;
}

export class VircadiaIntegrationTest {
  private ws?: WebSocket;
  private scene!: THREE.Scene;
  private avatars: Map<string, Avatar> = new Map();
  private presenceData: Map<string, PresenceInfo> = new Map();
  private avatarMeshes: Map<string, THREE.Mesh> = new Map();

  private avatarUpdateLatencies: number[] = [];
  private presenceUpdates: number = 0;
  private collaborationEvents: number = 0;
  private audioPackets: number = 0;
  private reconnections: number = 0;
  private issues: string[] = [];

  constructor(private config: VircadiaTestConfig) {}

  /**
   * Initialize Three.js scene for testing
   */
  private initializeScene(): void {
    this.scene = new THREE.Scene();

    // Add ground plane
    const groundGeometry = new THREE.PlaneGeometry(100, 100);
    const groundMaterial = new THREE.MeshStandardMaterial({ color: 0x808080 });
    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2;
    this.scene.add(ground);

    // Add lighting
    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(5, 10, 5);
    this.scene.add(light as any);
    this.scene.add(new THREE.AmbientLight(0x404040) as any);
  }

  /**
   * Connect to Vircadia domain
   */
  private async connectToDomain(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.config.domainUrl);

      const timeout = setTimeout(() => {
        reject(new Error('Domain connection timeout'));
      }, 10000);

      this.ws.on('open', () => {
        clearTimeout(timeout);

        // Send authentication
        this.ws!.send(JSON.stringify({
          type: 'authenticate',
          username: this.config.username,
          timestamp: Date.now()
        }));

        resolve();
      });

      this.ws.on('message', (data: string) => {
        this.handleDomainMessage(data);
      });

      this.ws.on('error', (error) => {
        this.issues.push(`Domain connection error: ${error.message}`);
      });

      this.ws.on('close', () => {
        this.reconnections++;
      });
    });
  }

  /**
   * Handle message from Vircadia domain
   */
  private handleDomainMessage(data: string): void {
    try {
      const message = JSON.parse(data);

      switch (message.type) {
        case 'avatarUpdate':
          this.handleAvatarUpdate(message);
          break;
        case 'presenceUpdate':
          this.handlePresenceUpdate(message);
          break;
        case 'collaborationEvent':
          this.handleCollaborationEvent(message);
          break;
        case 'audioData':
          this.handleAudioData(message);
          break;
      }
    } catch (error) {
      this.issues.push(`Message parse error: ${error}`);
    }
  }

  /**
   * Handle avatar position/rotation update
   */
  private handleAvatarUpdate(message: any): void {
    const { avatarId, displayName, position, rotation, timestamp } = message;

    // Calculate latency
    const latency = Date.now() - timestamp;
    this.avatarUpdateLatencies.push(latency);

    // Update or create avatar
    let avatar = this.avatars.get(avatarId);

    if (!avatar) {
      avatar = {
        id: avatarId,
        displayName,
        position: new THREE.Vector3(),
        rotation: new THREE.Quaternion(),
        lastUpdate: Date.now()
      };
      this.avatars.set(avatarId, avatar);
      this.createAvatarMesh(avatarId);
    }

    // Update avatar state
    avatar.position.set(position.x, position.y, position.z);
    avatar.rotation.set(rotation.x, rotation.y, rotation.z, rotation.w);
    avatar.lastUpdate = Date.now();

    // Update Three.js mesh
    this.updateAvatarMesh(avatarId, avatar);
  }

  /**
   * Create Three.js mesh for avatar
   */
  private createAvatarMesh(avatarId: string): void {
    // Create simple avatar representation
    const group = new THREE.Group();

    // Body (capsule approximation)
    const bodyGeometry = new THREE.CylinderGeometry(0.3, 0.3, 1.6, 16);
    const bodyMaterial = new THREE.MeshStandardMaterial({ color: 0x3498db });
    const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
    body.position.y = 0.8;
    group.add(body);

    // Head
    const headGeometry = new THREE.SphereGeometry(0.2, 16, 16);
    const headMaterial = new THREE.MeshStandardMaterial({ color: 0xffd700 });
    const head = new THREE.Mesh(headGeometry, headMaterial);
    head.position.y = 1.8;
    group.add(head);

    this.scene.add(group);
    this.avatarMeshes.set(avatarId, group as any);
  }

  /**
   * Update avatar mesh position/rotation
   */
  private updateAvatarMesh(avatarId: string, avatar: Avatar): void {
    const mesh = this.avatarMeshes.get(avatarId);
    if (!mesh) return;

    mesh.position.copy(avatar.position);
    mesh.quaternion.copy(avatar.rotation);
  }

  /**
   * Handle presence update
   */
  private handlePresenceUpdate(message: any): void {
    const { userId, online, location } = message;

    this.presenceData.set(userId, {
      userId,
      online,
      location,
      lastSeen: Date.now()
    });

    this.presenceUpdates++;
  }

  /**
   * Handle collaboration event (object creation, editing, etc.)
   */
  private handleCollaborationEvent(message: any): void {
    this.collaborationEvents++;

    // Could test specific collaboration features here
    const { eventType, data } = message;

    switch (eventType) {
      case 'objectCreated':
        // Verify object appears in scene
        break;
      case 'objectModified':
        // Verify object updates correctly
        break;
      case 'objectDeleted':
        // Verify object removed from scene
        break;
    }
  }

  /**
   * Handle audio data packet
   */
  private handleAudioData(message: any): void {
    this.audioPackets++;
    // Could test spatial audio positioning here
  }

  /**
   * Test avatar synchronization
   */
  private async testAvatarSync(): Promise<boolean> {
    // Send avatar updates
    for (let i = 0; i < 10; i++) {
      this.ws!.send(JSON.stringify({
        type: 'updateAvatar',
        position: {
          x: Math.random() * 10,
          y: 0,
          z: Math.random() * 10
        },
        rotation: {
          x: 0,
          y: Math.random() * Math.PI * 2,
          z: 0,
          w: 1
        },
        timestamp: Date.now()
      }));

      await new Promise(resolve => setTimeout(resolve, 100));
    }

    // Wait for responses
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Check if updates were received
    if (this.avatarUpdateLatencies.length === 0) {
      this.issues.push('No avatar updates received');
      return false;
    }

    const avgLatency = this.avatarUpdateLatencies.reduce((a, b) => a + b, 0) / this.avatarUpdateLatencies.length;

    if (avgLatency > 500) {
      this.issues.push(`High avatar sync latency: ${avgLatency.toFixed(2)}ms`);
      return false;
    }

    return true;
  }

  /**
   * Test presence indicators
   */
  private async testPresence(): Promise<boolean> {
    // Request presence info
    this.ws!.send(JSON.stringify({
      type: 'getPresence',
      timestamp: Date.now()
    }));

    await new Promise(resolve => setTimeout(resolve, 2000));

    if (this.presenceUpdates === 0) {
      this.issues.push('No presence updates received');
      return false;
    }

    // Verify presence data is accurate
    let accurate = true;
    this.presenceData.forEach((presence, userId) => {
      if (Date.now() - presence.lastSeen > 5000) {
        this.issues.push(`Stale presence data for user ${userId}`);
        accurate = false;
      }
    });

    return accurate;
  }

  /**
   * Test collaboration features
   */
  private async testCollaboration(): Promise<boolean> {
    // Create test object
    this.ws!.send(JSON.stringify({
      type: 'createObject',
      objectType: 'cube',
      position: { x: 0, y: 1, z: 0 },
      timestamp: Date.now()
    }));

    await new Promise(resolve => setTimeout(resolve, 1000));

    if (this.collaborationEvents === 0) {
      this.issues.push('No collaboration events received');
      return false;
    }

    return true;
  }

  /**
   * Test audio integration
   */
  private async testAudio(): Promise<boolean> {
    // Enable audio stream
    this.ws!.send(JSON.stringify({
      type: 'enableAudio',
      timestamp: Date.now()
    }));

    await new Promise(resolve => setTimeout(resolve, 3000));

    if (this.audioPackets === 0) {
      this.issues.push('No audio packets received');
      return false;
    }

    return true;
  }

  /**
   * Test Three.js compatibility
   */
  private testThreeJsCompatibility(): boolean {
    // Verify avatars are properly rendered
    if (this.avatarMeshes.size === 0) {
      this.issues.push('No avatar meshes created in Three.js scene');
      return false;
    }

    // Verify scene structure
    if (this.scene.children.length === 0) {
      this.issues.push('Three.js scene is empty');
      return false;
    }

    // All Vircadia objects should be Three.js compatible
    return true;
  }

  /**
   * Run full Vircadia integration test
   */
  async run(): Promise<VircadiaTestResult> {
    console.log('Starting Vircadia integration test...');

    // Reset metrics
    this.avatars.clear();
    this.presenceData.clear();
    this.avatarMeshes.clear();
    this.avatarUpdateLatencies = [];
    this.presenceUpdates = 0;
    this.collaborationEvents = 0;
    this.audioPackets = 0;
    this.reconnections = 0;
    this.issues = [];

    // Initialize
    this.initializeScene();
    await this.connectToDomain();

    // Run tests
    const avatarSyncWorking = await this.testAvatarSync();
    const presenceAccurate = await this.testPresence();
    const collaborationFunctional = await this.testCollaboration();
    const audioWorking = await this.testAudio();
    const threeJsCompatible = this.testThreeJsCompatibility();
    const domainConnectionStable = this.reconnections === 0;

    // Calculate metrics
    const avgAvatarLatency = this.avatarUpdateLatencies.length > 0
      ? this.avatarUpdateLatencies.reduce((a, b) => a + b, 0) / this.avatarUpdateLatencies.length
      : 0;

    const presenceFrequency = this.presenceUpdates / this.config.testDuration;

    const passed = avatarSyncWorking && presenceAccurate && collaborationFunctional &&
                   audioWorking && threeJsCompatible && domainConnectionStable;

    // Cleanup
    if (this.ws) {
      this.ws.close();
    }

    const result: VircadiaTestResult = {
      avatarSyncWorking,
      presenceAccurate,
      collaborationFunctional,
      audioWorking,
      domainConnectionStable,
      threeJsCompatible,
      issues: [...this.issues],
      details: {
        avatarUpdateLatency: avgAvatarLatency,
        presenceUpdateFrequency: presenceFrequency,
        collaborationEvents: this.collaborationEvents,
        audioPacketsReceived: this.audioPackets,
        reconnections: this.reconnections
      },
      timestamp: new Date(),
      passed
    };

    console.log(`Vircadia integration test ${passed ? 'PASSED' : 'FAILED'}`);

    return result;
  }

  /**
   * Generate Vircadia test report
   */
  static generateReport(result: VircadiaTestResult): string {
    let report = '# Vircadia Integration Test Report\n\n';
    report += `Generated: ${result.timestamp.toISOString()}\n\n`;

    report += `## Overall Result: ${result.passed ? '✅ PASSED' : '❌ FAILED'}\n\n`;

    report += '## Feature Tests\n\n';
    report += '| Feature | Status |\n';
    report += '|---------|--------|\n';
    report += `| Avatar Synchronization | ${result.avatarSyncWorking ? '✅' : '❌'} |\n`;
    report += `| Presence Indicators | ${result.presenceAccurate ? '✅' : '❌'} |\n`;
    report += `| Collaboration Features | ${result.collaborationFunctional ? '✅' : '❌'} |\n`;
    report += `| Audio Integration | ${result.audioWorking ? '✅' : '❌'} |\n`;
    report += `| Domain Connection | ${result.domainConnectionStable ? '✅' : '❌'} |\n`;
    report += `| Three.js Compatibility | ${result.threeJsCompatible ? '✅' : '❌'} |\n`;

    report += '\n## Performance Details\n\n';
    report += `- **Avatar Update Latency**: ${result.details.avatarUpdateLatency.toFixed(2)}ms\n`;
    report += `- **Presence Update Frequency**: ${result.details.presenceUpdateFrequency.toFixed(2)} updates/sec\n`;
    report += `- **Collaboration Events**: ${result.details.collaborationEvents}\n`;
    report += `- **Audio Packets**: ${result.details.audioPacketsReceived}\n`;
    report += `- **Reconnections**: ${result.details.reconnections}\n`;

    if (result.issues.length > 0) {
      report += '\n## ⚠️ Issues Detected\n\n';
      result.issues.forEach(issue => {
        report += `- ${issue}\n`;
      });
    }

    return report;
  }
}

export const DEFAULT_VIRCADIA_CONFIG: VircadiaTestConfig = {
  domainUrl: 'ws://localhost:40102',
  username: 'test-user',
  testDuration: 60
};
