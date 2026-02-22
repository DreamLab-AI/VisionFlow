

export class AudioContextManager {
  private static instance: AudioContextManager;
  private audioContext: AudioContext | null = null;
  private initialized: boolean = false;

  private constructor() {}

  static getInstance(): AudioContextManager {
    if (!AudioContextManager.instance) {
      AudioContextManager.instance = new AudioContextManager();
    }
    return AudioContextManager.instance;
  }

  
  getContext(): AudioContext {
    if (!this.audioContext) {
      interface WindowWithWebkit extends Window { webkitAudioContext?: typeof AudioContext; }
      const AudioCtx = window.AudioContext || (window as WindowWithWebkit).webkitAudioContext;
      this.audioContext = new AudioCtx!();
      this.initialized = true;
    }
    return this.audioContext;
  }

  
  async resume(): Promise<void> {
    const ctx = this.getContext();
    if (ctx.state === 'suspended') {
      await ctx.resume();
    }
  }

  
  async suspend(): Promise<void> {
    if (this.audioContext && this.audioContext.state === 'running') {
      await this.audioContext.suspend();
    }
  }

  
  async close(): Promise<void> {
    if (this.audioContext) {
      await this.audioContext.close();
      this.audioContext = null;
      this.initialized = false;
    }
  }

  
  getState(): AudioContextState | null {
    return this.audioContext ? this.audioContext.state : null;
  }

  
  isInitialized(): boolean {
    return this.initialized;
  }
}