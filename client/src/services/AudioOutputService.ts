

import { AudioContextManager } from './AudioContextManager';
import { gatedConsole } from '../utils/console';

export interface AudioQueueItem {
  id: string;
  buffer: ArrayBuffer;
  timestamp: number;
  metadata?: any;
}

export type AudioOutputState = 'idle' | 'playing' | 'paused' | 'buffering';

export class AudioOutputService {
  private static instance: AudioOutputService;
  private audioContext: AudioContext;
  private playbackQueue: AudioQueueItem[] = [];
  private currentSource: AudioBufferSourceNode | null = null;
  private gainNode: GainNode;
  private state: AudioOutputState = 'idle';
  private listeners: Map<string, Set<Function>> = new Map();
  private isProcessing = false;
  private stopRequested = false;
  private volume = 1.0;

  private constructor() {
    this.audioContext = AudioContextManager.getInstance().getContext();
    this.gainNode = this.audioContext.createGain();
    this.gainNode.connect(this.audioContext.destination);
  }

  static getInstance(): AudioOutputService {
    if (!AudioOutputService.instance) {
      AudioOutputService.instance = new AudioOutputService();
    }
    return AudioOutputService.instance;
  }

  
  async queueAudio(audioData: ArrayBuffer, id?: string): Promise<void> {
    const item: AudioQueueItem = {
      id: id || Date.now().toString(),
      buffer: audioData,
      timestamp: Date.now()
    };

    this.playbackQueue.push(item);
    this.emit('audioQueued', item);

    
    if (!this.isProcessing) {
      this.processQueue();
    }
  }

  
  private async processQueue() {
    if (this.isProcessing || this.playbackQueue.length === 0) {
      return;
    }

    this.isProcessing = true;
    this.stopRequested = false;
    this.setState('buffering');

    while (this.playbackQueue.length > 0 && this.state !== 'paused') {
      const item = this.playbackQueue.shift()!;
      
      try {
        await this.playAudioBuffer(item);
      } catch (error) {
        gatedConsole.voice.error('Error playing audio:', error);
        this.emit('error', { item, error });
      }
    }

    this.isProcessing = false;
    this.setState('idle');
  }

  
  private async playAudioBuffer(item: AudioQueueItem): Promise<void> {
    try {
      
      const audioBuffer = await this.audioContext.decodeAudioData(item.buffer.slice(0));

      if (this.stopRequested) return;

      this.currentSource = this.audioContext.createBufferSource();
      this.currentSource.buffer = audioBuffer;
      this.currentSource.connect(this.gainNode);

      
      return new Promise((resolve) => {
        if (!this.currentSource) {
          resolve();
          return;
        }

        this.currentSource.onended = () => {
          this.currentSource = null;
          this.emit('audioEnded', item);
          resolve();
        };

        if (this.stopRequested) {
          resolve();
          return;
        }

        this.setState('playing');
        this.emit('audioStarted', item);
        this.currentSource.start(0);
      });
    } catch (error) {
      gatedConsole.voice.error('Failed to decode audio:', error);
      throw error;
    }
  }

  
  stop() {
    this.stopRequested = true;
    if (this.currentSource) {
      try {
        this.currentSource.stop();
        this.currentSource.disconnect();
      } catch (e) {
        
      }
      this.currentSource = null;
    }

    this.playbackQueue = [];
    this.isProcessing = false;
    this.setState('idle');
    this.emit('stopped');
  }

  
  pause() {
    if (this.state === 'playing') {
      this.setState('paused');
      
      if (this.currentSource) {
        this.currentSource.stop();
        this.currentSource = null;
      }
      this.emit('paused');
    }
  }

  
  resume() {
    if (this.state === 'paused') {
      this.setState('idle');
      this.processQueue();
      this.emit('resumed');
    }
  }

  
  setVolume(volume: number) {
    this.volume = Math.max(0, Math.min(1, volume));
    this.gainNode.gain.setValueAtTime(this.volume, this.audioContext.currentTime);
    this.emit('volumeChanged', this.volume);
  }

  
  getVolume(): number {
    return this.volume;
  }

  
  getQueueLength(): number {
    return this.playbackQueue.length;
  }

  
  clearQueue() {
    this.playbackQueue = [];
    this.emit('queueCleared');
  }

  
  getState(): AudioOutputState {
    return this.state;
  }

  
  private setState(state: AudioOutputState) {
    this.state = state;
    this.emit('stateChange', state);
  }

  
  on(event: string, callback: Function) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);
  }

  off(event: string, callback: Function) {
    if (this.listeners.has(event)) {
      this.listeners.get(event)!.delete(callback);
    }
  }

  private emit(event: string, ...args: any[]) {
    if (this.listeners.has(event)) {
      this.listeners.get(event)!.forEach(callback => {
        callback(...args);
      });
    }
  }

  
  static isSupported(): boolean {
    return !!(window.AudioContext && window.ArrayBuffer);
  }
}