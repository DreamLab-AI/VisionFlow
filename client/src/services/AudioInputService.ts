

import { AudioContextManager } from './AudioContextManager';
import { gatedConsole } from '../utils/console';
import { createLogger } from '../utils/loggerConfig';

const logger = createLogger('AudioInputService');

// Legacy vendor-prefixed APIs for older browsers
interface NavigatorWithLegacy extends Navigator {
  webkitGetUserMedia?: Navigator['mediaDevices']['getUserMedia'];
  mozGetUserMedia?: Navigator['mediaDevices']['getUserMedia'];
  msGetUserMedia?: Navigator['mediaDevices']['getUserMedia'];
}

interface WindowWithWebkit extends Window {
  webkitAudioContext?: typeof AudioContext;
}

export interface AudioConstraints {
  echoCancellation?: boolean;
  noiseSuppression?: boolean;
  autoGainControl?: boolean;
  sampleRate?: number;
  channelCount?: number;
}

export interface AudioChunk {
  data: ArrayBuffer;
  timestamp: number;
  duration: number;
}

export type AudioInputState = 'idle' | 'requesting' | 'ready' | 'recording' | 'paused' | 'error';

export class AudioInputService {
  private static instance: AudioInputService;
  private stream: MediaStream | null = null;
  private mediaRecorder: MediaRecorder | null = null;
  private audioContext: AudioContext | null = null;
  private sourceNode: MediaStreamAudioSourceNode | null = null;
  private analyserNode: AnalyserNode | null = null;
  private processorNode: AudioWorkletNode | null = null;
  private state: AudioInputState = 'idle';
  private listeners: Map<string, Set<Function>> = new Map();
  private audioChunks: AudioChunk[] = [];
  private recordingStartTime: number = 0;
  private audioBlobs: Blob[] = [];  

  private constructor() {
    this.initializeAudioContext();
  }

  static getInstance(): AudioInputService {
    if (!AudioInputService.instance) {
      AudioInputService.instance = new AudioInputService();
    }
    return AudioInputService.instance;
  }

  private async initializeAudioContext() {
    this.audioContext = AudioContextManager.getInstance().getContext();
  }

  
  async requestMicrophoneAccess(constraints: AudioConstraints = {}): Promise<boolean> {
    try {
      this.setState('requesting');

      
      if (!navigator || !navigator.mediaDevices) {
        throw new Error('Browser does not support media devices. Please use a modern browser with HTTPS.');
      }

      
      const nav = navigator as NavigatorWithLegacy;
      const getUserMedia = navigator.mediaDevices?.getUserMedia ||
                          nav.webkitGetUserMedia ||
                          nav.mozGetUserMedia ||
                          nav.msGetUserMedia;

      if (!getUserMedia) {
        throw new Error('Browser does not support microphone access. Please use a modern browser with HTTPS.');
      }

      
      
      
      
      logger.warn('DEVELOPER MODE: HTTPS check bypassed in AudioInputService');

      const defaultConstraints: MediaStreamConstraints = {
        audio: {
          echoCancellation: constraints.echoCancellation ?? true,
          noiseSuppression: constraints.noiseSuppression ?? true,
          autoGainControl: constraints.autoGainControl ?? true,
          sampleRate: constraints.sampleRate ?? 48000,
          channelCount: constraints.channelCount ?? 1
        }
      };

      // Use modern API
      if (navigator.mediaDevices?.getUserMedia) {
        this.stream = await navigator.mediaDevices.getUserMedia(defaultConstraints);
      } else if (getUserMedia) {
        // Fallback to legacy API with callback style
        this.stream = await new Promise<MediaStream>((resolve, reject) => {
          (getUserMedia as (constraints: MediaStreamConstraints, successCb: (stream: MediaStream) => void, errorCb: (err: DOMException) => void) => void).call(navigator, defaultConstraints, resolve, reject);
        });
      } else {
        throw new Error('getUserMedia is not supported');
      }

      await this.setupAudioNodes();
      this.setState('ready');
      return true;
    } catch (error) {
      gatedConsole.voice.error('Failed to access microphone:', error);
      this.setState('error');
      this.emit('error', error);
      return false;
    }
  }

  
  private async setupAudioNodes() {
    if (!this.stream || !this.audioContext) return;

    
    this.sourceNode = this.audioContext.createMediaStreamSource(this.stream);

    
    this.analyserNode = this.audioContext.createAnalyser();
    this.analyserNode.fftSize = 2048;
    this.analyserNode.smoothingTimeConstant = 0.8;

    
    this.sourceNode.connect(this.analyserNode);

    
    try {
      await this.setupAudioWorklet();
    } catch (error) {
      gatedConsole.voice.warn('Audio worklet setup failed, continuing without processing:', error);
    }
  }

  
  private async setupAudioWorklet() {
    if (!this.audioContext || !this.sourceNode) return;

    
    await this.audioContext.audioWorklet.addModule('/audio-processor.js');

    
    this.processorNode = new AudioWorkletNode(this.audioContext, 'audio-processor', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      processorOptions: {
        bufferSize: 4096
      }
    });

    
    this.processorNode.port.onmessage = (event) => {
      if (event.data.type === 'audioLevel') {
        this.emit('audioLevel', event.data.level);
      }
    };

    
    this.sourceNode.disconnect();
    this.sourceNode.connect(this.processorNode);
    this.processorNode.connect(this.analyserNode!);
  }

  
  async startRecording(mimeType: string = 'audio/webm;codecs=opus'): Promise<void> {
    if (!this.stream || this.state !== 'ready') {
      throw new Error('Microphone not ready. Call requestMicrophoneAccess first.');
    }

    this.audioChunks = [];
    this.audioBlobs = [];  
    this.recordingStartTime = Date.now();

    
    const supportedType = this.getSupportedMimeType(mimeType);

    this.mediaRecorder = new MediaRecorder(this.stream, {
      mimeType: supportedType,
      audioBitsPerSecond: 128000
    });

    this.mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        
        this.audioBlobs.push(event.data);
        
        this.handleAudioData(event.data);
      }
    };

    this.mediaRecorder.onerror = (error) => {
      gatedConsole.voice.error('MediaRecorder error:', error);
      this.emit('error', error);
      this.stopRecording();
    };

    this.mediaRecorder.onstop = async () => {
      
      logger.debug('Recording stopped, creating complete audio blob', { chunks: this.audioBlobs.length });
      const completeAudio = new Blob(this.audioBlobs, { type: this.mediaRecorder?.mimeType || 'audio/webm' });
      logger.debug('Complete audio blob created', { size: completeAudio.size, type: completeAudio.type });
      this.emit('recordingComplete', completeAudio);
      this.emit('recordingStopped', this.audioChunks);
    };

    
    this.mediaRecorder.start(100); 
    this.setState('recording');
    this.emit('recordingStarted');
  }

  
  stopRecording(): void {
    if (this.mediaRecorder && this.state === 'recording') {
      logger.debug('Stopping recording', { blobsCollected: this.audioBlobs.length });
      this.mediaRecorder.stop();
      this.setState('ready');
    }
  }

  
  pauseRecording(): void {
    if (this.mediaRecorder && this.state === 'recording') {
      this.mediaRecorder.pause();
      this.setState('paused');
      this.emit('recordingPaused');
    }
  }

  
  resumeRecording(): void {
    if (this.mediaRecorder && this.state === 'paused') {
      this.mediaRecorder.resume();
      this.setState('recording');
      this.emit('recordingResumed');
    }
  }

  
  private async handleAudioData(blob: Blob) {
    const arrayBuffer = await blob.arrayBuffer();
    const chunk: AudioChunk = {
      data: arrayBuffer,
      timestamp: Date.now() - this.recordingStartTime,
      duration: 100 
    };

    this.audioChunks.push(chunk);
    this.emit('audioChunk', chunk);
  }

  
  private getSupportedMimeType(preferred: string): string {
    const types = [
      preferred,
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
      'audio/mp4'
    ];

    for (const type of types) {
      if (MediaRecorder.isTypeSupported(type)) {
        return type;
      }
    }

    return ''; 
  }

  
  getAudioLevel(): number {
    if (!this.analyserNode) return 0;

    const bufferLength = this.analyserNode.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    this.analyserNode.getByteFrequencyData(dataArray);

    let sum = 0;
    for (let i = 0; i < bufferLength; i++) {
      sum += dataArray[i];
    }

    return sum / (bufferLength * 255);
  }

  
  getFrequencyData(): Uint8Array {
    if (!this.analyserNode) return new Uint8Array(0);

    const bufferLength = this.analyserNode.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    this.analyserNode.getByteFrequencyData(dataArray);

    return dataArray;
  }

  
  getWaveformData(): Uint8Array {
    if (!this.analyserNode) return new Uint8Array(0);

    const bufferLength = this.analyserNode.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    this.analyserNode.getByteTimeDomainData(dataArray);

    return dataArray;
  }

  
  async release() {
    this.stopRecording();

    if (this.processorNode) {
      this.processorNode.disconnect();
      this.processorNode = null;
    }

    if (this.analyserNode) {
      this.analyserNode.disconnect();
      this.analyserNode = null;
    }

    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }

    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }

    this.setState('idle');
  }

  
  getState(): AudioInputState {
    return this.state;
  }

  
  private setState(state: AudioInputState) {
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

  private emit(event: string, ...args: unknown[]) {
    if (this.listeners.has(event)) {
      this.listeners.get(event)!.forEach(callback => {
        callback(...args);
      });
    }
  }

  
  static isSupported(): boolean {
    const hasGetUserMedia = !!(navigator.mediaDevices && typeof navigator.mediaDevices.getUserMedia === 'function');
    const hasMediaRecorder = typeof MediaRecorder !== 'undefined';
    const hasAudioContext = typeof AudioContext !== 'undefined' || typeof (window as WindowWithWebkit).webkitAudioContext !== 'undefined';
    return hasGetUserMedia && hasMediaRecorder && hasAudioContext;
  }

  
  static getBrowserSupport(): {
    mediaDevices: boolean;
    getUserMedia: boolean;
    mediaRecorder: boolean;
    audioContext: boolean;
    isHttps: boolean;
  } {
    return {
      mediaDevices: !!navigator.mediaDevices,
      getUserMedia: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) ||
                   !!((navigator as NavigatorWithLegacy).webkitGetUserMedia ||
                      (navigator as NavigatorWithLegacy).mozGetUserMedia ||
                      (navigator as NavigatorWithLegacy).msGetUserMedia),
      mediaRecorder: !!window.MediaRecorder,
      audioContext: !!(window.AudioContext || (window as WindowWithWebkit).webkitAudioContext),
      isHttps: location.protocol === 'https:' ||
               location.hostname === 'localhost' ||
               location.hostname === '127.0.0.1'
    };
  }
}