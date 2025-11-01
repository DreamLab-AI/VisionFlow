

import { AudioOutputService } from './AudioOutputService';
import { AudioInputService, AudioChunk } from './AudioInputService';
import { useSettingsStore } from '../store/settingsStore';
import { gatedConsole } from '../utils/console';
import { createLogger } from '../utils/loggerConfig';

const logger = createLogger('VoiceWebSocketService');

export interface VoiceMessage {
  type: 'tts' | 'stt' | 'audio_chunk' | 'transcription' | 'error' | 'connected';
  data?: any;
}

export interface TTSRequest {
  text: string;
  voice?: string;
  speed?: number;
  stream?: boolean;
}

export interface TranscriptionResult {
  text: string;
  isFinal: boolean;
  confidence?: number;
  timestamp?: number;
}

export class VoiceWebSocketService {
  private static instance: VoiceWebSocketService;
  private socket: WebSocket | null = null;
  private audioOutput: AudioOutputService;
  private audioInput: AudioInputService;
  private isStreamingAudio = false;
  private transcriptionCallback?: (result: TranscriptionResult) => void;
  private listeners: Map<string, Set<Function>> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 2000;

  private constructor() {
    this.audioOutput = AudioOutputService.getInstance();
    this.audioInput = AudioInputService.getInstance();

    
    this.setupAudioInputListeners();
  }

  static getInstance(): VoiceWebSocketService {
    if (!VoiceWebSocketService.instance) {
      VoiceWebSocketService.instance = new VoiceWebSocketService();
    }
    return VoiceWebSocketService.instance;
  }

  
  async connectToSpeech(baseUrl: string): Promise<void> {
    const wsUrl = baseUrl.replace(/^http/, 'ws') + '/ws/speech';
    await this.connect(wsUrl);
  }

  
  async connect(url: string): Promise<void> {
    
    if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
      return;
    }

    
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }

    return new Promise((resolve, reject) => {
      try {
        this.socket = new WebSocket(url);

        this.socket.onopen = () => {
          gatedConsole.voice.log('Voice WebSocket connected');
          this.reconnectAttempts = 0;
          this.emit('connected');
          resolve();
        };

        this.socket.onmessage = (event) => {
          this.handleMessage(event);
        };

        this.socket.onclose = (event) => {
          gatedConsole.voice.log('Voice WebSocket disconnected');
          this.emit('disconnected', event);
          if (event.code !== 1000) { 
            this.attemptReconnect(url);
          }
        };

        this.socket.onerror = (error) => {
          gatedConsole.voice.error('Voice WebSocket error:', error);
          this.emit('error', error);
          reject(error);
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  
  private handleMessage(event: MessageEvent): void {
    
    if (event.data instanceof ArrayBuffer || event.data instanceof Blob) {
      this.handleAudioData(event.data);
      return;
    }

    
    try {
      const message: VoiceMessage = JSON.parse(event.data);
      
      
      if (message.type === 'error') {
        logger.error('Raw error message from server', message);
      }

      switch (message.type) {
        case 'connected':
          gatedConsole.voice.log('Connected to voice service:', message.data);
          this.emit('voiceConnected', message.data);
          break;

        case 'transcription':
          this.handleTranscription(message.data);
          break;

        case 'error':
          const errorMsg = message.data || message.error || 'Unknown voice service error';
          gatedConsole.voice.error('Voice service error:', errorMsg);
          this.emit('voiceError', errorMsg);
          break;

        default:
          this.emit('message', message);
      }
    } catch (error) {
      gatedConsole.voice.error('Failed to parse voice message:', error);
    }
  }

  
  private async handleAudioData(data: ArrayBuffer | Blob) {
    try {
      
      const buffer = data instanceof Blob ? await data.arrayBuffer() : data;

      
      await this.audioOutput.queueAudio(buffer);
      this.emit('audioReceived', buffer);
    } catch (error) {
      gatedConsole.voice.error('Failed to handle audio data:', error);
      this.emit('audioError', error);
    }
  }

  
  private handleTranscription(data: TranscriptionResult) {
    if (this.transcriptionCallback) {
      this.transcriptionCallback(data);
    }
    this.emit('transcription', data);
  }

  
  async sendTextForTTS(request: TTSRequest): Promise<void> {
    if (!this.isConnected()) {
      throw new Error('Not connected to voice service');
    }

    const message: VoiceMessage = {
      type: 'tts',
      data: request
    };

    this.send(JSON.stringify(message));
    this.emit('ttsSent', request);
  }

  
  async startAudioStreaming(options?: { language?: string }): Promise<void> {
    if (!this.isConnected()) {
      throw new Error('Not connected to voice service');
    }

    if (this.isStreamingAudio) {
      gatedConsole.voice.warn('Audio streaming already active');
      return;
    }

    
    const support = AudioInputService.getBrowserSupport();
    logger.warn('DEVELOPER MODE: Browser support checks bypassed', support);
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    try {
      
      const micAccess = await this.audioInput.requestMicrophoneAccess();
      if (!micAccess) {
        throw new Error('Microphone access denied');
      }

      
      await this.audioInput.startRecording();
      this.isStreamingAudio = true;

      
      const message = {
        type: 'stt',
        action: 'start',
        language: options?.language || 'en',
        model: options?.model || 'whisper-1',
        ...options
      };

      this.send(JSON.stringify(message));
      this.emit('audioStreamingStarted');
    } catch (error) {
      
      this.isStreamingAudio = false;
      this.audioInput.stopRecording();
      throw error;
    }
  }

  
  stopAudioStreaming() {
    if (!this.isStreamingAudio) {
      return;
    }

    
    this.audioInput.stopRecording();
    
    
    
    
    if (this.isConnected()) {
      
      const message = {
        type: 'stt',
        action: 'stop'
      };

      this.send(JSON.stringify(message));
    }

    
    setTimeout(() => {
      this.isStreamingAudio = false;
      this.emit('audioStreamingStopped');
    }, 100);
  }

  
  private setupAudioInputListeners() {
    
    this.audioInput.on('recordingComplete', async (completeAudio: Blob) => {
      logger.debug('Recording complete event received', { blobSize: completeAudio.size });
      if (this.isStreamingAudio && this.isConnected()) {
        
        try {
          const arrayBuffer = await completeAudio.arrayBuffer();
          logger.debug('Sending complete audio file', { bytes: arrayBuffer.byteLength });
          gatedConsole.voice.log('Sending complete audio file:', {
            size: arrayBuffer.byteLength,
            type: completeAudio.type
          });
          
          this.sendBinary(arrayBuffer);
        } catch (error) {
          console.error('[VoiceWebSocketService] Failed to send audio:', error);
          gatedConsole.voice.error('Failed to send audio:', error);
        }
      } else {
        logger.debug('Not sending audio', { streaming: this.isStreamingAudio, connected: this.isConnected() });
      }
    });

    
    this.audioInput.on('audioChunk', (chunk: AudioChunk) => {
      
      
    });

    this.audioInput.on('error', (error: any) => {
      gatedConsole.voice.error('Audio input error:', error);
      this.emit('audioInputError', error);
    });

    this.audioInput.on('audioLevel', (level: number) => {
      this.emit('audioLevel', level);
    });

    this.audioInput.on('stateChange', (state: AudioInputState) => {
      this.emit('audioInputStateChange', state);
    });
  }

  
  private isConnected(): boolean {
    return this.socket?.readyState === WebSocket.OPEN;
  }

  
  private send(data: string): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(data);
    }
  }

  
  private sendBinary(data: ArrayBuffer): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(data);
    }
  }

  
  private attemptReconnect(url: string) {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        gatedConsole.voice.log(`Attempting to reconnect voice WebSocket (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        this.connect(url).catch((error) => gatedConsole.voice.error('Reconnect failed:', error));
      }, this.reconnectDelay);
    }
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

  
  onTranscription(callback: (result: TranscriptionResult) => void) {
    this.transcriptionCallback = callback;
  }

  
  stopAllAudio() {
    this.stopAudioStreaming();
    this.audioOutput.stop();
  }

  
  async disconnect(): Promise<void> {
    this.stopAllAudio();
    if (this.socket) {
      this.socket.close(1000, 'Normal closure'); 
      this.socket = null;
    }
    
    this.reconnectAttempts = this.maxReconnectAttempts;
  }

  
  getAudioOutput(): AudioOutputService {
    return this.audioOutput;
  }

  
  getAudioInput(): AudioInputService {
    return this.audioInput;
  }
}