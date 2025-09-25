/**
 * VoiceWebSocketService - Specialized WebSocket service for voice/audio communication
 * Handles bidirectional audio streaming with Whisper STT and Kokoros TTS integration
 */

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

    // Set up audio input listeners
    this.setupAudioInputListeners();
  }

  static getInstance(): VoiceWebSocketService {
    if (!VoiceWebSocketService.instance) {
      VoiceWebSocketService.instance = new VoiceWebSocketService();
    }
    return VoiceWebSocketService.instance;
  }

  /**
   * Connect to the speech WebSocket endpoint
   */
  async connectToSpeech(baseUrl: string): Promise<void> {
    const wsUrl = baseUrl.replace(/^http/, 'ws') + '/ws/speech';
    await this.connect(wsUrl);
  }

  /**
   * Connect to WebSocket
   */
  async connect(url: string): Promise<void> {
    // Check if already connected or connecting
    if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
      return;
    }

    // Clean up any existing connection
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
          if (event.code !== 1000) { // Only reconnect if not normal closure
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

  /**
   * Handle WebSocket messages
   */
  private handleMessage(event: MessageEvent): void {
    // Handle binary audio data
    if (event.data instanceof ArrayBuffer || event.data instanceof Blob) {
      this.handleAudioData(event.data);
      return;
    }

    // Handle text messages
    try {
      const message: VoiceMessage = JSON.parse(event.data);
      
      // Debug logging to see what we're receiving
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

  /**
   * Handle incoming audio data
   */
  private async handleAudioData(data: ArrayBuffer | Blob) {
    try {
      // Convert Blob to ArrayBuffer if needed
      const buffer = data instanceof Blob ? await data.arrayBuffer() : data;

      // Queue audio for playback
      await this.audioOutput.queueAudio(buffer);
      this.emit('audioReceived', buffer);
    } catch (error) {
      gatedConsole.voice.error('Failed to handle audio data:', error);
      this.emit('audioError', error);
    }
  }

  /**
   * Handle transcription results
   */
  private handleTranscription(data: TranscriptionResult) {
    if (this.transcriptionCallback) {
      this.transcriptionCallback(data);
    }
    this.emit('transcription', data);
  }

  /**
   * Send text for TTS
   */
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

  /**
   * Start streaming audio for STT
   */
  async startAudioStreaming(options?: { language?: string }): Promise<void> {
    if (!this.isConnected()) {
      throw new Error('Not connected to voice service');
    }

    if (this.isStreamingAudio) {
      gatedConsole.voice.warn('Audio streaming already active');
      return;
    }

    // DEVELOPER WORKAROUND: Bypass all browser checks for testing
    const support = AudioInputService.getBrowserSupport();
    logger.warn('DEVELOPER MODE: Browser support checks bypassed', support);
    
    // Original checks commented out for testing
    // if (!support.getUserMedia) {
    //   throw new Error('Browser does not support microphone access. Please use a modern browser with HTTPS.');
    // }
    // if (!support.isHttps) {
    //   throw new Error('Microphone access requires HTTPS or localhost. Please use a secure connection.');
    // }
    // if (!support.mediaRecorder) {
    //   throw new Error('Browser does not support audio recording. Please use a modern browser.');
    // }
    // if (!support.audioContext) {
    //   throw new Error('Browser does not support Web Audio API. Please use a modern browser.');
    // }

    try {
      // Request microphone access
      const micAccess = await this.audioInput.requestMicrophoneAccess();
      if (!micAccess) {
        throw new Error('Microphone access denied');
      }

      // Start recording
      await this.audioInput.startRecording();
      this.isStreamingAudio = true;

      // Send start streaming message - server expects flat structure
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
      // Clean up on error
      this.isStreamingAudio = false;
      this.audioInput.stopRecording();
      throw error;
    }
  }

  /**
   * Stop audio streaming
   */
  stopAudioStreaming() {
    if (!this.isStreamingAudio) {
      return;
    }

    // Stop recording first - this will trigger the recordingComplete event
    this.audioInput.stopRecording();
    
    // Keep the flag true until after the audio is sent
    // We'll set it to false after sending the complete audio
    
    if (this.isConnected()) {
      // Send stop message - server expects flat structure
      const message = {
        type: 'stt',
        action: 'stop'
      };

      this.send(JSON.stringify(message));
    }

    // Delay setting the flag to false to allow recordingComplete to send audio
    setTimeout(() => {
      this.isStreamingAudio = false;
      this.emit('audioStreamingStopped');
    }, 100);
  }

  /**
   * Set up audio input listeners
   */
  private setupAudioInputListeners() {
    // Listen for complete recording instead of chunks
    this.audioInput.on('recordingComplete', async (completeAudio: Blob) => {
      logger.debug('Recording complete event received', { blobSize: completeAudio.size });
      if (this.isStreamingAudio && this.isConnected()) {
        // Convert blob to ArrayBuffer and send complete audio
        try {
          const arrayBuffer = await completeAudio.arrayBuffer();
          logger.debug('Sending complete audio file', { bytes: arrayBuffer.byteLength });
          gatedConsole.voice.log('Sending complete audio file:', {
            size: arrayBuffer.byteLength,
            type: completeAudio.type
          });
          // Send the complete audio file
          this.sendBinary(arrayBuffer);
        } catch (error) {
          console.error('[VoiceWebSocketService] Failed to send audio:', error);
          gatedConsole.voice.error('Failed to send audio:', error);
        }
      } else {
        logger.debug('Not sending audio', { streaming: this.isStreamingAudio, connected: this.isConnected() });
      }
    });

    // Still listen to chunks for audio level monitoring
    this.audioInput.on('audioChunk', (chunk: AudioChunk) => {
      // Use chunks only for monitoring, not sending
      // This keeps audio level indicators working
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

  /**
   * Check if connected
   */
  private isConnected(): boolean {
    return this.socket?.readyState === WebSocket.OPEN;
  }

  /**
   * Send text message
   */
  private send(data: string): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(data);
    }
  }

  /**
   * Send binary data
   */
  private sendBinary(data: ArrayBuffer): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(data);
    }
  }

  /**
   * Attempt to reconnect
   */
  private attemptReconnect(url: string) {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        gatedConsole.voice.log(`Attempting to reconnect voice WebSocket (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        this.connect(url).catch((error) => gatedConsole.voice.error('Reconnect failed:', error));
      }, this.reconnectDelay);
    }
  }

  /**
   * Event emitter functionality
   */
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

  /**
   * Set transcription callback
   */
  onTranscription(callback: (result: TranscriptionResult) => void) {
    this.transcriptionCallback = callback;
  }

  /**
   * Stop all audio operations
   */
  stopAllAudio() {
    this.stopAudioStreaming();
    this.audioOutput.stop();
  }

  /**
   * Clean up resources
   */
  async disconnect(): Promise<void> {
    this.stopAllAudio();
    if (this.socket) {
      this.socket.close(1000, 'Normal closure'); // Send normal closure code
      this.socket = null;
    }
    // Reset reconnection attempts to prevent reconnecting after manual disconnect
    this.reconnectAttempts = this.maxReconnectAttempts;
  }

  /**
   * Get audio output service instance
   */
  getAudioOutput(): AudioOutputService {
    return this.audioOutput;
  }

  /**
   * Get audio input service instance
   */
  getAudioInput(): AudioInputService {
    return this.audioInput;
  }
}