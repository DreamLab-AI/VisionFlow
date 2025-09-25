/**
 * Centralized Voice Interaction Hook Architecture
 *
 * This is the proposed centralized architecture that consolidates all voice
 * state management patterns into a single, consistent approach using React Context.
 *
 * This implementation addresses the 3 current patterns identified:
 * 1. Direct service access (eliminated)
 * 2. Hook-based abstraction (enhanced)
 * 3. Mixed patterns (consolidated)
 */

import React, { createContext, useContext, useCallback, useEffect, useRef, useState, useMemo } from 'react';
import { VoiceWebSocketService, TranscriptionResult, TTSRequest } from '../services/VoiceWebSocketService';
import { AudioInputService, AudioInputState } from '../services/AudioInputService';
import { AudioOutputService, AudioOutputState } from '../services/AudioOutputService';
import { useSettingsStore } from '../store/settingsStore';
import { gatedConsole } from '../utils/console';

// =============================================================================
// TYPE DEFINITIONS
// =============================================================================

export interface BrowserSupportInfo {
  mediaDevices: boolean;
  getUserMedia: boolean;
  mediaRecorder: boolean;
  audioContext: boolean;
  isHttps: boolean;
  webRTC: boolean;
  isSupported: boolean;
}

export interface VoiceSettings {
  kokoro?: {
    defaultVoice?: string;
    defaultSpeed?: number;
    stream?: boolean;
    defaultFormat?: string;
  };
  whisper?: {
    defaultModel?: string;
    defaultLanguage?: string;
    temperature?: number;
  };
  audio?: {
    echoCancellation?: boolean;
    noiseSuppression?: boolean;
    autoGainControl?: boolean;
    sampleRate?: number;
    volume?: number;
  };
}

export interface VoiceListenOptions {
  language?: string;
  model?: string;
  continuous?: boolean;
  interimResults?: boolean;
}

export interface TTSOptions {
  voice?: string;
  speed?: number;
  format?: string;
  stream?: boolean;
}

export interface VoicePermissionState {
  microphone: 'granted' | 'denied' | 'prompt' | 'checking';
  lastChecked: number | null;
  error: string | null;
}

// =============================================================================
// CONTEXT STATE INTERFACE
// =============================================================================

export interface VoiceContextState {
  // Connection state
  isConnected: boolean;
  isConnecting: boolean;
  connectionError: string | null;
  lastConnectionAttempt: number | null;

  // Audio input state
  isListening: boolean;
  audioLevel: number;
  audioInputState: AudioInputState;
  microphonePermission: VoicePermissionState;

  // Audio output state
  isSpeaking: boolean;
  audioOutputState: AudioOutputState;
  volume: number;
  audioQueueLength: number;

  // Transcription state
  transcription: string;
  partialTranscription: string;
  transcriptionHistory: TranscriptionResult[];
  maxHistoryLength: number;

  // Settings state
  voiceSettings: VoiceSettings;

  // Browser support and capabilities
  browserSupport: BrowserSupportInfo;

  // General state
  lastError: Error | null;
  isInitialized: boolean;
}

export interface VoiceContextActions {
  // Connection management
  connect: () => Promise<void>;
  disconnect: () => Promise<void>;
  reconnect: () => Promise<void>;

  // Audio input control
  startListening: (options?: VoiceListenOptions) => Promise<void>;
  stopListening: () => void;
  toggleListening: () => Promise<void>;

  // Audio output control
  speak: (text: string, options?: TTSOptions) => Promise<void>;
  stopSpeaking: () => void;
  clearAudioQueue: () => void;
  setVolume: (volume: number) => void;

  // Transcription management
  clearTranscription: () => void;
  clearTranscriptionHistory: () => void;
  setMaxHistoryLength: (length: number) => void;

  // Settings management
  updateVoiceSettings: (settings: Partial<VoiceSettings>) => void;

  // Permission management
  requestMicrophonePermission: () => Promise<boolean>;
  checkBrowserSupport: () => BrowserSupportInfo;

  // Utility functions
  clearAllErrors: () => void;
  getAudioLevel: () => number;
  getConnectionStatus: () => 'connected' | 'connecting' | 'disconnected' | 'error';
}

type VoiceContextType = VoiceContextState & VoiceContextActions;

// =============================================================================
// CONTEXT CREATION
// =============================================================================

const VoiceContext = createContext<VoiceContextType | null>(null);

// =============================================================================
// BROWSER SUPPORT DETECTION
// =============================================================================

function detectBrowserSupport(): BrowserSupportInfo {
  const mediaDevices = !!navigator.mediaDevices;
  const getUserMedia = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) ||
                      !!((navigator as any).webkitGetUserMedia ||
                         (navigator as any).mozGetUserMedia ||
                         (navigator as any).msGetUserMedia);
  const mediaRecorder = !!window.MediaRecorder;
  const audioContext = !!(window.AudioContext || (window as any).webkitAudioContext);
  const isHttps = location.protocol === 'https:' ||
                  location.hostname === 'localhost' ||
                  location.hostname === '127.0.0.1';
  const webRTC = !!(window.RTCPeerConnection || (window as any).webkitRTCPeerConnection);

  return {
    mediaDevices,
    getUserMedia,
    mediaRecorder,
    audioContext,
    isHttps,
    webRTC,
    isSupported: getUserMedia && mediaRecorder && audioContext && isHttps
  };
}

// =============================================================================
// VOICE CONTEXT PROVIDER
// =============================================================================

interface VoiceProviderProps {
  children: React.ReactNode;
  autoConnect?: boolean;
  maxHistoryLength?: number;
}

export function VoiceProvider({
  children,
  autoConnect = true,
  maxHistoryLength = 50
}: VoiceProviderProps): JSX.Element {
  // Service references (stable across re-renders)
  const voiceServiceRef = useRef<VoiceWebSocketService>();
  const audioInputRef = useRef<AudioInputService>();
  const audioOutputRef = useRef<AudioOutputService>();

  // Settings integration
  const settings = useSettingsStore((state) => state.settings);
  const updateSettings = useSettingsStore((state) => state.updateSettings);

  // Core state
  const [state, setState] = useState<VoiceContextState>(() => ({
    // Connection state
    isConnected: false,
    isConnecting: false,
    connectionError: null,
    lastConnectionAttempt: null,

    // Audio input state
    isListening: false,
    audioLevel: 0,
    audioInputState: 'idle',
    microphonePermission: {
      microphone: 'prompt',
      lastChecked: null,
      error: null
    },

    // Audio output state
    isSpeaking: false,
    audioOutputState: 'idle',
    volume: 1.0,
    audioQueueLength: 0,

    // Transcription state
    transcription: '',
    partialTranscription: '',
    transcriptionHistory: [],
    maxHistoryLength,

    // Settings state
    voiceSettings: {
      kokoro: settings?.kokoro,
      whisper: settings?.whisper,
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        sampleRate: 48000,
        volume: 1.0
      }
    },

    // Browser support
    browserSupport: detectBrowserSupport(),

    // General state
    lastError: null,
    isInitialized: false
  }));

  // =============================================================================
  // SERVICE INITIALIZATION
  // =============================================================================

  useEffect(() => {
    // Initialize services once
    voiceServiceRef.current = VoiceWebSocketService.getInstance();
    audioInputRef.current = AudioInputService.getInstance();
    audioOutputRef.current = AudioOutputService.getInstance();

    setState(prev => ({ ...prev, isInitialized: true }));
  }, []);

  // =============================================================================
  // EVENT LISTENERS SETUP
  // =============================================================================

  useEffect(() => {
    if (!state.isInitialized) return;

    const voiceService = voiceServiceRef.current!;
    const audioInput = audioInputRef.current!;
    const audioOutput = audioOutputRef.current!;

    // Voice service events
    const handleConnected = () => {
      setState(prev => ({
        ...prev,
        isConnected: true,
        isConnecting: false,
        connectionError: null
      }));
    };

    const handleDisconnected = () => {
      setState(prev => ({
        ...prev,
        isConnected: false,
        isConnecting: false,
        isListening: false
      }));
    };

    const handleTranscription = (result: TranscriptionResult) => {
      setState(prev => {
        const newHistory = result.isFinal
          ? [...prev.transcriptionHistory, result].slice(-prev.maxHistoryLength)
          : prev.transcriptionHistory;

        return {
          ...prev,
          transcription: result.isFinal ? result.text : prev.transcription,
          partialTranscription: result.isFinal ? '' : result.text,
          transcriptionHistory: newHistory
        };
      });
    };

    const handleVoiceError = (error: any) => {
      setState(prev => ({
        ...prev,
        lastError: new Error(error),
        connectionError: error
      }));
    };

    // Audio input events
    const handleAudioLevel = (level: number) => {
      setState(prev => ({ ...prev, audioLevel: level }));
    };

    const handleAudioInputStateChange = (inputState: AudioInputState) => {
      setState(prev => ({ ...prev, audioInputState: inputState }));
    };

    const handleAudioStreamingStarted = () => {
      setState(prev => ({ ...prev, isListening: true }));
    };

    const handleAudioStreamingStopped = () => {
      setState(prev => ({ ...prev, isListening: false }));
    };

    // Audio output events
    const handleAudioOutputStateChange = (outputState: AudioOutputState) => {
      setState(prev => ({ ...prev, audioOutputState: outputState }));
    };

    const handleAudioStarted = () => {
      setState(prev => ({ ...prev, isSpeaking: true }));
    };

    const handleAudioEnded = () => {
      setState(prev => ({ ...prev, isSpeaking: false }));
    };

    const handleQueueChanged = () => {
      setState(prev => ({
        ...prev,
        audioQueueLength: audioOutput.getQueueLength()
      }));
    };

    // Register all event listeners
    voiceService.on('connected', handleConnected);
    voiceService.on('disconnected', handleDisconnected);
    voiceService.on('transcription', handleTranscription);
    voiceService.on('voiceError', handleVoiceError);
    voiceService.on('audioStreamingStarted', handleAudioStreamingStarted);
    voiceService.on('audioStreamingStopped', handleAudioStreamingStopped);

    audioInput.on('audioLevel', handleAudioLevel);
    audioInput.on('stateChange', handleAudioInputStateChange);

    audioOutput.on('stateChange', handleAudioOutputStateChange);
    audioOutput.on('audioStarted', handleAudioStarted);
    audioOutput.on('audioEnded', handleAudioEnded);
    audioOutput.on('audioQueued', handleQueueChanged);
    audioOutput.on('queueCleared', handleQueueChanged);

    // Cleanup function
    return () => {
      voiceService.off('connected', handleConnected);
      voiceService.off('disconnected', handleDisconnected);
      voiceService.off('transcription', handleTranscription);
      voiceService.off('voiceError', handleVoiceError);
      voiceService.off('audioStreamingStarted', handleAudioStreamingStarted);
      voiceService.off('audioStreamingStopped', handleAudioStreamingStopped);

      audioInput.off('audioLevel', handleAudioLevel);
      audioInput.off('stateChange', handleAudioInputStateChange);

      audioOutput.off('stateChange', handleAudioOutputStateChange);
      audioOutput.off('audioStarted', handleAudioStarted);
      audioOutput.off('audioEnded', handleAudioEnded);
      audioOutput.off('audioQueued', handleQueueChanged);
      audioOutput.off('queueCleared', handleQueueChanged);
    };
  }, [state.isInitialized]);

  // =============================================================================
  // AUTO-CONNECT LOGIC
  // =============================================================================

  useEffect(() => {
    if (autoConnect && state.isInitialized && !state.isConnected && !state.isConnecting) {
      const baseUrl = settings?.system?.customBackendUrl || window.location.origin;
      if (baseUrl) {
        connect().catch((error) =>
          gatedConsole.voice.error('Auto-connect failed:', error)
        );
      }
    }
  }, [autoConnect, state.isInitialized, state.isConnected, state.isConnecting, settings?.system?.customBackendUrl]);

  // =============================================================================
  // ACTION IMPLEMENTATIONS
  // =============================================================================

  const connect = useCallback(async (): Promise<void> => {
    if (!voiceServiceRef.current || state.isConnected || state.isConnecting) return;

    setState(prev => ({
      ...prev,
      isConnecting: true,
      connectionError: null,
      lastConnectionAttempt: Date.now()
    }));

    try {
      const baseUrl = settings?.system?.customBackendUrl || window.location.origin;
      await voiceServiceRef.current.connectToSpeech(baseUrl);
    } catch (error) {
      setState(prev => ({
        ...prev,
        isConnecting: false,
        connectionError: error instanceof Error ? error.message : String(error),
        lastError: error instanceof Error ? error : new Error(String(error))
      }));
      throw error;
    }
  }, [state.isConnected, state.isConnecting, settings?.system?.customBackendUrl]);

  const disconnect = useCallback(async (): Promise<void> => {
    if (voiceServiceRef.current) {
      await voiceServiceRef.current.disconnect();
    }
  }, []);

  const reconnect = useCallback(async (): Promise<void> => {
    await disconnect();
    await new Promise(resolve => setTimeout(resolve, 1000)); // Brief delay
    await connect();
  }, [connect, disconnect]);

  const startListening = useCallback(async (options?: VoiceListenOptions): Promise<void> => {
    if (!voiceServiceRef.current || !state.isConnected || state.isListening) return;

    try {
      await voiceServiceRef.current.startAudioStreaming(options);
    } catch (error) {
      setState(prev => ({
        ...prev,
        lastError: error instanceof Error ? error : new Error(String(error))
      }));
      throw error;
    }
  }, [state.isConnected, state.isListening]);

  const stopListening = useCallback((): void => {
    if (voiceServiceRef.current && state.isListening) {
      voiceServiceRef.current.stopAudioStreaming();
    }
  }, [state.isListening]);

  const toggleListening = useCallback(async (): Promise<void> => {
    if (state.isListening) {
      stopListening();
    } else {
      await startListening();
    }
  }, [state.isListening, startListening, stopListening]);

  const speak = useCallback(async (text: string, options?: TTSOptions): Promise<void> => {
    if (!voiceServiceRef.current || !state.isConnected) {
      throw new Error('Not connected to voice service');
    }

    try {
      const ttsRequest: TTSRequest = {
        text,
        voice: options?.voice || state.voiceSettings.kokoro?.defaultVoice,
        speed: options?.speed || state.voiceSettings.kokoro?.defaultSpeed,
        stream: options?.stream ?? state.voiceSettings.kokoro?.stream ?? true
      };

      await voiceServiceRef.current.sendTextForTTS(ttsRequest);
    } catch (error) {
      setState(prev => ({
        ...prev,
        lastError: error instanceof Error ? error : new Error(String(error))
      }));
      throw error;
    }
  }, [state.isConnected, state.voiceSettings]);

  const stopSpeaking = useCallback((): void => {
    if (audioOutputRef.current) {
      audioOutputRef.current.stop();
    }
  }, []);

  const clearAudioQueue = useCallback((): void => {
    if (audioOutputRef.current) {
      audioOutputRef.current.clearQueue();
    }
  }, []);

  const setVolume = useCallback((volume: number): void => {
    if (audioOutputRef.current) {
      audioOutputRef.current.setVolume(volume);
      setState(prev => ({ ...prev, volume }));
    }
  }, []);

  const clearTranscription = useCallback((): void => {
    setState(prev => ({
      ...prev,
      transcription: '',
      partialTranscription: ''
    }));
  }, []);

  const clearTranscriptionHistory = useCallback((): void => {
    setState(prev => ({ ...prev, transcriptionHistory: [] }));
  }, []);

  const setMaxHistoryLength = useCallback((length: number): void => {
    setState(prev => ({
      ...prev,
      maxHistoryLength: length,
      transcriptionHistory: prev.transcriptionHistory.slice(-length)
    }));
  }, []);

  const updateVoiceSettings = useCallback((newSettings: Partial<VoiceSettings>): void => {
    setState(prev => ({
      ...prev,
      voiceSettings: {
        ...prev.voiceSettings,
        ...newSettings
      }
    }));

    // Update global settings if needed
    updateSettings({
      kokoro: newSettings.kokoro ? { ...settings?.kokoro, ...newSettings.kokoro } : settings?.kokoro,
      whisper: newSettings.whisper ? { ...settings?.whisper, ...newSettings.whisper } : settings?.whisper
    });
  }, [settings, updateSettings]);

  const requestMicrophonePermission = useCallback(async (): Promise<boolean> => {
    if (!audioInputRef.current) return false;

    setState(prev => ({
      ...prev,
      microphonePermission: { ...prev.microphonePermission, microphone: 'checking' }
    }));

    try {
      const granted = await audioInputRef.current.requestMicrophoneAccess();
      setState(prev => ({
        ...prev,
        microphonePermission: {
          microphone: granted ? 'granted' : 'denied',
          lastChecked: Date.now(),
          error: null
        }
      }));
      return granted;
    } catch (error) {
      setState(prev => ({
        ...prev,
        microphonePermission: {
          microphone: 'denied',
          lastChecked: Date.now(),
          error: error instanceof Error ? error.message : String(error)
        }
      }));
      return false;
    }
  }, []);

  const checkBrowserSupport = useCallback((): BrowserSupportInfo => {
    const support = detectBrowserSupport();
    setState(prev => ({ ...prev, browserSupport: support }));
    return support;
  }, []);

  const clearAllErrors = useCallback((): void => {
    setState(prev => ({
      ...prev,
      lastError: null,
      connectionError: null,
      microphonePermission: {
        ...prev.microphonePermission,
        error: null
      }
    }));
  }, []);

  const getAudioLevel = useCallback((): number => {
    return audioInputRef.current?.getAudioLevel() || 0;
  }, []);

  const getConnectionStatus = useCallback((): 'connected' | 'connecting' | 'disconnected' | 'error' => {
    if (state.connectionError) return 'error';
    if (state.isConnecting) return 'connecting';
    if (state.isConnected) return 'connected';
    return 'disconnected';
  }, [state.isConnected, state.isConnecting, state.connectionError]);

  // =============================================================================
  // CONTEXT VALUE MEMOIZATION
  // =============================================================================

  const contextValue = useMemo((): VoiceContextType => ({
    // State
    ...state,

    // Actions
    connect,
    disconnect,
    reconnect,
    startListening,
    stopListening,
    toggleListening,
    speak,
    stopSpeaking,
    clearAudioQueue,
    setVolume,
    clearTranscription,
    clearTranscriptionHistory,
    setMaxHistoryLength,
    updateVoiceSettings,
    requestMicrophonePermission,
    checkBrowserSupport,
    clearAllErrors,
    getAudioLevel,
    getConnectionStatus
  }), [
    state,
    connect,
    disconnect,
    reconnect,
    startListening,
    stopListening,
    toggleListening,
    speak,
    stopSpeaking,
    clearAudioQueue,
    setVolume,
    clearTranscription,
    clearTranscriptionHistory,
    setMaxHistoryLength,
    updateVoiceSettings,
    requestMicrophonePermission,
    checkBrowserSupport,
    clearAllErrors,
    getAudioLevel,
    getConnectionStatus
  ]);

  return (
    <VoiceContext.Provider value={contextValue}>
      {children}
    </VoiceContext.Provider>
  );
}

// =============================================================================
// HOOK EXPORTS
// =============================================================================

/**
 * Primary hook for comprehensive voice functionality
 */
export function useVoiceInteractionCentralized(): VoiceContextType {
  const context = useContext(VoiceContext);
  if (!context) {
    throw new Error('useVoiceInteractionCentralized must be used within a VoiceProvider');
  }
  return context;
}

/**
 * Specialized hook for connection management
 */
export function useVoiceConnection() {
  const {
    isConnected,
    isConnecting,
    connectionError,
    lastConnectionAttempt,
    connect,
    disconnect,
    reconnect,
    getConnectionStatus
  } = useVoiceInteractionCentralized();

  return {
    isConnected,
    isConnecting,
    connectionError,
    lastConnectionAttempt,
    connect,
    disconnect,
    reconnect,
    getConnectionStatus,
    status: getConnectionStatus()
  };
}

/**
 * Specialized hook for audio input management
 */
export function useVoiceInput() {
  const {
    isListening,
    audioLevel,
    audioInputState,
    microphonePermission,
    startListening,
    stopListening,
    toggleListening,
    requestMicrophonePermission,
    getAudioLevel
  } = useVoiceInteractionCentralized();

  return {
    isListening,
    audioLevel,
    audioInputState,
    microphonePermission,
    startListening,
    stopListening,
    toggleListening,
    requestMicrophonePermission,
    getAudioLevel
  };
}

/**
 * Specialized hook for audio output management
 */
export function useVoiceOutput() {
  const {
    isSpeaking,
    audioOutputState,
    volume,
    audioQueueLength,
    speak,
    stopSpeaking,
    clearAudioQueue,
    setVolume
  } = useVoiceInteractionCentralized();

  return {
    isSpeaking,
    audioOutputState,
    volume,
    audioQueueLength,
    speak,
    stopSpeaking,
    clearAudioQueue,
    setVolume
  };
}

/**
 * Specialized hook for transcription management
 */
export function useVoiceTranscription() {
  const {
    transcription,
    partialTranscription,
    transcriptionHistory,
    maxHistoryLength,
    clearTranscription,
    clearTranscriptionHistory,
    setMaxHistoryLength
  } = useVoiceInteractionCentralized();

  return {
    transcription,
    partialTranscription,
    transcriptionHistory,
    maxHistoryLength,
    clearTranscription,
    clearTranscriptionHistory,
    setMaxHistoryLength
  };
}

/**
 * Specialized hook for settings management
 */
export function useVoiceSettings() {
  const {
    voiceSettings,
    updateVoiceSettings
  } = useVoiceInteractionCentralized();

  return {
    voiceSettings,
    updateVoiceSettings
  };
}

/**
 * Utility hook for audio level monitoring
 */
export function useAudioLevel(): number {
  const { audioLevel } = useVoiceInteractionCentralized();
  return audioLevel;
}

/**
 * Utility hook for voice permissions
 */
export function useVoicePermissions() {
  const {
    microphonePermission,
    requestMicrophonePermission
  } = useVoiceInteractionCentralized();

  return {
    ...microphonePermission,
    request: requestMicrophonePermission
  };
}

/**
 * Utility hook for browser support information
 */
export function useVoiceBrowserSupport() {
  const {
    browserSupport,
    checkBrowserSupport
  } = useVoiceInteractionCentralized();

  return {
    ...browserSupport,
    refresh: checkBrowserSupport
  };
}