/**
 * PushToTalkService — Controls audio routing between agent commands and voice chat.
 *
 * Two modes driven by a single PTT key (default: Space):
 *   PTT held:    Mic audio → VisionFlow /ws/speech → Turbo Whisper STT → agent commands
 *   PTT released: Mic audio → LiveKit SFU → spatial voice chat with other users
 *
 * The service doesn't capture audio itself — it coordinates AudioInputService,
 * VoiceWebSocketService, and LiveKitVoiceService by toggling routing.
 */

import { createLogger } from '../utils/loggerConfig';

const logger = createLogger('PushToTalkService');

export type PTTMode = 'push' | 'toggle';
export type PTTState = 'idle' | 'commanding' | 'chatting';

export interface PTTConfig {
  /** Key to use for push-to-talk (default: ' ' for Space) */
  key: string;
  /** 'push' = hold to talk to agents, 'toggle' = press to start/stop */
  mode: PTTMode;
  /** Minimum hold duration (ms) before audio is sent (prevents accidental taps) */
  minHoldDuration: number;
  /** Whether voice chat is active when PTT is NOT held */
  voiceChatEnabled: boolean;
}

const DEFAULT_CONFIG: PTTConfig = {
  key: ' ',
  mode: 'push',
  minHoldDuration: 150,
  voiceChatEnabled: true,
};

type PTTEventCallback = (state: PTTState) => void;

export class PushToTalkService {
  private static instance: PushToTalkService;
  private config: PTTConfig;
  private state: PTTState = 'idle';
  private keyDownTime = 0;
  private listeners: Set<PTTEventCallback> = new Set();
  private boundKeyDown: (e: KeyboardEvent) => void;
  private boundKeyUp: (e: KeyboardEvent) => void;
  private userId: string | null = null;
  private wsNotifyCallback: ((pttActive: boolean) => void) | null = null;

  private constructor(config?: Partial<PTTConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.boundKeyDown = this.handleKeyDown.bind(this);
    this.boundKeyUp = this.handleKeyUp.bind(this);
  }

  static getInstance(config?: Partial<PTTConfig>): PushToTalkService {
    if (!PushToTalkService.instance) {
      PushToTalkService.instance = new PushToTalkService(config);
    }
    return PushToTalkService.instance;
  }

  /** Start listening for PTT key events */
  activate(userId: string): void {
    this.userId = userId;
    document.addEventListener('keydown', this.boundKeyDown);
    document.addEventListener('keyup', this.boundKeyUp);
    logger.info(`PTT activated for user ${userId}, key="${this.config.key}", mode=${this.config.mode}`);

    if (this.config.voiceChatEnabled) {
      this.setState('chatting');
    }
  }

  /** Stop listening and reset state */
  deactivate(): void {
    document.removeEventListener('keydown', this.boundKeyDown);
    document.removeEventListener('keyup', this.boundKeyUp);
    this.setState('idle');
    this.userId = null;
    logger.info('PTT deactivated');
  }

  /** Register a callback to notify the server of PTT state changes */
  onServerNotify(callback: (pttActive: boolean) => void): void {
    this.wsNotifyCallback = callback;
  }

  /** Listen for state changes */
  onStateChange(callback: PTTEventCallback): () => void {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }

  getState(): PTTState {
    return this.state;
  }

  getUserId(): string | null {
    return this.userId;
  }

  updateConfig(config: Partial<PTTConfig>): void {
    this.config = { ...this.config, ...config };
    logger.info('PTT config updated', this.config);
  }

  private handleKeyDown(e: KeyboardEvent): void {
    if (e.key !== this.config.key) return;
    if (e.repeat) return; // Ignore key repeat

    // Don't capture PTT when typing in input fields
    const target = e.target as HTMLElement;
    if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable) {
      return;
    }

    e.preventDefault();

    if (this.config.mode === 'push') {
      this.keyDownTime = Date.now();
      this.setState('commanding');
    } else {
      // Toggle mode: press to switch between commanding and chatting
      if (this.state === 'commanding') {
        this.setState(this.config.voiceChatEnabled ? 'chatting' : 'idle');
      } else {
        this.setState('commanding');
      }
    }
  }

  private handleKeyUp(e: KeyboardEvent): void {
    if (e.key !== this.config.key) return;
    if (this.config.mode !== 'push') return; // Only relevant for push mode

    e.preventDefault();

    const holdDuration = Date.now() - this.keyDownTime;

    if (holdDuration < this.config.minHoldDuration) {
      // Too short — treat as accidental tap, revert
      logger.debug(`PTT tap too short (${holdDuration}ms < ${this.config.minHoldDuration}ms), ignoring`);
      this.setState(this.config.voiceChatEnabled ? 'chatting' : 'idle');
      return;
    }

    // Release PTT → switch back to voice chat or idle
    this.setState(this.config.voiceChatEnabled ? 'chatting' : 'idle');
  }

  private setState(newState: PTTState): void {
    if (this.state === newState) return;
    const oldState = this.state;
    this.state = newState;

    logger.debug(`PTT state: ${oldState} → ${newState}`);

    // Notify server of PTT state
    if (this.wsNotifyCallback) {
      this.wsNotifyCallback(newState === 'commanding');
    }

    // Notify listeners
    this.listeners.forEach(cb => {
      try { cb(newState); } catch (err) {
        logger.error('PTT listener error:', err);
      }
    });
  }
}
