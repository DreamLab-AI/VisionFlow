

interface LogEntry {
  level: 'debug' | 'info' | 'warn' | 'error';
  namespace: string;
  message: string;
  timestamp: string;
  data?: any;
  userAgent?: string;
  url?: string;
  stack?: string;
}

class RemoteLogger {
  private buffer: LogEntry[] = [];
  private flushInterval: number = 1000; 
  private maxBufferSize: number = 50;
  private flushTimer: NodeJS.Timeout | null = null;
  private enabled: boolean = true;
  private serverEndpoint: string;

  constructor() {
    
    
    const apiUrl = (import.meta?.env?.VITE_API_URL) || 'http://visionflow_container:4000';

    this.serverEndpoint = `${apiUrl}/api/client-logs`;
    console.log('[RemoteLogger] Configured endpoint:', this.serverEndpoint);

    
    this.startFlushTimer();

    
    this.interceptConsole();

    
    if (typeof window !== 'undefined') {
      window.addEventListener('beforeunload', () => {
        this.flush(true); 
      });
    }
  }

  
  private interceptConsole(): void {
    const originalConsole = {
      log: console.log,
      debug: console.debug,
      info: console.info,
      warn: console.warn,
      error: console.error
    };

    
    console.log = (...args: any[]) => {
      originalConsole.log(...args);
      this.log('info', 'console', this.formatArgs(args));
    };

    console.debug = (...args: any[]) => {
      originalConsole.debug(...args);
      this.log('debug', 'console', this.formatArgs(args));
    };

    console.info = (...args: any[]) => {
      originalConsole.info(...args);
      this.log('info', 'console', this.formatArgs(args));
    };

    console.warn = (...args: any[]) => {
      originalConsole.warn(...args);
      this.log('warn', 'console', this.formatArgs(args));
    };

    console.error = (...args: any[]) => {
      originalConsole.error(...args);
      this.log('error', 'console', this.formatArgs(args), this.extractStack(args));
    };
  }

  
  private formatArgs(args: any[]): string {
    return args.map(arg => {
      if (typeof arg === 'object') {
        try {
          return JSON.stringify(arg, null, 2);
        } catch (e) {
          return String(arg);
        }
      }
      return String(arg);
    }).join(' ');
  }

  
  private extractStack(args: any[]): string | undefined {
    for (const arg of args) {
      if (arg instanceof Error && arg.stack) {
        return arg.stack;
      }
    }
    return undefined;
  }

  
  public log(
    level: LogEntry['level'],
    namespace: string,
    message: string,
    stack?: string,
    data?: any
  ): void {
    if (!this.enabled) return;

    const entry: LogEntry = {
      level,
      namespace,
      message,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href,
      stack,
      data
    };

    this.buffer.push(entry);

    
    if (this.buffer.length >= this.maxBufferSize) {
      this.flush();
    }
  }

  
  private startFlushTimer(): void {
    if (this.flushTimer) return;

    this.flushTimer = setInterval(() => {
      if (this.buffer.length > 0) {
        this.flush();
      }
    }, this.flushInterval);
  }

  
  private stopFlushTimer(): void {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
      this.flushTimer = null;
    }
  }

  
  public async flush(sync: boolean = false): Promise<void> {
    if (this.buffer.length === 0) return;

    const logs = [...this.buffer];
    this.buffer = [];

    try {
      const payload = {
        logs,
        sessionId: this.getSessionId(),
        timestamp: new Date().toISOString()
      };

      if (sync) {
        
        if (navigator.sendBeacon) {
          const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
          navigator.sendBeacon(this.serverEndpoint, blob);
        }
      } else {
        
        const response = await fetch(this.serverEndpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(payload)
        });

        if (!response.ok) {
          console.error('[RemoteLogger] Failed to send logs:', response.status, response.statusText);
        }
      }
    } catch (error) {
      
      if (!sync) {
        this.buffer = logs.concat(this.buffer);
      }
      
    }
  }

  
  private getSessionId(): string {
    let sessionId = sessionStorage.getItem('remote-logger-session');
    if (!sessionId) {
      sessionId = `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      sessionStorage.setItem('remote-logger-session', sessionId);
    }
    return sessionId;
  }

  
  public setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (!enabled) {
      this.flush(); 
      this.stopFlushTimer();
    } else {
      this.startFlushTimer();
    }
  }

  
  public configure(options: {
    flushInterval?: number;
    maxBufferSize?: number;
    serverEndpoint?: string;
    enabled?: boolean;
  }): void {
    if (options.flushInterval !== undefined) {
      this.flushInterval = options.flushInterval;
      this.stopFlushTimer();
      this.startFlushTimer();
    }

    if (options.maxBufferSize !== undefined) {
      this.maxBufferSize = options.maxBufferSize;
    }

    if (options.serverEndpoint !== undefined) {
      this.serverEndpoint = options.serverEndpoint;
    }

    if (options.enabled !== undefined) {
      this.setEnabled(options.enabled);
    }
  }

  
  public createLogger(namespace: string) {
    return {
      debug: (message: string, data?: any) => this.log('debug', namespace, message, undefined, data),
      info: (message: string, data?: any) => this.log('info', namespace, message, undefined, data),
      warn: (message: string, data?: any) => this.log('warn', namespace, message, undefined, data),
      error: (message: string, error?: Error | any, data?: any) => {
        const stack = error instanceof Error ? error.stack : undefined;
        const errorData = { ...data, error: error instanceof Error ? error.message : error };
        this.log('error', namespace, message, stack, errorData);
      }
    };
  }

  
  public logXRInfo(): void {
    const xrInfo: any = {
      webXRSupported: 'xr' in navigator,
      userAgent: navigator.userAgent,
      protocol: window.location.protocol,
      hostname: window.location.hostname,
      timestamp: new Date().toISOString()
    };

    
    if ('xr' in navigator && navigator.xr) {
      navigator.xr.isSessionSupported('immersive-vr').then(supported => {
        xrInfo.vrSupported = supported;
        this.log('info', 'xr-capabilities', 'VR Support Check', undefined, xrInfo);
      }).catch(e => {
        xrInfo.vrSupportError = e.message;
      });

      navigator.xr.isSessionSupported('immersive-ar').then(supported => {
        xrInfo.arSupported = supported;
        this.log('info', 'xr-capabilities', 'AR Support Check', undefined, xrInfo);
      }).catch(e => {
        xrInfo.arSupportError = e.message;
      });
    }

    
    const isQuest = /OculusBrowser|Quest/i.test(navigator.userAgent);
    xrInfo.isQuestDevice = isQuest;

    if (isQuest) {
      
      const questMatch = navigator.userAgent.match(/Quest\s*(\d+)?/i);
      if (questMatch) {
        xrInfo.questVersion = questMatch[1] || 'Unknown';
      }
    }

    this.log('info', 'xr-detection', 'XR Environment Info', undefined, xrInfo);
  }
}

// Create and export singleton instance
export const remoteLogger = new RemoteLogger();

// Export for convenience
export const createRemoteLogger = (namespace: string) => remoteLogger.createLogger(namespace);

// Log XR info on load
if (typeof window !== 'undefined') {
  
  setTimeout(() => {
    remoteLogger.logXRInfo();
  }, 1000);
}