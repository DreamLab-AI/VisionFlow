

// Moved from loggerConfig.ts - utility functions for error handling
export function createErrorMetadata(error: unknown): Record<string, any> {
  if (error instanceof Error) {
    return {
      message: error.message,
      name: error.name,
      stack: error.stack,
    };
  }
  if (typeof error === 'object' && error !== null) {
    try {
      const errorKeys = Object.getOwnPropertyNames(error);
      const serializableError = errorKeys.reduce((acc, key) => {
        acc[key] = (error as any)[key];
        return acc;
      }, {} as Record<string, any>);

      const serializedErrorString = JSON.stringify(serializableError, null, 2);
      return {
        message: `Non-Error object encountered. Details: ${serializedErrorString.substring(0, 500)}${serializedErrorString.length > 500 ? '...' : ''}`,
        originalErrorType: 'Object',
      };
    } catch (e) {
      return {
        message: `Non-Error object (serialization failed): ${String(error)}`,
        originalErrorType: typeof error,
      };
    }
  }
  return {
    message: `Unknown error type: ${String(error)}`,
    originalErrorType: typeof error,
  };
}

export function createDataMetadata(data: Record<string, any>): Record<string, any> {
  return {
    ...data,
    timestamp: new Date().toISOString(),
  };
}

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

const LOG_LEVEL_PRIORITY: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

const LOG_COLORS = {
  debug: '#8c8c8c', 
  info: '#4c9aff',  
  warn: '#ffab00',  
  error: '#ff5630', 
};

// Central log storage
const logStorage: LogEntry[] = [];
const DEFAULT_MAX_LOG_ENTRIES = 1000;

export interface LogEntry {
  timestamp: string;
  level: LogLevel;
  namespace: string;
  message: string;
  args: any[];
}

export interface LoggerOptions {
  disabled?: boolean;
  level?: LogLevel;
  maxLogEntries?: number;
  respectRuntimeSettings?: boolean;
}


function getEnvironmentLogLevel(): LogLevel {
  
  const envLogLevel = import.meta.env?.VITE_LOG_LEVEL || import.meta.env?.LOG_LEVEL;

  
  const level = envLogLevel?.toLowerCase();

  if (isValidLogLevel(level)) {
    return level as LogLevel;
  }

  
  const isDev = import.meta.env?.DEV || import.meta.env?.MODE === 'development';
  return isDev ? 'debug' : 'info';
}


function isValidLogLevel(level: any): level is LogLevel {
  const validLevels: LogLevel[] = ['debug', 'info', 'warn', 'error'];
  return validLevels.includes(level);
}


export function createLogger(namespace: string, options: LoggerOptions = {}) {
  const { disabled = false, level = getEnvironmentLogLevel(), maxLogEntries = DEFAULT_MAX_LOG_ENTRIES } = options;
  let currentDisabled = disabled;
  let currentLevel = level;
  let levelPriority = LOG_LEVEL_PRIORITY[level];

  function shouldLog(msgLevel: LogLevel): boolean {
    if (currentDisabled) return false;
    return LOG_LEVEL_PRIORITY[msgLevel] >= levelPriority;
  }

  function updateLevel(newLevel: LogLevel): void {
    currentLevel = newLevel;
    levelPriority = LOG_LEVEL_PRIORITY[newLevel];
  }

  function setEnabled(enabled: boolean): void {
    currentDisabled = !enabled;
  }

  function getCurrentConfig(): { level: LogLevel; enabled: boolean } {
    return {
      level: currentLevel,
      enabled: !currentDisabled
    };
  }

  function formatMessage(message: any): string {
    if (typeof message === 'string') return message;
    if (message instanceof Error) {
      return message.stack ? message.stack : message.message;
    }
    try {
      return JSON.stringify(message, (key, value) => {
        if (typeof value === 'object' && value !== null) {
          if (value === message && key !== '') return '[Circular Reference]';
        }
        return value;
      }, 2);
    } catch (e) {
      return String(message);
    }
  }

  function formatArgs(args: any[]): any[] {
    return args.map(arg => {
      if (arg instanceof Error) {
        return { message: arg.message, name: arg.name, stack: arg.stack };
      }
      return arg;
    });
  }

  function createLogMethod(logLevel: LogLevel) {
    return function(message: any, ...args: any[]) {
      if (!shouldLog(logLevel)) return;

      const color = LOG_COLORS[logLevel];
      const now = new Date();
      const timestamp = now.toISOString();
      const consoleTimestamp = now.toISOString().split('T')[1].slice(0, -1);
      const prefix = `%c[${consoleTimestamp}] [${namespace}]`;

      const formattedArgs = formatArgs(args);
      const formattedMessage = formatMessage(message);

      logStorage.push({
        timestamp,
        level: logLevel,
        namespace,
        message: formattedMessage,
        args: formattedArgs,
      });

      if (logStorage.length > maxLogEntries) {
        logStorage.shift();
      }

      console[logLevel === 'debug' ? 'log' : logLevel](
        `${prefix} ${formattedMessage}`,
        `color: ${color}; font-weight: bold;`,
        ...args
      );
    };
  }

  function getLogs(): LogEntry[] {
    return [...logStorage];
  }

  return {
    debug: createLogMethod('debug'),
    info: createLogMethod('info'),
    warn: createLogMethod('warn'),
    error: createLogMethod('error'),
    getLogs,
    updateLevel,
    setEnabled,
    getCurrentConfig,
    isEnabled: () => !currentDisabled,
    namespace,
  };
}

export const DEFAULT_LOG_LEVEL = getEnvironmentLogLevel();
