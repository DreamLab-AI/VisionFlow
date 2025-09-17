import { createLogger } from '../../../utils/logger';
import { LogEntry } from '../types';

const logger = createLogger('DockerLogParser');

export interface DockerLogLine {
  log: string;
  stream: 'stdout' | 'stderr';
  time: string;
}

export interface ParsedDockerLog {
  timestamp: Date;
  level: LogEntry['level'];
  source: string;
  message: string;
  agentId?: string;
  metadata?: Record<string, any>;
  rawLine: string;
}

export class DockerLogParser {
  private static logLevelPatterns = [
    { pattern: /\b(CRITICAL|FATAL)\b/i, level: 'critical' as const },
    { pattern: /\b(ERROR|ERR)\b/i, level: 'error' as const },
    { pattern: /\b(WARN|WARNING)\b/i, level: 'warn' as const },
    { pattern: /\b(INFO|INFORMATION)\b/i, level: 'info' as const },
    { pattern: /\b(DEBUG|DBG|TRACE)\b/i, level: 'debug' as const }
  ];

  private static agentIdPatterns = [
    /agent[_-]?id[:\s=]+([a-zA-Z0-9_-]+)/i,
    /\[agent[:\s]+([a-zA-Z0-9_-]+)\]/i,
    /agent[:\s]+([a-zA-Z0-9_-]{8,})/i,
    /\b([a-zA-Z0-9_-]{8,})\s*:\s*agent/i
  ];

  private static sourcePatterns = [
    /^\[([^\]]+)\]/,
    /^([A-Z][a-zA-Z0-9_]*)\s*:/,
    /source[:\s=]+([a-zA-Z0-9_.-]+)/i
  ];

  /**
   * Parse a raw Docker log line (JSON format)
   */
  static parseDockerLogLine(rawLine: string): DockerLogLine | null {
    try {
      const parsed = JSON.parse(rawLine);
      if (parsed.log && parsed.time) {
        return {
          log: parsed.log.trim(),
          stream: parsed.stream || 'stdout',
          time: parsed.time
        };
      }
    } catch (error) {
      logger.debug('Failed to parse Docker log line as JSON:', error);
    }

    // Try to parse as plain text with timestamp
    const textMatch = rawLine.match(/^(\d{4}-\d{2}-\d{2}T[\d:.]+Z?)\s+(.+)$/);
    if (textMatch) {
      return {
        log: textMatch[2].trim(),
        stream: 'stdout',
        time: textMatch[1]
      };
    }

    return null;
  }

  /**
   * Parse Docker log line into structured log entry
   */
  static parseLogLine(dockerLog: DockerLogLine): ParsedDockerLog {
    const timestamp = new Date(dockerLog.time);
    let message = dockerLog.log;
    let level: LogEntry['level'] = 'info';
    let source = dockerLog.stream;
    let agentId: string | undefined;
    const metadata: Record<string, any> = {};

    // Extract log level
    for (const { pattern, level: logLevel } of this.logLevelPatterns) {
      if (pattern.test(message)) {
        level = logLevel;
        break;
      }
    }

    // Extract agent ID
    for (const pattern of this.agentIdPatterns) {
      const match = message.match(pattern);
      if (match) {
        agentId = match[1];
        break;
      }
    }

    // Extract source
    for (const pattern of this.sourcePatterns) {
      const match = message.match(pattern);
      if (match) {
        source = match[1];
        // Remove source prefix from message
        message = message.replace(pattern, '').trim();
        break;
      }
    }

    // Extract metadata patterns
    const metadataPatterns = [
      { name: 'requestId', pattern: /request[_-]?id[:\s=]+([a-zA-Z0-9_-]+)/i },
      { name: 'taskId', pattern: /task[_-]?id[:\s=]+([a-zA-Z0-9_-]+)/i },
      { name: 'duration', pattern: /duration[:\s=]+(\d+(?:\.\d+)?)\s*(ms|s|seconds?)/i },
      { name: 'statusCode', pattern: /status[:\s=]+(\d+)/i },
      { name: 'userId', pattern: /user[_-]?id[:\s=]+([a-zA-Z0-9_-]+)/i }
    ];

    for (const { name, pattern } of metadataPatterns) {
      const match = message.match(pattern);
      if (match) {
        metadata[name] = match[1];
        if (name === 'duration' && match[2]) {
          metadata[`${name}Unit`] = match[2];
        }
      }
    }

    // Clean up message
    message = message
      .replace(/\s+/g, ' ')
      .replace(/^[\s\-:]+/, '')
      .trim();

    return {
      timestamp,
      level,
      source,
      message,
      agentId,
      metadata: Object.keys(metadata).length > 0 ? metadata : undefined,
      rawLine: dockerLog.log
    };
  }

  /**
   * Parse multiple Docker log lines from a string
   */
  static parseLogString(logString: string): ParsedDockerLog[] {
    const lines = logString.split('\n').filter(line => line.trim());
    const parsedLogs: ParsedDockerLog[] = [];

    for (const line of lines) {
      const dockerLog = this.parseDockerLogLine(line);
      if (dockerLog) {
        parsedLogs.push(this.parseLogLine(dockerLog));
      }
    }

    return parsedLogs;
  }

  /**
   * Parse log file and return structured entries
   */
  static async parseLogFile(file: File): Promise<ParsedDockerLog[]> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onload = (event) => {
        try {
          const content = event.target?.result as string;
          const parsed = this.parseLogString(content);
          resolve(parsed);
        } catch (error) {
          reject(error);
        }
      };

      reader.onerror = () => reject(new Error('Failed to read log file'));
      reader.readAsText(file);
    });
  }

  /**
   * Convert parsed logs to LogEntry format
   */
  static toLogEntries(parsedLogs: ParsedDockerLog[]): LogEntry[] {
    return parsedLogs.map((log, index) => ({
      id: `docker-${log.timestamp.getTime()}-${index}`,
      timestamp: log.timestamp,
      level: log.level,
      source: log.source,
      message: log.message,
      agentId: log.agentId,
      metadata: {
        ...log.metadata,
        dockerStream: log.rawLine.length > log.message.length ? 'parsed' : 'raw',
        originalLine: log.rawLine
      }
    }));
  }

  /**
   * Filter logs by criteria
   */
  static filterLogs(
    logs: ParsedDockerLog[],
    filters: {
      level?: LogEntry['level'][];
      agentIds?: string[];
      sources?: string[];
      timeRange?: { start: Date; end: Date };
      searchText?: string;
    }
  ): ParsedDockerLog[] {
    return logs.filter(log => {
      // Level filter
      if (filters.level && filters.level.length > 0 && !filters.level.includes(log.level)) {
        return false;
      }

      // Agent ID filter
      if (filters.agentIds && filters.agentIds.length > 0 && log.agentId && !filters.agentIds.includes(log.agentId)) {
        return false;
      }

      // Source filter
      if (filters.sources && filters.sources.length > 0 && !filters.sources.includes(log.source)) {
        return false;
      }

      // Time range filter
      if (filters.timeRange) {
        if (log.timestamp < filters.timeRange.start || log.timestamp > filters.timeRange.end) {
          return false;
        }
      }

      // Search text filter
      if (filters.searchText) {
        const searchText = filters.searchText.toLowerCase();
        const searchableText = [
          log.message,
          log.source,
          log.agentId,
          JSON.stringify(log.metadata)
        ].join(' ').toLowerCase();

        if (!searchableText.includes(searchText)) {
          return false;
        }
      }

      return true;
    });
  }

  /**
   * Get log statistics
   */
  static getLogStats(logs: ParsedDockerLog[]) {
    const stats = {
      total: logs.length,
      byLevel: {} as Record<LogEntry['level'], number>,
      bySource: {} as Record<string, number>,
      byAgent: {} as Record<string, number>,
      timeRange: {
        start: logs.length > 0 ? logs[0].timestamp : null,
        end: logs.length > 0 ? logs[logs.length - 1].timestamp : null
      },
      errorRate: 0
    };

    for (const log of logs) {
      // Count by level
      stats.byLevel[log.level] = (stats.byLevel[log.level] || 0) + 1;

      // Count by source
      stats.bySource[log.source] = (stats.bySource[log.source] || 0) + 1;

      // Count by agent
      if (log.agentId) {
        stats.byAgent[log.agentId] = (stats.byAgent[log.agentId] || 0) + 1;
      }

      // Update time range
      if (!stats.timeRange.start || log.timestamp < stats.timeRange.start) {
        stats.timeRange.start = log.timestamp;
      }
      if (!stats.timeRange.end || log.timestamp > stats.timeRange.end) {
        stats.timeRange.end = log.timestamp;
      }
    }

    // Calculate error rate
    const errorLogs = (stats.byLevel.error || 0) + (stats.byLevel.critical || 0);
    stats.errorRate = stats.total > 0 ? (errorLogs / stats.total) * 100 : 0;

    return stats;
  }
}

export default DockerLogParser;