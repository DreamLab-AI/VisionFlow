/**
 * Module-level structured logger for server-side code
 *
 * Wraps pino (already available via Fastify) for consistent structured
 * logging outside of request context. Inside request handlers, prefer
 * request.log instead.
 *
 * No new dependencies -- pino ships with fastify.
 */

let pino;
try {
  pino = (await import('pino')).default;
} catch {
  // Fallback: thin console wrapper when pino is not resolvable
  pino = null;
}

function createFallbackLogger() {
  const noop = () => {};
  const levelPriority = { fatal: 60, error: 50, warn: 40, info: 30, debug: 20, trace: 10 };
  const configuredLevel = process.env.LOG_LEVEL || 'info';
  const threshold = levelPriority[configuredLevel] ?? 30;

  function makeMethod(level, consoleFn) {
    return (...args) => {
      if ((levelPriority[level] ?? 30) >= threshold) {
        consoleFn(`[${new Date().toISOString()}] ${level.toUpperCase()}:`, ...args);
      }
    };
  }

  return {
    fatal: makeMethod('fatal', console.error),
    error: makeMethod('error', console.error),
    warn: makeMethod('warn', console.warn),
    info: makeMethod('info', console.info),
    debug: makeMethod('debug', console.debug),
    trace: noop,
    child: () => createFallbackLogger(),
  };
}

const logger = pino
  ? pino({ level: process.env.LOG_LEVEL || 'info' })
  : createFallbackLogger();

export default logger;
