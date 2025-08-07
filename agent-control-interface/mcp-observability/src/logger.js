import winston from 'winston';

const { createLogger: createWinstonLogger, format, transports } = winston;

// Create logger factory
export function createLogger(name) {
  return createWinstonLogger({
    level: process.env.LOG_LEVEL || 'info',
    format: format.combine(
      format.timestamp({
        format: 'YYYY-MM-DD HH:mm:ss'
      }),
      format.errors({ stack: true }),
      format.splat(),
      format.json(),
      format.printf(({ timestamp, level, message, name, ...metadata }) => {
        let msg = `${timestamp} [${level.toUpperCase()}] [${name}] ${message}`;
        
        // Add metadata if present
        if (Object.keys(metadata).length > 0) {
          msg += ` ${JSON.stringify(metadata)}`;
        }
        
        return msg;
      })
    ),
    defaultMeta: { name },
    transports: [
      new transports.Console({
        format: format.combine(
          format.colorize(),
          format.simple()
        )
      })
    ]
  });
}