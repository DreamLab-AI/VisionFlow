/**
 * Logger utility for consistent logging across the application
 */

const winston = require('winston');
const path = require('path');

const logLevel = process.env.LOG_LEVEL || 'info';
const debugMode = process.argv.includes('--debug');

class Logger {
    constructor(component) {
        this.logger = winston.createLogger({
            level: debugMode ? 'debug' : logLevel,
            format: winston.format.combine(
                winston.format.timestamp(),
                winston.format.errors({ stack: true }),
                winston.format.printf(({ timestamp, level, message, component, ...meta }) => {
                    const metaStr = Object.keys(meta).length ? JSON.stringify(meta) : '';
                    return `[${timestamp}] [${level.toUpperCase()}] [${component}] ${message} ${metaStr}`;
                })
            ),
            defaultMeta: { component },
            transports: [
                new winston.transports.Console({
                    format: winston.format.combine(
                        winston.format.colorize(),
                        winston.format.simple()
                    )
                })
            ]
        });

        // Add file transport in production
        if (process.env.NODE_ENV === 'production') {
            this.logger.add(new winston.transports.File({
                filename: path.join('/workspace/logs', 'agent-control.log'),
                maxsize: 10485760, // 10MB
                maxFiles: 5
            }));
        }
    }

    info(message, ...args) {
        this.logger.info(message, ...args);
    }

    error(message, ...args) {
        this.logger.error(message, ...args);
    }

    warn(message, ...args) {
        this.logger.warn(message, ...args);
    }

    debug(message, ...args) {
        this.logger.debug(message, ...args);
    }
}

module.exports = { Logger };