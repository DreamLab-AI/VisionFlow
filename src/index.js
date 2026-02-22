import { startServer } from './server.js';
import logger from './utils/logger.js';

process.on('uncaughtException', (err) => {
  console.error('Uncaught exception:', err);
  process.exit(1);
});
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled rejection at:', promise, 'reason:', reason);
});

const PORT = process.env.PORT || 3000;
const HOST = process.env.HOST || '0.0.0.0';

startServer(PORT, HOST).then((fastify) => {
  logger.info({ host: HOST, port: PORT }, `JavaScript Solid Server running at http://${HOST}:${PORT}`);

  // Graceful shutdown on SIGTERM / SIGINT
  const signals = ['SIGTERM', 'SIGINT'];
  signals.forEach(signal => {
    process.on(signal, async () => {
      logger.info(`Received ${signal}, shutting down gracefully...`);
      try {
        await fastify.close();
        logger.info('Server closed');
        process.exit(0);
      } catch (err) {
        logger.error('Error during shutdown:', err);
        process.exit(1);
      }
    });
  });
});
