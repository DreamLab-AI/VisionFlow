const express = require('express');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');

const app = express();
const PORT = 9600;
const WORKER_POOL_SIZE = parseInt(process.env.CLAUDE_WORKER_POOL_SIZE || '4', 10);
const MAX_QUEUE_SIZE = parseInt(process.env.CLAUDE_MAX_QUEUE_SIZE || '50', 10);

app.use(bodyParser.json({ limit: '10mb' }));

// Worker pool implementation
class ClaudeWorkerPool {
    constructor(size) {
        this.size = size;
        this.workers = [];
        this.queue = [];
        this.initWorkers();
    }

    initWorkers() {
        for (let i = 0; i < this.size; i++) {
            this.workers.push({ busy: false, id: i });
        }
    }

    async execute(prompt, timeout = 30000) {
        // Check queue size
        if (this.queue.length >= MAX_QUEUE_SIZE) {
            throw new Error(`Queue full (max ${MAX_QUEUE_SIZE})`);
        }

        return new Promise((resolve, reject) => {
            const request = { prompt, timeout, resolve, reject };
            this.queue.push(request);
            this.processQueue();
        });
    }

    processQueue() {
        if (this.queue.length === 0) return;

        const worker = this.workers.find(w => !w.busy);
        if (!worker) return;

        const request = this.queue.shift();
        worker.busy = true;

        this.runClaude(worker, request)
            .then(result => request.resolve(result))
            .catch(err => request.reject(err))
            .finally(() => {
                worker.busy = false;
                this.processQueue();
            });
    }

    async runClaude(worker, { prompt, timeout }, retryCount = 0) {
        const MAX_RETRIES = 3;
        const BASE_DELAY = 1000; // 1 second

        return new Promise((resolve, reject) => {
            const claudeProcess = spawn('claude', [
                '--dangerously-skip-permissions',
                '--print'
            ], {
                env: {
                    ...process.env,
                    CLAUDE_CONFIG_DIR: '/home/claude/.claude',
                    ANTHROPIC_BASE_URL: process.env.ANTHROPIC_BASE_URL || 'https://api.z.ai/api/anthropic',
                    ANTHROPIC_AUTH_TOKEN: process.env.ANTHROPIC_AUTH_TOKEN
                }
            });

            let stdout = '';
            let stderr = '';
            let timeoutHandle;
            let killed = false;

            timeoutHandle = setTimeout(() => {
                killed = true;
                claudeProcess.kill('SIGTERM');
                reject(new Error(`Request timeout after ${timeout}ms`));
            }, timeout);

            claudeProcess.stdout.on('data', (data) => {
                stdout += data.toString();
            });

            claudeProcess.stderr.on('data', (data) => {
                stderr += data.toString();
            });

            claudeProcess.stdin.write(prompt);
            claudeProcess.stdin.end();

            claudeProcess.on('close', (code) => {
                clearTimeout(timeoutHandle);
                if (killed) return;

                if (code === 0) {
                    resolve({
                        success: true,
                        response: stdout.trim(),
                        stderr: stderr.trim()
                    });
                } else {
                    const error = {
                        success: false,
                        error: 'Claude process failed',
                        code: code,
                        stdout: stdout.trim(),
                        stderr: stderr.trim()
                    };

                    // Retry on transient errors (network, API rate limits)
                    const isRetryable = code === 124 || // Timeout
                                       stderr.includes('ECONNRESET') ||
                                       stderr.includes('ETIMEDOUT') ||
                                       stderr.includes('rate_limit') ||
                                       stderr.includes('429');

                    if (isRetryable && retryCount < MAX_RETRIES) {
                        const delay = BASE_DELAY * Math.pow(2, retryCount); // Exponential backoff
                        console.log(`Retry attempt ${retryCount + 1}/${MAX_RETRIES} after ${delay}ms`);

                        setTimeout(() => {
                            this.runClaude(worker, { prompt, timeout }, retryCount + 1)
                                .then(resolve)
                                .catch(reject);
                        }, delay);
                    } else {
                        reject(error);
                    }
                }
            });

            claudeProcess.on('error', (err) => {
                clearTimeout(timeoutHandle);
                if (killed) return;

                const error = { success: false, error: err.message };

                // Retry on spawn errors
                if (retryCount < MAX_RETRIES && (err.code === 'ECONNREFUSED' || err.code === 'ENOTFOUND')) {
                    const delay = BASE_DELAY * Math.pow(2, retryCount);
                    console.log(`Retry attempt ${retryCount + 1}/${MAX_RETRIES} after ${delay}ms (spawn error)`);

                    setTimeout(() => {
                        this.runClaude(worker, { prompt, timeout }, retryCount + 1)
                            .then(resolve)
                            .catch(reject);
                    }, delay);
                } else {
                    reject(error);
                }
            });
        });
    }

    getStats() {
        return {
            poolSize: this.size,
            busyWorkers: this.workers.filter(w => w.busy).length,
            queueLength: this.queue.length,
            maxQueueSize: MAX_QUEUE_SIZE
        };
    }
}

const pool = new ClaudeWorkerPool(WORKER_POOL_SIZE);

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({
        status: 'ok',
        service: 'claude-zai-wrapper',
        ...pool.getStats()
    });
});

// Main prompt endpoint
app.post('/prompt', async (req, res) => {
    const { prompt, timeout = 30000 } = req.body;

    if (!prompt) {
        return res.status(400).json({ error: 'prompt is required' });
    }

    try {
        const result = await pool.execute(prompt, timeout);
        res.json(result);
    } catch (error) {
        if (error.message && error.message.includes('Queue full')) {
            return res.status(503).json({
                success: false,
                error: error.message,
                ...pool.getStats()
            });
        }
        if (error.message && error.message.includes('timeout')) {
            return res.status(408).json({
                success: false,
                error: error.message
            });
        }
        res.status(500).json(error.success !== undefined ? error : {
            success: false,
            error: error.message
        });
    }
});

app.listen(PORT, '0.0.0.0', () => {
    console.log(`Z.AI Claude Code wrapper listening on port ${PORT}`);
    console.log(`Worker pool size: ${WORKER_POOL_SIZE}`);
    console.log(`Max queue size: ${MAX_QUEUE_SIZE}`);
});
