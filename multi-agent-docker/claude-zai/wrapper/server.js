const express = require('express');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');

const app = express();
const PORT = 9600;

app.use(bodyParser.json({ limit: '10mb' }));

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ status: 'ok', service: 'claude-zai-wrapper' });
});

// Main prompt endpoint
app.post('/prompt', async (req, res) => {
    const { prompt, timeout = 30000 } = req.body;

    if (!prompt) {
        return res.status(400).json({ error: 'prompt is required' });
    }

    try {
        const claudeProcess = spawn('claude', [
            '--dangerously-skip-permissions',
            '--print'
        ], {
            env: {
                ...process.env,
                CLAUDE_CONFIG_DIR: '/root/.claude'
            }
        });

        let stdout = '';
        let stderr = '';
        let timeoutHandle;

        // Set timeout
        timeoutHandle = setTimeout(() => {
            claudeProcess.kill();
            res.status(408).json({
                error: 'Request timeout',
                timeout: timeout
            });
        }, timeout);

        // Collect output
        claudeProcess.stdout.on('data', (data) => {
            stdout += data.toString();
        });

        claudeProcess.stderr.on('data', (data) => {
            stderr += data.toString();
        });

        // Write prompt to stdin
        claudeProcess.stdin.write(prompt);
        claudeProcess.stdin.end();

        // Handle completion
        claudeProcess.on('close', (code) => {
            clearTimeout(timeoutHandle);

            if (code === 0) {
                res.json({
                    success: true,
                    response: stdout.trim(),
                    stderr: stderr.trim()
                });
            } else {
                res.status(500).json({
                    success: false,
                    error: 'Claude process failed',
                    code: code,
                    stderr: stderr.trim()
                });
            }
        });

        claudeProcess.on('error', (err) => {
            clearTimeout(timeoutHandle);
            res.status(500).json({
                success: false,
                error: err.message
            });
        });

    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

app.listen(PORT, '0.0.0.0', () => {
    console.log(`Z.AI Claude Code wrapper listening on port ${PORT}`);
});
