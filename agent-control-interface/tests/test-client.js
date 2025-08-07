#!/usr/bin/env node

/**
 * Test Client for Agent Control Interface
 * 
 * Simulates VisionFlow client connections and tests the JSON-RPC API.
 */

const net = require('net');
const readline = require('readline');

class TestClient {
    constructor() {
        this.socket = null;
        this.requestId = 1;
        this.pendingRequests = new Map();
    }

    connect(host = 'localhost', port = 9500) {
        return new Promise((resolve, reject) => {
            console.log(`Connecting to ${host}:${port}...`);
            
            this.socket = net.createConnection({ host, port }, () => {
                console.log('✓ Connected to Agent Control Interface');
                resolve();
            });

            this.socket.on('data', (data) => {
                this.handleResponse(data.toString());
            });

            this.socket.on('error', (err) => {
                console.error('Connection error:', err.message);
                reject(err);
            });

            this.socket.on('close', () => {
                console.log('Connection closed');
                process.exit(0);
            });
        });
    }

    handleResponse(data) {
        const lines = data.split('\n').filter(line => line.trim());
        
        for (const line of lines) {
            try {
                const response = JSON.parse(line);
                console.log('\n← Response:', JSON.stringify(response, null, 2));
                
                if (response.id && this.pendingRequests.has(response.id)) {
                    const { resolve, reject } = this.pendingRequests.get(response.id);
                    this.pendingRequests.delete(response.id);
                    
                    if (response.error) {
                        reject(new Error(response.error.message));
                    } else {
                        resolve(response.result);
                    }
                }
            } catch (e) {
                console.error('Failed to parse response:', e.message);
            }
        }
    }

    sendRequest(method, params = {}) {
        return new Promise((resolve, reject) => {
            const id = (this.requestId++).toString();
            const request = {
                jsonrpc: '2.0',
                id,
                method,
                params
            };
            
            console.log('\n→ Request:', JSON.stringify(request, null, 2));
            
            this.pendingRequests.set(id, { resolve, reject });
            this.socket.write(JSON.stringify(request) + '\n');
            
            // Timeout after 5 seconds
            setTimeout(() => {
                if (this.pendingRequests.has(id)) {
                    this.pendingRequests.delete(id);
                    reject(new Error('Request timeout'));
                }
            }, 5000);
        });
    }

    async runTests() {
        console.log('\n====================================');
        console.log('Running API Tests');
        console.log('====================================\n');
        
        try {
            // Test 1: Initialize
            console.log('Test 1: Initialize Session');
            console.log('--------------------------');
            await this.sendRequest('initialize', {
                protocolVersion: '0.1.0',
                clientInfo: {
                    name: 'test-client',
                    version: '1.0.0'
                }
            });
            
            // Test 2: List Agents
            console.log('\nTest 2: List Agents');
            console.log('-------------------');
            await this.sendRequest('agents/list', {});
            
            // Test 3: Initialize Swarm
            console.log('\nTest 3: Initialize Swarm');
            console.log('------------------------');
            await this.sendRequest('tools/call', {
                name: 'swarm.initialize',
                arguments: {
                    topology: 'hierarchical',
                    agentTypes: ['coordinator', 'coder', 'tester']
                }
            });
            
            // Test 4: Get Visualization Snapshot
            console.log('\nTest 4: Get Visualization Snapshot');
            console.log('----------------------------------');
            await this.sendRequest('tools/call', {
                name: 'visualization.snapshot',
                arguments: {
                    includePositions: false,
                    includeConnections: true
                }
            });
            
            // Test 5: Get System Metrics
            console.log('\nTest 5: Get System Metrics');
            console.log('--------------------------');
            await this.sendRequest('tools/call', {
                name: 'metrics.get',
                arguments: {
                    includeAgents: true,
                    includePerformance: true
                }
            });
            
            console.log('\n====================================');
            console.log('All tests completed successfully! ✓');
            console.log('====================================\n');
            
        } catch (error) {
            console.error('\n✗ Test failed:', error.message);
        }
    }

    async runInteractive() {
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });

        console.log('\nInteractive Mode');
        console.log('================');
        console.log('Commands:');
        console.log('  init         - Initialize session');
        console.log('  agents       - List agents');
        console.log('  swarm        - Initialize swarm');
        console.log('  snapshot     - Get visualization snapshot');
        console.log('  metrics      - Get system metrics');
        console.log('  test         - Run all tests');
        console.log('  quit         - Exit\n');

        const prompt = () => {
            rl.question('> ', async (command) => {
                try {
                    switch (command.trim().toLowerCase()) {
                        case 'init':
                            await this.sendRequest('initialize', {
                                protocolVersion: '0.1.0',
                                clientInfo: { name: 'interactive', version: '1.0.0' }
                            });
                            break;
                        case 'agents':
                            await this.sendRequest('agents/list', {});
                            break;
                        case 'swarm':
                            await this.sendRequest('tools/call', {
                                name: 'swarm.initialize',
                                arguments: { topology: 'mesh' }
                            });
                            break;
                        case 'snapshot':
                            await this.sendRequest('tools/call', {
                                name: 'visualization.snapshot',
                                arguments: { includeConnections: true }
                            });
                            break;
                        case 'metrics':
                            await this.sendRequest('tools/call', {
                                name: 'metrics.get',
                                arguments: { includeAgents: true }
                            });
                            break;
                        case 'test':
                            await this.runTests();
                            break;
                        case 'quit':
                        case 'exit':
                            console.log('Goodbye!');
                            process.exit(0);
                            break;
                        default:
                            console.log('Unknown command. Type "quit" to exit.');
                    }
                } catch (error) {
                    console.error('Error:', error.message);
                }
                prompt();
            });
        };

        prompt();
    }
}

// Main execution
async function main() {
    const client = new TestClient();
    
    try {
        // Parse command line arguments
        const args = process.argv.slice(2);
        const host = args[0] || 'localhost';
        const port = parseInt(args[1]) || 9500;
        const mode = args[2] || 'test';
        
        await client.connect(host, port);
        
        if (mode === 'interactive') {
            await client.runInteractive();
        } else {
            await client.runTests();
            process.exit(0);
        }
        
    } catch (error) {
        console.error('Failed to connect:', error.message);
        console.log('\nUsage: node test-client.js [host] [port] [mode]');
        console.log('  host: Server hostname (default: localhost)');
        console.log('  port: Server port (default: 9500)');
        console.log('  mode: "test" or "interactive" (default: test)');
        console.log('\nExample: node test-client.js 172.18.0.10 9500 interactive');
        process.exit(1);
    }
}

main();