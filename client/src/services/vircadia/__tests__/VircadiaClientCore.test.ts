/**
 * VircadiaClientCore Tests - Phase 5: Unit Testing
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { ClientCore, ClientCoreConfig } from '../VircadiaClientCore';

describe('VircadiaClientCore', () => {
    let client: ClientCore;
    let mockWebSocket: any;

    const defaultConfig: ClientCoreConfig = {
        serverUrl: 'ws://localhost:3020/world/ws',
        authToken: 'test-token',
        authProvider: 'system',
        reconnectAttempts: 3,
        reconnectDelay: 1000,
        debug: false
    };

    beforeEach(() => {
        // Mock WebSocket
        mockWebSocket = {
            send: vi.fn(),
            close: vi.fn(),
            addEventListener: vi.fn(),
            removeEventListener: vi.fn(),
            readyState: 1 // OPEN
        };

        global.WebSocket = vi.fn(() => mockWebSocket) as any;

        client = new ClientCore(defaultConfig);
    });

    afterEach(() => {
        client.dispose();
        vi.clearAllMocks();
    });

    describe('Connection Management', () => {
        it('should create client with config', () => {
            expect(client).toBeDefined();
            expect(client.Utilities).toBeDefined();
            expect(client.Utilities.Connection).toBeDefined();
        });

        it('should connect to WebSocket server', async () => {
            const connectPromise = client.Utilities.Connection.connect({ timeoutMs: 5000 });

            // Simulate WebSocket open
            const openHandler = mockWebSocket.addEventListener.mock.calls.find(
                (call: any) => call[0] === 'open'
            )?.[1];
            openHandler?.();

            const info = await connectPromise;

            expect(info.isConnected).toBe(true);
            expect(mockWebSocket.send).toHaveBeenCalled();
        });

        it('should handle connection timeout', async () => {
            const connectPromise = client.Utilities.Connection.connect({ timeoutMs: 100 });

            await expect(connectPromise).rejects.toThrow('Connection timeout');
        });

        it('should reconnect on connection drop', async () => {
            // First connection
            const connectPromise = client.Utilities.Connection.connect({ timeoutMs: 5000 });
            const openHandler = mockWebSocket.addEventListener.mock.calls.find(
                (call: any) => call[0] === 'open'
            )?.[1];
            openHandler?.();
            await connectPromise;

            // Simulate connection drop
            const closeHandler = mockWebSocket.addEventListener.mock.calls.find(
                (call: any) => call[0] === 'close'
            )?.[1];
            closeHandler?.();

            // Should attempt reconnect
            expect(global.WebSocket).toHaveBeenCalledTimes(2);
        });

        it('should get connection info', async () => {
            const info = client.Utilities.Connection.getConnectionInfo();
            expect(info).toHaveProperty('isConnected');
            expect(info).toHaveProperty('agentId');
            expect(info).toHaveProperty('sessionId');
        });
    });

    describe('Query System', () => {
        it('should send SQL query', async () => {
            // Connect first
            const connectPromise = client.Utilities.Connection.connect({ timeoutMs: 5000 });
            const openHandler = mockWebSocket.addEventListener.mock.calls.find(
                (call: any) => call[0] === 'open'
            )?.[1];
            openHandler?.();
            await connectPromise;

            // Send query
            const queryPromise = client.Utilities.Connection.query({
                query: 'SELECT * FROM entity.entities',
                timeoutMs: 3000
            });

            // Simulate response
            const messageHandler = mockWebSocket.addEventListener.mock.calls.find(
                (call: any) => call[0] === 'message'
            )?.[1];
            messageHandler?.({
                data: JSON.stringify({
                    type: 'QUERY_RESPONSE',
                    requestId: expect.any(String),
                    result: [{ id: 1, name: 'test' }]
                })
            });

            const result = await queryPromise;
            expect(result).toHaveProperty('result');
            expect(mockWebSocket.send).toHaveBeenCalled();
        });

        it('should handle query timeout', async () => {
            // Connect first
            const connectPromise = client.Utilities.Connection.connect({ timeoutMs: 5000 });
            const openHandler = mockWebSocket.addEventListener.mock.calls.find(
                (call: any) => call[0] === 'open'
            )?.[1];
            openHandler?.();
            await connectPromise;

            // Send query with short timeout
            const queryPromise = client.Utilities.Connection.query({
                query: 'SELECT * FROM entity.entities',
                timeoutMs: 100
            });

            await expect(queryPromise).rejects.toThrow('Query timeout');
        });
    });

    describe('Event System', () => {
        it('should handle status change events', async () => {
            const listener = vi.fn();
            client.Utilities.Connection.addEventListener('statusChange', listener);

            // Connect
            const connectPromise = client.Utilities.Connection.connect({ timeoutMs: 5000 });
            const openHandler = mockWebSocket.addEventListener.mock.calls.find(
                (call: any) => call[0] === 'open'
            )?.[1];
            openHandler?.();
            await connectPromise;

            expect(listener).toHaveBeenCalled();
        });

        it('should handle sync updates', () => {
            const listener = vi.fn();
            client.Utilities.Connection.addEventListener('syncUpdate', listener);

            // Simulate sync message
            const messageHandler = mockWebSocket.addEventListener.mock.calls.find(
                (call: any) => call[0] === 'message'
            )?.[1];
            messageHandler?.({
                data: JSON.stringify({
                    type: 'SYNC_GROUP_UPDATES_RESPONSE',
                    updates: []
                })
            });

            expect(listener).toHaveBeenCalled();
        });

        it('should remove event listeners', () => {
            const listener = vi.fn();
            const unsubscribe = client.Utilities.Connection.addEventListener('statusChange', listener);
            unsubscribe();

            // Status change should not trigger listener
            const openHandler = mockWebSocket.addEventListener.mock.calls.find(
                (call: any) => call[0] === 'open'
            )?.[1];
            openHandler?.();

            expect(listener).not.toHaveBeenCalled();
        });
    });

    describe('Disposal', () => {
        it('should dispose client and close WebSocket', () => {
            client.dispose();
            expect(mockWebSocket.close).toHaveBeenCalled();
        });
    });
});
