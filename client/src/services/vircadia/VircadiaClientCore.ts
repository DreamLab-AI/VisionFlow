

export type ClientCoreConnectionState =
    | "connected"
    | "connecting"
    | "reconnecting"
    | "disconnected";

export interface ClientCoreConnectionInfo {
    status: ClientCoreConnectionState;
    isConnected: boolean;
    isConnecting: boolean;
    isReconnecting: boolean;
    connectionDuration?: number;
    reconnectAttempts: number;
    pendingRequests: Array<{
        requestId: string;
        elapsedMs: number;
    }>;
    agentId: string | null;
    sessionId: string | null;
}

export type ClientCoreConnectionEventListener = () => void;

export interface ClientCoreConfig {
    serverUrl: string;
    authToken: string;
    authProvider: string;
    reconnectAttempts?: number;
    reconnectDelay?: number;
    debug?: boolean;
    suppress?: boolean;
}

export interface QueryOptions {
    query: string;
    parameters?: unknown[];
    timeoutMs?: number;
}

export interface QueryResult<T = unknown> {
    success: boolean;
    result?: T;
    errorMessage?: string;
    timestamp: number;
}

// WebSocket message types from Vircadia schema
export enum MessageType {
    GENERAL_ERROR_RESPONSE = "GENERAL_ERROR_RESPONSE",
    QUERY_REQUEST = "QUERY_REQUEST",
    QUERY_RESPONSE = "QUERY_RESPONSE",
    SYNC_GROUP_UPDATES_RESPONSE = "SYNC_GROUP_UPDATES_RESPONSE",
    TICK_NOTIFICATION_RESPONSE = "TICK_NOTIFICATION_RESPONSE",
    SESSION_INFO_RESPONSE = "SESSION_INFO_RESPONSE"
}

export interface WebSocketMessage {
    type: MessageType;
    timestamp: number;
    requestId?: string;
    errorMessage?: string | null;
}

export interface QueryRequest extends WebSocketMessage {
    type: MessageType.QUERY_REQUEST;
    query: string;
    parameters: unknown[];
}

export interface QueryResponse<T = unknown> extends WebSocketMessage {
    type: MessageType.QUERY_RESPONSE;
    result?: T;
}

export interface SessionInfoResponse extends WebSocketMessage {
    type: MessageType.SESSION_INFO_RESPONSE;
    agentId: string;
    sessionId: string;
}

const debugLog = (config: ClientCoreConfig, message: string, ...args: unknown[]) => {
    if (config.debug && !config.suppress) {
        console.log(`[VircadiaClient] ${message}`, ...args);
    }
};

const debugError = (config: ClientCoreConfig, message: string, ...args: unknown[]) => {
    if (!config.suppress) {
        console.error(`[VircadiaClient] ${message}`, ...args);
    }
};

class CoreConnectionManager {
    private ws: WebSocket | null = null;
    private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    private heartbeatTimer: ReturnType<typeof setTimeout> | null = null;
    private reconnectCount = 0;
    private lastHeartbeatTime: number = 0;
    private pendingRequests = new Map<
        string,
        {
            resolve: (value: unknown) => void;
            reject: (reason: unknown) => void;
            timeout: ReturnType<typeof setTimeout>;
        }
    >();
    private eventListeners = new Map<string, Set<ClientCoreConnectionEventListener>>();
    private lastStatus: ClientCoreConnectionState = "disconnected";
    private connectionStartTime: number | null = null;
    private connectionPromise: Promise<ClientCoreConnectionInfo> | null = null;
    private agentId: string | null = null;
    private sessionId: string | null = null;
    private readonly HEARTBEAT_INTERVAL_MS = 30000; 
    private readonly HEARTBEAT_TIMEOUT_MS = 10000;  

    constructor(private config: ClientCoreConfig) {}

    addEventListener(event: string, listener: ClientCoreConnectionEventListener): void {
        if (!this.eventListeners.has(event)) {
            this.eventListeners.set(event, new Set());
        }
        this.eventListeners.get(event)?.add(listener);
    }

    removeEventListener(event: string, listener: ClientCoreConnectionEventListener): void {
        this.eventListeners.get(event)?.delete(listener);
    }

    private emit(event: string): void {
        this.eventListeners.get(event)?.forEach(listener => listener());
    }

    private updateStatus(newStatus: ClientCoreConnectionState): void {
        if (this.lastStatus !== newStatus) {
            this.lastStatus = newStatus;
            this.emit("statusChange");
            debugLog(this.config, `Connection status changed to: ${newStatus}`);
        }
    }

    connect(options?: { timeoutMs?: number }): Promise<ClientCoreConnectionInfo> {
        if (this.connectionPromise) {
            return this.connectionPromise;
        }

        this.connectionPromise = new Promise((resolve, reject) => {
            const timeoutMs = options?.timeoutMs || 30000;
            const timeoutTimer = setTimeout(() => {
                this.disconnect();
                reject(new Error("Connection timeout"));
            }, timeoutMs);

            try {
                this.updateStatus("connecting");
                this.connectionStartTime = Date.now();

                const url = new URL(this.config.serverUrl);

                debugLog(this.config, `Connecting to: ${url.toString()}`);

                this.ws = new WebSocket(url.toString());

                this.ws.onopen = () => {
                    clearTimeout(timeoutTimer);
                    // Send auth credentials as first message instead of URL params
                    this.ws?.send(JSON.stringify({
                        type: 'auth',
                        token: this.config.authToken,
                        provider: this.config.authProvider
                    }));
                    this.updateStatus("connected");
                    this.reconnectCount = 0;
                    this.startHeartbeat();
                    debugLog(this.config, "WebSocket connected");
                    resolve(this.getConnectionInfo());
                };

                this.ws.onmessage = (event) => {
                    this.handleMessage(event.data);
                };

                this.ws.onerror = (error) => {
                    clearTimeout(timeoutTimer);
                    debugError(this.config, "WebSocket error:", error);
                    this.connectionPromise = null;
                    reject(error);
                };

                this.ws.onclose = () => {
                    clearTimeout(timeoutTimer);
                    this.updateStatus("disconnected");
                    this.handleReconnection();
                };

            } catch (error) {
                clearTimeout(timeoutTimer);
                this.connectionPromise = null;
                debugError(this.config, "Connection error:", error);
                reject(error);
            }
        });

        return this.connectionPromise;
    }

    private handleMessage(data: string): void {
        try {
            const message: WebSocketMessage = JSON.parse(data);
            debugLog(this.config, "Received message:", message.type);

            switch (message.type) {
                case MessageType.SESSION_INFO_RESPONSE: {
                    const sessionInfo = message as SessionInfoResponse;
                    this.agentId = sessionInfo.agentId;
                    this.sessionId = sessionInfo.sessionId;
                    debugLog(this.config, `Session info: agentId=${this.agentId}, sessionId=${this.sessionId}`);
                    break;
                }

                case MessageType.QUERY_RESPONSE: {
                    const response = message as QueryResponse;
                    if (response.requestId) {
                        const pending = this.pendingRequests.get(response.requestId);
                        if (pending) {
                            clearTimeout(pending.timeout);
                            this.pendingRequests.delete(response.requestId);

                            if (response.errorMessage) {
                                pending.reject(new Error(response.errorMessage));
                            } else {
                                pending.resolve(response);
                            }
                        }
                    }
                    break;
                }

                case MessageType.SYNC_GROUP_UPDATES_RESPONSE:
                    this.emit("syncUpdate");
                    break;

                case MessageType.TICK_NOTIFICATION_RESPONSE:
                    this.emit("tick");
                    break;

                case MessageType.GENERAL_ERROR_RESPONSE:
                    debugError(this.config, "Server error:", message.errorMessage);
                    this.emit("error");
                    break;
            }
        } catch (error) {
            debugError(this.config, "Failed to parse message:", error);
        }
    }

    private handleReconnection(): void {
        if (this.reconnectCount >= (this.config.reconnectAttempts || 5)) {
            debugLog(this.config, "Max reconnection attempts reached");
            this.connectionPromise = null;
            return;
        }

        this.updateStatus("reconnecting");
        this.reconnectCount++;

        const delay = this.config.reconnectDelay || 5000;
        debugLog(this.config, `Reconnecting in ${delay}ms (attempt ${this.reconnectCount})`);

        this.reconnectTimer = setTimeout(() => {
            this.connectionPromise = null;
            this.connect();
        }, delay);
    }

    disconnect(): void {
        this.stopHeartbeat();

        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }

        this.pendingRequests.forEach(({ timeout }) => clearTimeout(timeout));
        this.pendingRequests.clear();

        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }

        this.updateStatus("disconnected");
        this.connectionPromise = null;
        this.agentId = null;
        this.sessionId = null;
    }

    
    private startHeartbeat(): void {
        this.stopHeartbeat();
        this.lastHeartbeatTime = Date.now();

        this.heartbeatTimer = setInterval(() => {
            if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
                this.stopHeartbeat();
                return;
            }

            const timeSinceLastHeartbeat = Date.now() - this.lastHeartbeatTime;

            
            if (timeSinceLastHeartbeat > this.HEARTBEAT_INTERVAL_MS + this.HEARTBEAT_TIMEOUT_MS) {
                debugLog(this.config, "Heartbeat timeout - connection appears stale");
                this.stopHeartbeat();
                this.ws?.close();
                return;
            }

            
            this.query({ query: "SELECT 1 as heartbeat", timeoutMs: this.HEARTBEAT_TIMEOUT_MS })
                .then(() => {
                    this.lastHeartbeatTime = Date.now();
                    debugLog(this.config, "Heartbeat successful");
                })
                .catch((error) => {
                    debugError(this.config, "Heartbeat failed:", error);
                    this.stopHeartbeat();
                    this.ws?.close();
                });

        }, this.HEARTBEAT_INTERVAL_MS);

        debugLog(this.config, "Heartbeat started");
    }

    
    private stopHeartbeat(): void {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = null;
            debugLog(this.config, "Heartbeat stopped");
        }
    }

    query<T = unknown>(options: QueryOptions): Promise<QueryResult<T>> {
        return new Promise((resolve, reject) => {
            if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
                reject(new Error("WebSocket not connected"));
                return;
            }

            const requestId = crypto.randomUUID();
            const timeoutMs = options.timeoutMs || 10000;

            const timeout = setTimeout(() => {
                this.pendingRequests.delete(requestId);
                reject(new Error("Query timeout"));
            }, timeoutMs);

            this.pendingRequests.set(requestId, { resolve: resolve as (value: unknown) => void, reject, timeout });

            const request: QueryRequest = {
                type: MessageType.QUERY_REQUEST,
                timestamp: Date.now(),
                requestId,
                query: options.query,
                parameters: options.parameters || [],
                errorMessage: null
            };

            debugLog(this.config, `Sending query: ${options.query.substring(0, 100)}...`);
            this.ws.send(JSON.stringify(request));
        });
    }

    getConnectionInfo(): ClientCoreConnectionInfo {
        const now = Date.now();
        return {
            status: this.lastStatus,
            isConnected: this.lastStatus === "connected",
            isConnecting: this.lastStatus === "connecting",
            isReconnecting: this.lastStatus === "reconnecting",
            connectionDuration: this.connectionStartTime ? now - this.connectionStartTime : undefined,
            reconnectAttempts: this.reconnectCount,
            pendingRequests: Array.from(this.pendingRequests.entries()).map(([requestId, pending]) => ({
                requestId,
                elapsedMs: now - (pending as any).startTime || 0
            })),
            agentId: this.agentId,
            sessionId: this.sessionId
        };
    }
}

export class ClientCore {
    private connectionManager: CoreConnectionManager;
    private _utilities: {
        Connection: {
            connect: (options?: { timeoutMs?: number }) => Promise<ClientCoreConnectionInfo>;
            disconnect: () => void;
            query: <T = unknown>(options: QueryOptions) => Promise<QueryResult<T>>;
            getConnectionInfo: () => ClientCoreConnectionInfo;
            addEventListener: (event: string, listener: ClientCoreConnectionEventListener) => void;
            removeEventListener: (event: string, listener: ClientCoreConnectionEventListener) => void;
        };
    } | null = null;

    constructor(config: ClientCoreConfig) {
        this.connectionManager = new CoreConnectionManager(config);
    }

    get Utilities() {
        if (!this._utilities) {
            this._utilities = {
                Connection: {
                    connect: (options?: { timeoutMs?: number }) =>
                        this.connectionManager.connect(options),

                    disconnect: () =>
                        this.connectionManager.disconnect(),

                    query: <T = unknown>(options: QueryOptions) =>
                        this.connectionManager.query<T>(options),

                    getConnectionInfo: () =>
                        this.connectionManager.getConnectionInfo(),

                    addEventListener: (event: string, listener: ClientCoreConnectionEventListener) =>
                        this.connectionManager.addEventListener(event, listener),

                    removeEventListener: (event: string, listener: ClientCoreConnectionEventListener) =>
                        this.connectionManager.removeEventListener(event, listener)
                }
            };
        }
        return this._utilities;
    }

    dispose(): void {
        this.connectionManager.disconnect();
    }
}
