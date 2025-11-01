import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react';
import { ClientCore, ClientCoreConfig, ClientCoreConnectionInfo } from '../services/vircadia/VircadiaClientCore';
import { createLogger } from '../utils/loggerConfig';

const logger = createLogger('VircadiaContext');

interface VircadiaContextValue {
    client: ClientCore | null;
    connectionInfo: ClientCoreConnectionInfo | null;
    isConnected: boolean;
    isConnecting: boolean;
    error: Error | null;
    connect: () => Promise<void>;
    disconnect: () => void;
}

const VircadiaContext = createContext<VircadiaContextValue | null>(null);

interface VircadiaProviderProps {
    children: React.ReactNode;
    config?: Partial<ClientCoreConfig>;
    autoConnect?: boolean;
}

export const VircadiaProvider: React.FC<VircadiaProviderProps> = ({
    children,
    config,
    autoConnect = false
}) => {
    const [client, setClient] = useState<ClientCore | null>(null);
    const [connectionInfo, setConnectionInfo] = useState<ClientCoreConnectionInfo | null>(null);
    const [error, setError] = useState<Error | null>(null);
    const clientRef = useRef<ClientCore | null>(null);

    
    const defaultConfig: ClientCoreConfig = {
        serverUrl: import.meta.env.VITE_VIRCADIA_SERVER_URL || 'ws://localhost:3020/world/ws',
        authToken: import.meta.env.VITE_VIRCADIA_AUTH_TOKEN || '',
        authProvider: import.meta.env.VITE_VIRCADIA_AUTH_PROVIDER || 'system',
        reconnectAttempts: 5,
        reconnectDelay: 5000,
        debug: import.meta.env.DEV || false,
        suppress: false,
        ...config
    };

    
    useEffect(() => {
        logger.info('Initializing Vircadia client with config:', {
            serverUrl: defaultConfig.serverUrl,
            authProvider: defaultConfig.authProvider,
            debug: defaultConfig.debug
        });

        const vircadiaClient = new ClientCore(defaultConfig);
        setClient(vircadiaClient);
        clientRef.current = vircadiaClient;

        
        const handleStatusChange = () => {
            const info = vircadiaClient.Utilities.Connection.getConnectionInfo();
            setConnectionInfo(info);
            logger.info('Connection status changed:', info.status);
        };

        vircadiaClient.Utilities.Connection.addEventListener('statusChange', handleStatusChange);

        
        return () => {
            vircadiaClient.Utilities.Connection.removeEventListener('statusChange', handleStatusChange);
            vircadiaClient.dispose();
            clientRef.current = null;
        };
    }, []);

    
    useEffect(() => {
        if (autoConnect && client && !connectionInfo?.isConnected) {
            connect();
        }
    }, [autoConnect, client]);

    const connect = useCallback(async () => {
        if (!client) {
            const err = new Error('Vircadia client not initialized');
            setError(err);
            logger.error('Cannot connect: client not initialized');
            return;
        }

        try {
            setError(null);
            logger.info('Connecting to Vircadia server...');

            const info = await client.Utilities.Connection.connect({ timeoutMs: 30000 });

            setConnectionInfo(info);
            logger.info('Connected to Vircadia server', {
                agentId: info.agentId,
                sessionId: info.sessionId,
                duration: info.connectionDuration
            });
        } catch (err) {
            const error = err instanceof Error ? err : new Error(String(err));
            setError(error);
            logger.error('Failed to connect to Vircadia server:', error);
        }
    }, [client]);

    const disconnect = useCallback(() => {
        if (client) {
            logger.info('Disconnecting from Vircadia server');
            client.Utilities.Connection.disconnect();
            setConnectionInfo(null);
        }
    }, [client]);

    const value: VircadiaContextValue = {
        client,
        connectionInfo,
        isConnected: connectionInfo?.isConnected || false,
        isConnecting: connectionInfo?.isConnecting || false,
        error,
        connect,
        disconnect
    };

    return (
        <VircadiaContext.Provider value={value}>
            {children}
        </VircadiaContext.Provider>
    );
};

export const useVircadia = () => {
    const context = useContext(VircadiaContext);
    if (!context) {
        throw new Error('useVircadia must be used within a VircadiaProvider');
    }
    return context;
};

// Hook for Quest 3 XR integration
export const useVircadiaXR = () => {
    const { client, isConnected, connectionInfo } = useVircadia();
    const [xrReady, setXRReady] = useState(false);

    useEffect(() => {
        if (isConnected && client && connectionInfo?.agentId && connectionInfo?.sessionId) {
            logger.info('Vircadia XR ready', {
                agentId: connectionInfo.agentId,
                sessionId: connectionInfo.sessionId
            });
            setXRReady(true);
        } else {
            setXRReady(false);
        }
    }, [isConnected, client, connectionInfo]);

    return {
        client,
        isConnected,
        xrReady,
        agentId: connectionInfo?.agentId || null,
        sessionId: connectionInfo?.sessionId || null
    };
};
