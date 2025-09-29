import { useEffect, useRef, useState, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import { NeuralWebSocketMessage, NeuralDashboardState } from '../types/neural';

interface UseNeuralWebSocketOptions {
  autoConnect?: boolean;
  reconnectAttempts?: number;
  reconnectDelay?: number;
  onMessage?: (message: NeuralWebSocketMessage) => void;
  onError?: (error: Error) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
}

interface UseNeuralWebSocketReturn {
  socket: Socket | null;
  isConnected: boolean;
  isConnecting: boolean;
  error: Error | null;
  lastMessage: NeuralWebSocketMessage | null;
  connect: () => void;
  disconnect: () => void;
  sendMessage: (type: string, payload: any) => void;
  subscribe: (eventType: string, callback: (data: any) => void) => () => void;
}

export const useNeuralWebSocket = (
  url?: string,
  options: UseNeuralWebSocketOptions = {}
): UseNeuralWebSocketReturn => {
  const {
    autoConnect = true,
    reconnectAttempts = 5,
    reconnectDelay = 3000,
    onMessage,
    onError,
    onConnect,
    onDisconnect
  } = options;

  const socketRef = useRef<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [lastMessage, setLastMessage] = useState<NeuralWebSocketMessage | null>(null);
  const reconnectCountRef = useRef(0);
  const subscriptionsRef = useRef<Map<string, Set<(data: any) => void>>>(new Map());

  const wsUrl = url || `ws://${window.location.hostname}:8080/neural`;

  const handleConnect = useCallback(() => {
    setIsConnected(true);
    setIsConnecting(false);
    setError(null);
    reconnectCountRef.current = 0;
    console.log('Neural WebSocket connected');
    onConnect?.();
  }, [onConnect]);

  const handleDisconnect = useCallback((reason: string) => {
    setIsConnected(false);
    setIsConnecting(false);
    console.log('Neural WebSocket disconnected:', reason);
    onDisconnect?.();

    // Attempt reconnection if not manually disconnected
    if (reason !== 'io client disconnect' && reconnectCountRef.current < reconnectAttempts) {
      setTimeout(() => {
        reconnectCountRef.current++;
        console.log(`Attempting to reconnect... (${reconnectCountRef.current}/${reconnectAttempts})`);
        setIsConnecting(true);
        socketRef.current?.connect();
      }, reconnectDelay);
    }
  }, [onDisconnect, reconnectAttempts, reconnectDelay]);

  const handleError = useCallback((err: Error) => {
    setError(err);
    setIsConnecting(false);
    console.error('Neural WebSocket error:', err);
    onError?.(err);
  }, [onError]);

  const handleMessage = useCallback((message: NeuralWebSocketMessage) => {
    setLastMessage(message);
    onMessage?.(message);

    // Notify subscribers
    const callbacks = subscriptionsRef.current.get(message.type);
    if (callbacks) {
      callbacks.forEach(callback => callback(message.payload));
    }
  }, [onMessage]);

  const connect = useCallback(() => {
    if (socketRef.current?.connected) return;

    setIsConnecting(true);
    setError(null);

    try {
      const socket = io(wsUrl, {
        transports: ['websocket'],
        upgrade: false,
        rememberUpgrade: false,
        timeout: 10000,
        forceNew: true
      });

      socket.on('connect', handleConnect);
      socket.on('disconnect', handleDisconnect);
      socket.on('connect_error', handleError);

      // Handle neural-specific events
      socket.on('agent_update', (data) => handleMessage({
        type: 'agent_update',
        payload: data,
        timestamp: new Date(),
        source: 'server'
      }));

      socket.on('swarm_topology', (data) => handleMessage({
        type: 'swarm_topology',
        payload: data,
        timestamp: new Date(),
        source: 'server'
      }));

      socket.on('memory_sync', (data) => handleMessage({
        type: 'memory_sync',
        payload: data,
        timestamp: new Date(),
        source: 'server'
      }));

      socket.on('consensus_update', (data) => handleMessage({
        type: 'consensus_update',
        payload: data,
        timestamp: new Date(),
        source: 'server'
      }));

      socket.on('metrics_update', (data) => handleMessage({
        type: 'metrics_update',
        payload: data,
        timestamp: new Date(),
        source: 'server'
      }));

      socket.on('task_result', (data) => handleMessage({
        type: 'task_result',
        payload: data,
        timestamp: new Date(),
        source: 'server'
      }));

      socketRef.current = socket;
    } catch (err) {
      handleError(err as Error);
    }
  }, [wsUrl, handleConnect, handleDisconnect, handleError, handleMessage]);

  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }
    setIsConnected(false);
    setIsConnecting(false);
  }, []);

  const sendMessage = useCallback((type: string, payload: any) => {
    if (socketRef.current?.connected) {
      const message: NeuralWebSocketMessage = {
        type: type as any,
        payload,
        timestamp: new Date(),
        source: 'client'
      };
      socketRef.current.emit(type, payload);
    } else {
      console.warn('Cannot send message: WebSocket not connected');
    }
  }, []);

  const subscribe = useCallback((eventType: string, callback: (data: any) => void) => {
    if (!subscriptionsRef.current.has(eventType)) {
      subscriptionsRef.current.set(eventType, new Set());
    }
    subscriptionsRef.current.get(eventType)!.add(callback);

    // Return unsubscribe function
    return () => {
      const callbacks = subscriptionsRef.current.get(eventType);
      if (callbacks) {
        callbacks.delete(callback);
        if (callbacks.size === 0) {
          subscriptionsRef.current.delete(eventType);
        }
      }
    };
  }, []);

  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  return {
    socket: socketRef.current,
    isConnected,
    isConnecting,
    error,
    lastMessage,
    connect,
    disconnect,
    sendMessage,
    subscribe
  };
};

export default useNeuralWebSocket;