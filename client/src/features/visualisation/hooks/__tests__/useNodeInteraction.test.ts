import { renderHook, act } from '@testing-library/react';
import { useNodeInteraction } from '../useNodeInteraction';

// Mock dependencies
jest.mock('../../../utils/loggerConfig', () => ({
  createLogger: () => ({
    debug: jest.fn(),
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn()
  })
}));

jest.mock('../../../utils/clientDebugState', () => ({
  debugState: {
    isEnabled: () => false
  }
}));

jest.mock('lodash', () => ({
  throttle: (fn: Function, delay: number) => {
    const throttled = (...args: any[]) => fn(...args);
    throttled.cancel = jest.fn();
    return throttled;
  }
}));

describe('useNodeInteraction', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should initialize with default interaction state', () => {
    const { result } = renderHook(() => useNodeInteraction());

    expect(result.current.interactionState).toEqual({
      isInteracting: false,
      isDragging: false,
      isClicking: false,
      nodeId: null,
      instanceId: null,
      lastInteractionTime: 0
    });
  });

  it('should start interaction correctly', () => {
    const onInteractionStart = jest.fn();
    const { result } = renderHook(() =>
      useNodeInteraction({ onInteractionStart })
    );

    act(() => {
      result.current.startInteraction('test-node-1', 1);
    });

    expect(result.current.interactionState.isInteracting).toBe(true);
    expect(result.current.interactionState.isClicking).toBe(true);
    expect(result.current.interactionState.nodeId).toBe('test-node-1');
    expect(result.current.interactionState.instanceId).toBe(1);
    expect(onInteractionStart).toHaveBeenCalledWith('test-node-1', 1);
  });

  it('should end interaction correctly', () => {
    const onInteractionEnd = jest.fn();
    const { result } = renderHook(() =>
      useNodeInteraction({ onInteractionEnd })
    );

    // Start interaction first
    act(() => {
      result.current.startInteraction('test-node-1', 1);
    });

    // End interaction
    act(() => {
      result.current.endInteraction();
    });

    expect(result.current.interactionState.isInteracting).toBe(false);
    expect(result.current.interactionState.nodeId).toBe(null);
    expect(result.current.interactionState.instanceId).toBe(null);
  });

  it('should update position during interaction', () => {
    const onPositionUpdate = jest.fn();
    const { result } = renderHook(() =>
      useNodeInteraction({ onPositionUpdate })
    );

    // Start interaction first
    act(() => {
      result.current.startInteraction('test-node-1', 1);
    });

    // Update position
    const position = { x: 10, y: 20, z: 30 };
    act(() => {
      result.current.updatePosition(position);
    });

    expect(onPositionUpdate).toHaveBeenCalledWith('test-node-1', position);
  });

  it('should report active interaction status correctly', () => {
    const { result } = renderHook(() => useNodeInteraction());

    expect(result.current.isActivelyInteracting()).toBe(false);

    act(() => {
      result.current.startInteraction('test-node-1', 1);
    });

    expect(result.current.isActivelyInteracting()).toBe(true);

    act(() => {
      result.current.endInteraction();
    });

    expect(result.current.isActivelyInteracting()).toBe(false);
  });
});