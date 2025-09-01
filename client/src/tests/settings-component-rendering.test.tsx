/**
 * Comprehensive tests for React component rendering with selective settings subscriptions
 * 
 * Tests component-level optimization, selective re-rendering, and subscription patterns
 */

import React from 'react'
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { renderHook, act } from '@testing-library/react'
import { useSelectiveSettingsStore } from '../hooks/useSelectiveSettingsStore'
import { settingsApi } from '../api/settingsApi'

// Mock the settings API
vi.mock('../api/settingsApi', () => ({
  settingsApi: {
    getSettingsByPaths: vi.fn(),
    updateSettingsByPath: vi.fn(),
  }
}))

// Mock components to test selective rendering
const GlowIntensityComponent: React.FC = () => {
  const { getSettingValue, subscribeComponent, unsubscribeComponent } = useSelectiveSettingsStore()
  const [intensity, setIntensity] = React.useState<number>(1.0)
  const [renderCount, setRenderCount] = React.useState(0)

  React.useEffect(() => {
    setRenderCount(count => count + 1)
  })

  React.useEffect(() => {
    subscribeComponent('GlowIntensityComponent', ['visualisation.glow.intensity'])
    
    const unsubscribe = useSelectiveSettingsStore.getState().onPathUpdate(
      'visualisation.glow.intensity',
      (value: number) => setIntensity(value)
    )

    return () => {
      unsubscribe()
      unsubscribeComponent('GlowIntensityComponent')
    }
  }, [subscribeComponent, unsubscribeComponent])

  React.useEffect(() => {
    const currentValue = getSettingValue('visualisation.glow.intensity')
    if (currentValue !== undefined) {
      setIntensity(currentValue)
    }
  }, [getSettingValue])

  return (
    <div>
      <span data-testid="intensity-value">{intensity}</span>
      <span data-testid="render-count">{renderCount}</span>
    </div>
  )
}

const NodeGlowComponent: React.FC = () => {
  const { getSettingValue, subscribeComponent, unsubscribeComponent } = useSelectiveSettingsStore()
  const [nodeGlowStrength, setNodeGlowStrength] = React.useState<number>(1.0)
  const [renderCount, setRenderCount] = React.useState(0)

  React.useEffect(() => {
    setRenderCount(count => count + 1)
  })

  React.useEffect(() => {
    subscribeComponent('NodeGlowComponent', ['visualisation.glow.nodeGlowStrength'])
    
    const unsubscribe = useSelectiveSettingsStore.getState().onPathUpdate(
      'visualisation.glow.nodeGlowStrength',
      (value: number) => setNodeGlowStrength(value)
    )

    return () => {
      unsubscribe()
      unsubscribeComponent('NodeGlowComponent')
    }
  }, [subscribeComponent, unsubscribeComponent])

  React.useEffect(() => {
    const currentValue = getSettingValue('visualisation.glow.nodeGlowStrength')
    if (currentValue !== undefined) {
      setNodeGlowStrength(currentValue)
    }
  }, [getSettingValue])

  return (
    <div>
      <span data-testid="node-glow-value">{nodeGlowStrength}</span>
      <span data-testid="node-render-count">{renderCount}</span>
    </div>
  )
}

const MultiSubscriptionComponent: React.FC = () => {
  const { getSettingValue, subscribeComponent, unsubscribeComponent } = useSelectiveSettingsStore()
  const [settings, setSettings] = React.useState({
    intensity: 1.0,
    nodeGlowStrength: 1.0,
    baseColor: '#ffffff'
  })
  const [renderCount, setRenderCount] = React.useState(0)

  React.useEffect(() => {
    setRenderCount(count => count + 1)
  })

  React.useEffect(() => {
    const subscriptions = [
      'visualisation.glow.intensity',
      'visualisation.glow.nodeGlowStrength',
      'visualisation.glow.baseColor'
    ]

    subscribeComponent('MultiSubscriptionComponent', subscriptions)
    
    const unsubscribers = subscriptions.map(path => 
      useSelectiveSettingsStore.getState().onPathUpdate(path, (value: any) => {
        setSettings(prev => ({
          ...prev,
          [path.split('.').pop()!]: value
        }))
      })
    )

    return () => {
      unsubscribers.forEach(unsub => unsub())
      unsubscribeComponent('MultiSubscriptionComponent')
    }
  }, [subscribeComponent, unsubscribeComponent])

  return (
    <div>
      <span data-testid="multi-intensity">{settings.intensity}</span>
      <span data-testid="multi-node-glow">{settings.nodeGlowStrength}</span>
      <span data-testid="multi-base-color">{settings.baseColor}</span>
      <span data-testid="multi-render-count">{renderCount}</span>
    </div>
  )
}

const LazyLoadingComponent: React.FC<{ path: string }> = ({ path }) => {
  const { getSettingValueLazy } = useSelectiveSettingsStore()
  const [value, setValue] = React.useState<any>(null)
  const [loading, setLoading] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  const loadValue = React.useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const settingValue = await getSettingValueLazy(path)
      setValue(settingValue)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }, [path, getSettingValueLazy])

  return (
    <div>
      <button onClick={loadValue} data-testid="load-button">
        Load Setting
      </button>
      {loading && <span data-testid="loading">Loading...</span>}
      {error && <span data-testid="error">{error}</span>}
      {value !== null && <span data-testid="lazy-value">{String(value)}</span>}
    </div>
  )
}

describe('Settings Component Rendering Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // Reset the store state
    useSelectiveSettingsStore.getState().reset?.()
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  describe('Selective Subscription Rendering', () => {
    it('should only re-render components when their subscribed settings change', async () => {
      const mockSettings = {
        visualisation: {
          glow: {
            intensity: 1.5,
            nodeGlowStrength: 2.0
          }
        }
      }

      vi.mocked(settingsApi.getSettingsByPaths).mockResolvedValue(mockSettings)

      render(
        <div>
          <GlowIntensityComponent />
          <NodeGlowComponent />
        </div>
      )

      // Wait for components to initialize
      await waitFor(() => {
        expect(screen.getByTestId('intensity-value')).toHaveTextContent('1.5')
        expect(screen.getByTestId('node-glow-value')).toHaveTextContent('2.0')
      })

      // Get initial render counts
      const initialIntensityRenders = parseInt(screen.getByTestId('render-count').textContent || '0')
      const initialNodeRenders = parseInt(screen.getByTestId('node-render-count').textContent || '0')

      // Update only intensity setting
      act(() => {
        useSelectiveSettingsStore.getState().handleExternalUpdate('visualisation.glow.intensity', 2.5)
      })

      await waitFor(() => {
        expect(screen.getByTestId('intensity-value')).toHaveTextContent('2.5')
      })

      // GlowIntensityComponent should have re-rendered
      const newIntensityRenders = parseInt(screen.getByTestId('render-count').textContent || '0')
      expect(newIntensityRenders).toBeGreaterThan(initialIntensityRenders)

      // NodeGlowComponent should NOT have re-rendered
      const newNodeRenders = parseInt(screen.getByTestId('node-render-count').textContent || '0')
      expect(newNodeRenders).toBe(initialNodeRenders)

      // Node glow value should remain unchanged
      expect(screen.getByTestId('node-glow-value')).toHaveTextContent('2.0')
    })

    it('should handle multiple subscriptions in a single component efficiently', async () => {
      const mockSettings = {
        visualisation: {
          glow: {
            intensity: 1.0,
            nodeGlowStrength: 2.0,
            baseColor: '#ffffff'
          }
        }
      }

      vi.mocked(settingsApi.getSettingsByPaths).mockResolvedValue(mockSettings)

      render(<MultiSubscriptionComponent />)

      await waitFor(() => {
        expect(screen.getByTestId('multi-intensity')).toHaveTextContent('1.0')
        expect(screen.getByTestId('multi-node-glow')).toHaveTextContent('2.0')
        expect(screen.getByTestId('multi-base-color')).toHaveTextContent('#ffffff')
      })

      const initialRenderCount = parseInt(screen.getByTestId('multi-render-count').textContent || '0')

      // Update one setting
      act(() => {
        useSelectiveSettingsStore.getState().handleExternalUpdate('visualisation.glow.intensity', 1.5)
      })

      await waitFor(() => {
        expect(screen.getByTestId('multi-intensity')).toHaveTextContent('1.5')
      })

      // Should re-render once for the intensity change
      const newRenderCount = parseInt(screen.getByTestId('multi-render-count').textContent || '0')
      expect(newRenderCount).toBe(initialRenderCount + 1)

      // Other values should remain unchanged
      expect(screen.getByTestId('multi-node-glow')).toHaveTextContent('2.0')
      expect(screen.getByTestId('multi-base-color')).toHaveTextContent('#ffffff')
    })

    it('should handle component unmounting and subscription cleanup', async () => {
      const mockSettings = {
        visualisation: { glow: { intensity: 1.0 } }
      }

      vi.mocked(settingsApi.getSettingsByPaths).mockResolvedValue(mockSettings)

      const { unmount } = render(<GlowIntensityComponent />)

      await waitFor(() => {
        expect(screen.getByTestId('intensity-value')).toHaveTextContent('1.0')
      })

      // Component should be subscribed
      expect(useSelectiveSettingsStore.getState().getComponentCount()).toBe(1)
      expect(useSelectiveSettingsStore.getState().isSubscribed('visualisation.glow.intensity')).toBe(true)

      // Unmount component
      unmount()

      // Subscriptions should be cleaned up
      expect(useSelectiveSettingsStore.getState().getComponentCount()).toBe(0)
      expect(useSelectiveSettingsStore.getState().isSubscribed('visualisation.glow.intensity')).toBe(false)
    })

    it('should support lazy loading in components', async () => {
      const mockSettings = {
        visualisation: { glow: { intensity: 2.5 } }
      }

      vi.mocked(settingsApi.getSettingsByPaths).mockResolvedValue(mockSettings)

      render(<LazyLoadingComponent path="visualisation.glow.intensity" />)

      // Initially should not have loaded
      expect(screen.queryByTestId('lazy-value')).not.toBeInTheDocument()
      expect(settingsApi.getSettingsByPaths).not.toHaveBeenCalled()

      // Click load button
      fireEvent.click(screen.getByTestId('load-button'))

      // Should show loading state
      expect(screen.getByTestId('loading')).toBeInTheDocument()

      // Wait for load to complete
      await waitFor(() => {
        expect(screen.getByTestId('lazy-value')).toHaveTextContent('2.5')
        expect(screen.queryByTestId('loading')).not.toBeInTheDocument()
      })

      expect(settingsApi.getSettingsByPaths).toHaveBeenCalledWith(['visualisation.glow.intensity'])
    })

    it('should handle lazy loading errors gracefully', async () => {
      vi.mocked(settingsApi.getSettingsByPaths).mockRejectedValue(new Error('Network failed'))

      render(<LazyLoadingComponent path="visualisation.glow.intensity" />)

      fireEvent.click(screen.getByTestId('load-button'))

      await waitFor(() => {
        expect(screen.getByTestId('error')).toHaveTextContent('Network failed')
        expect(screen.queryByTestId('loading')).not.toBeInTheDocument()
        expect(screen.queryByTestId('lazy-value')).not.toBeInTheDocument()
      })
    })
  })

  describe('Performance Optimizations', () => {
    it('should batch multiple rapid updates to prevent excessive re-renders', async () => {
      vi.useFakeTimers()

      const mockSettings = {
        visualisation: { glow: { intensity: 1.0 } }
      }

      vi.mocked(settingsApi.getSettingsByPaths).mockResolvedValue(mockSettings)

      render(<GlowIntensityComponent />)

      await waitFor(() => {
        expect(screen.getByTestId('intensity-value')).toHaveTextContent('1.0')
      })

      const initialRenderCount = parseInt(screen.getByTestId('render-count').textContent || '0')

      // Make rapid updates
      act(() => {
        useSelectiveSettingsStore.getState().handleExternalUpdate('visualisation.glow.intensity', 1.1)
        useSelectiveSettingsStore.getState().handleExternalUpdate('visualisation.glow.intensity', 1.2)
        useSelectiveSettingsStore.getState().handleExternalUpdate('visualisation.glow.intensity', 1.3)
        useSelectiveSettingsStore.getState().handleExternalUpdate('visualisation.glow.intensity', 1.4)
        useSelectiveSettingsStore.getState().handleExternalUpdate('visualisation.glow.intensity', 1.5)
      })

      // Should batch updates - only final value should be rendered
      await waitFor(() => {
        expect(screen.getByTestId('intensity-value')).toHaveTextContent('1.5')
      })

      const finalRenderCount = parseInt(screen.getByTestId('render-count').textContent || '0')

      // Should have minimal re-renders despite multiple updates
      expect(finalRenderCount - initialRenderCount).toBeLessThanOrEqual(2)

      vi.useRealTimers()
    })

    it('should memoize subscription callbacks to prevent unnecessary re-subscriptions', async () => {
      const mockSettings = {
        visualisation: { glow: { intensity: 1.0 } }
      }

      vi.mocked(settingsApi.getSettingsByPaths).mockResolvedValue(mockSettings)

      const TestComponent: React.FC = () => {
        const { subscribeComponent, unsubscribeComponent } = useSelectiveSettingsStore()
        const [count, setCount] = React.useState(0)

        // Memoize the subscription callback
        const handleUpdate = React.useCallback((value: number) => {
          // Component updates based on setting change
        }, [])

        React.useEffect(() => {
          subscribeComponent('TestComponent', ['visualisation.glow.intensity'])
          
          const unsubscribe = useSelectiveSettingsStore.getState().onPathUpdate(
            'visualisation.glow.intensity',
            handleUpdate
          )

          return () => {
            unsubscribe()
            unsubscribeComponent('TestComponent')
          }
        }, [subscribeComponent, unsubscribeComponent, handleUpdate])

        return (
          <div>
            <button onClick={() => setCount(count + 1)} data-testid="increment">
              Count: {count}
            </button>
          </div>
        )
      }

      render(<TestComponent />)

      await waitFor(() => {
        expect(useSelectiveSettingsStore.getState().getComponentCount()).toBe(1)
      })

      const initialSubscriptionCount = useSelectiveSettingsStore.getState().getActiveSubscriptionCount()

      // Force component re-render (but not subscription re-creation due to memoization)
      fireEvent.click(screen.getByTestId('increment'))
      fireEvent.click(screen.getByTestId('increment'))

      // Subscription count should remain stable
      expect(useSelectiveSettingsStore.getState().getActiveSubscriptionCount()).toBe(initialSubscriptionCount)
    })

    it('should efficiently handle overlapping subscriptions between components', async () => {
      const mockSettings = {
        visualisation: {
          glow: {
            intensity: 1.0,
            nodeGlowStrength: 2.0
          }
        }
      }

      vi.mocked(settingsApi.getSettingsByPaths).mockResolvedValue(mockSettings)

      const SharedSubscriptionComponentA: React.FC = () => {
        const { subscribeComponent, unsubscribeComponent } = useSelectiveSettingsStore()

        React.useEffect(() => {
          subscribeComponent('ComponentA', ['visualisation.glow.intensity'])
          return () => unsubscribeComponent('ComponentA')
        }, [subscribeComponent, unsubscribeComponent])

        return <div data-testid="component-a">Component A</div>
      }

      const SharedSubscriptionComponentB: React.FC = () => {
        const { subscribeComponent, unsubscribeComponent } = useSelectiveSettingsStore()

        React.useEffect(() => {
          subscribeComponent('ComponentB', ['visualisation.glow.intensity']) // Same path as A
          return () => unsubscribeComponent('ComponentB')
        }, [subscribeComponent, unsubscribeComponent])

        return <div data-testid="component-b">Component B</div>
      }

      render(
        <div>
          <SharedSubscriptionComponentA />
          <SharedSubscriptionComponentB />
        </div>
      )

      await waitFor(() => {
        expect(screen.getByTestId('component-a')).toBeInTheDocument()
        expect(screen.getByTestId('component-b')).toBeInTheDocument()
      })

      // Should have 2 components but only 1 unique subscription path
      expect(useSelectiveSettingsStore.getState().getComponentCount()).toBe(2)
      expect(useSelectiveSettingsStore.getState().getActiveSubscriptionCount()).toBe(1)

      // API should only be called once for the shared path
      expect(settingsApi.getSettingsByPaths).toHaveBeenCalledTimes(1)
    })
  })

  describe('Error Handling in Components', () => {
    it('should display error states when settings fail to load', async () => {
      vi.mocked(settingsApi.getSettingsByPaths).mockRejectedValue(new Error('Failed to load settings'))

      const ErrorHandlingComponent: React.FC = () => {
        const { subscribeComponent, unsubscribeComponent, hasError, getLastError } = useSelectiveSettingsStore()
        const [error, setError] = React.useState<string | null>(null)

        React.useEffect(() => {
          subscribeComponent('ErrorHandlingComponent', ['visualisation.glow.intensity'])
          
          if (hasError()) {
            setError(getLastError())
          }

          return () => unsubscribeComponent('ErrorHandlingComponent')
        }, [subscribeComponent, unsubscribeComponent, hasError, getLastError])

        if (error) {
          return <div data-testid="component-error">{error}</div>
        }

        return <div data-testid="component-normal">Normal state</div>
      }

      render(<ErrorHandlingComponent />)

      await waitFor(() => {
        expect(screen.getByTestId('component-error')).toHaveTextContent('Failed to load settings')
      })
    })

    it('should recover from errors when settings become available', async () => {
      // First call fails, second succeeds
      vi.mocked(settingsApi.getSettingsByPaths)
        .mockRejectedValueOnce(new Error('Network error'))
        .mockResolvedValueOnce({
          visualisation: { glow: { intensity: 1.0 } }
        })

      const RecoveryComponent: React.FC = () => {
        const { getSettingValueLazy } = useSelectiveSettingsStore()
        const [value, setValue] = React.useState<number | null>(null)
        const [error, setError] = React.useState<string | null>(null)
        const [retrying, setRetrying] = React.useState(false)

        const loadSetting = async () => {
          setRetrying(true)
          setError(null)
          try {
            const setting = await getSettingValueLazy('visualisation.glow.intensity')
            setValue(setting)
          } catch (err) {
            setError(err instanceof Error ? err.message : 'Unknown error')
          } finally {
            setRetrying(false)
          }
        }

        React.useEffect(() => {
          loadSetting()
        }, [])

        if (error && !retrying) {
          return (
            <div>
              <span data-testid="error-message">{error}</span>
              <button onClick={loadSetting} data-testid="retry-button">
                Retry
              </button>
            </div>
          )
        }

        if (retrying) {
          return <div data-testid="retrying">Retrying...</div>
        }

        return <div data-testid="success-value">{value}</div>
      }

      render(<RecoveryComponent />)

      // Should show error first
      await waitFor(() => {
        expect(screen.getByTestId('error-message')).toHaveTextContent('Network error')
      })

      // Click retry
      fireEvent.click(screen.getByTestId('retry-button'))

      // Should show retrying state
      expect(screen.getByTestId('retrying')).toBeInTheDocument()

      // Should eventually succeed
      await waitFor(() => {
        expect(screen.getByTestId('success-value')).toHaveTextContent('1')
      })
    })
  })
})