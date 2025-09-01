/**
 * Comprehensive tests for partial settings state management in frontend
 * 
 * Tests the new partial state handling, selective subscriptions,
 * and granular API integration
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { create } from 'zustand'
import { Settings, SettingsUpdate } from '../types/generated/settings'
import { settingsApi } from '../api/settingsApi'
import { useSelectiveSettingsStore } from '../hooks/useSelectiveSettingsStore'

// Mock the settings API
vi.mock('../api/settingsApi', () => ({
  settingsApi: {
    fetchSettings: vi.fn(),
    updateSettings: vi.fn(),
    getSettingsByPaths: vi.fn(),
    updateSettingsByPath: vi.fn(),
  }
}))

// Mock toast
vi.mock('../features/design-system/components/Toast', () => ({
  toast: {
    error: vi.fn(),
    success: vi.fn(),
  }
}))

describe('Partial Settings State Management', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Selective Settings Store', () => {
    it('should load only requested settings paths', async () => {
      const mockPartialSettings = {
        visualisation: {
          glow: {
            intensity: 1.5,
            nodeGlowStrength: 2.0,
            baseColor: '#ff0000'
          }
        }
      }

      vi.mocked(settingsApi.getSettingsByPaths).mockResolvedValue(mockPartialSettings)

      const { result } = renderHook(() => useSelectiveSettingsStore())

      await act(async () => {
        await result.current.loadSettingsPaths([
          'visualisation.glow.intensity',
          'visualisation.glow.nodeGlowStrength',
          'visualisation.glow.baseColor'
        ])
      })

      expect(settingsApi.getSettingsByPaths).toHaveBeenCalledWith([
        'visualisation.glow.intensity',
        'visualisation.glow.nodeGlowStrength',
        'visualisation.glow.baseColor'
      ])

      expect(result.current.getSettingValue('visualisation.glow.intensity')).toBe(1.5)
      expect(result.current.getSettingValue('visualisation.glow.nodeGlowStrength')).toBe(2.0)
      expect(result.current.getSettingValue('visualisation.glow.baseColor')).toBe('#ff0000')
    })

    it('should handle nested path access correctly', async () => {
      const mockNestedSettings = {
        visualisation: {
          graphs: {
            logseq: {
              physics: {
                springK: 0.02,
                repelK: 150.0,
                autoBalanceConfig: {
                  stabilityVarianceThreshold: 120.0,
                  clusteringDistanceThreshold: 30.0
                }
              }
            }
          }
        },
        system: {
          network: {
            port: 8080,
            bindAddress: '0.0.0.0'
          }
        }
      }

      vi.mocked(settingsApi.getSettingsByPaths).mockResolvedValue(mockNestedSettings)

      const { result } = renderHook(() => useSelectiveSettingsStore())

      await act(async () => {
        await result.current.loadSettingsPaths([
          'visualisation.graphs.logseq.physics.springK',
          'visualisation.graphs.logseq.physics.autoBalanceConfig.stabilityVarianceThreshold',
          'system.network.port'
        ])
      })

      expect(result.current.getSettingValue('visualisation.graphs.logseq.physics.springK')).toBe(0.02)
      expect(result.current.getSettingValue('visualisation.graphs.logseq.physics.autoBalanceConfig.stabilityVarianceThreshold')).toBe(120.0)
      expect(result.current.getSettingValue('system.network.port')).toBe(8080)
    })

    it('should update specific paths without affecting others', async () => {
      // Setup initial state
      const mockInitialSettings = {
        visualisation: {
          glow: {
            intensity: 1.0,
            nodeGlowStrength: 1.0,
            baseColor: '#ffffff'
          }
        }
      }

      vi.mocked(settingsApi.getSettingsByPaths).mockResolvedValue(mockInitialSettings)
      vi.mocked(settingsApi.updateSettingsByPath).mockResolvedValue({
        updatedPaths: ['visualisation.glow.intensity'],
        newValues: { 'visualisation.glow.intensity': 2.5 }
      })

      const { result } = renderHook(() => useSelectiveSettingsStore())

      // Load initial settings
      await act(async () => {
        await result.current.loadSettingsPaths([
          'visualisation.glow.intensity',
          'visualisation.glow.nodeGlowStrength',
          'visualisation.glow.baseColor'
        ])
      })

      // Update only intensity
      await act(async () => {
        await result.current.updateSettingPath('visualisation.glow.intensity', 2.5)
      })

      expect(settingsApi.updateSettingsByPath).toHaveBeenCalledWith([
        { path: 'visualisation.glow.intensity', value: 2.5 }
      ])

      // Intensity should be updated
      expect(result.current.getSettingValue('visualisation.glow.intensity')).toBe(2.5)
      
      // Other values should remain unchanged
      expect(result.current.getSettingValue('visualisation.glow.nodeGlowStrength')).toBe(1.0)
      expect(result.current.getSettingValue('visualisation.glow.baseColor')).toBe('#ffffff')
    })

    it('should handle batch updates efficiently', async () => {
      const mockInitialSettings = {
        visualisation: {
          glow: {
            intensity: 1.0,
            nodeGlowStrength: 1.0,
            edgeGlowStrength: 1.0
          }
        }
      }

      const mockBatchUpdate = {
        updatedPaths: [
          'visualisation.glow.intensity',
          'visualisation.glow.nodeGlowStrength',
          'visualisation.glow.edgeGlowStrength'
        ],
        newValues: {
          'visualisation.glow.intensity': 2.0,
          'visualisation.glow.nodeGlowStrength': 3.0,
          'visualisation.glow.edgeGlowStrength': 2.5
        }
      }

      vi.mocked(settingsApi.getSettingsByPaths).mockResolvedValue(mockInitialSettings)
      vi.mocked(settingsApi.updateSettingsByPath).mockResolvedValue(mockBatchUpdate)

      const { result } = renderHook(() => useSelectiveSettingsStore())

      await act(async () => {
        await result.current.loadSettingsPaths([
          'visualisation.glow.intensity',
          'visualisation.glow.nodeGlowStrength',
          'visualisation.glow.edgeGlowStrength'
        ])
      })

      // Batch update
      await act(async () => {
        await result.current.batchUpdateSettings([
          { path: 'visualisation.glow.intensity', value: 2.0 },
          { path: 'visualisation.glow.nodeGlowStrength', value: 3.0 },
          { path: 'visualisation.glow.edgeGlowStrength', value: 2.5 }
        ])
      })

      expect(settingsApi.updateSettingsByPath).toHaveBeenCalledWith([
        { path: 'visualisation.glow.intensity', value: 2.0 },
        { path: 'visualisation.glow.nodeGlowStrength', value: 3.0 },
        { path: 'visualisation.glow.edgeGlowStrength', value: 2.5 }
      ])

      // All values should be updated
      expect(result.current.getSettingValue('visualisation.glow.intensity')).toBe(2.0)
      expect(result.current.getSettingValue('visualisation.glow.nodeGlowStrength')).toBe(3.0)
      expect(result.current.getSettingValue('visualisation.glow.edgeGlowStrength')).toBe(2.5)
    })

    it('should handle subscription-based updates', async () => {
      const mockSettings = {
        visualisation: {
          glow: { intensity: 1.0 }
        }
      }

      vi.mocked(settingsApi.getSettingsByPaths).mockResolvedValue(mockSettings)

      const { result } = renderHook(() => useSelectiveSettingsStore())

      let subscriptionCallback: ((path: string, value: any) => void) | null = null

      // Setup subscription
      act(() => {
        subscriptionCallback = result.current.subscribeToPath('visualisation.glow.intensity', (value) => {
          // This would normally update a component
        })
      })

      // Load settings
      await act(async () => {
        await result.current.loadSettingsPaths(['visualisation.glow.intensity'])
      })

      expect(result.current.isSubscribed('visualisation.glow.intensity')).toBe(true)
      expect(result.current.getSubscriptionCount()).toBe(1)

      // Simulate external update
      act(() => {
        result.current.handleExternalUpdate('visualisation.glow.intensity', 2.5)
      })

      expect(result.current.getSettingValue('visualisation.glow.intensity')).toBe(2.5)

      // Unsubscribe
      act(() => {
        if (subscriptionCallback) {
          result.current.unsubscribeFromPath('visualisation.glow.intensity')
        }
      })

      expect(result.current.isSubscribed('visualisation.glow.intensity')).toBe(false)
      expect(result.current.getSubscriptionCount()).toBe(0)
    })

    it('should implement lazy loading for unaccessed paths', async () => {
      const { result } = renderHook(() => useSelectiveSettingsStore())

      // Initially, no paths should be loaded
      expect(result.current.isPathLoaded('visualisation.glow.intensity')).toBe(false)

      const mockSettings = {
        visualisation: {
          glow: { intensity: 1.0 }
        }
      }

      vi.mocked(settingsApi.getSettingsByPaths).mockResolvedValue(mockSettings)

      // Accessing a path should trigger lazy loading
      await act(async () => {
        const value = await result.current.getSettingValueLazy('visualisation.glow.intensity')
        expect(value).toBe(1.0)
      })

      expect(settingsApi.getSettingsByPaths).toHaveBeenCalledWith(['visualisation.glow.intensity'])
      expect(result.current.isPathLoaded('visualisation.glow.intensity')).toBe(true)
    })

    it('should handle error states gracefully', async () => {
      const { result } = renderHook(() => useSelectiveSettingsStore())

      vi.mocked(settingsApi.getSettingsByPaths).mockRejectedValue(new Error('Network error'))

      await act(async () => {
        try {
          await result.current.loadSettingsPaths(['visualisation.glow.intensity'])
        } catch (error) {
          // Expected to throw
        }
      })

      expect(result.current.hasError()).toBe(true)
      expect(result.current.getLastError()).toContain('Network error')
      expect(result.current.isPathLoaded('visualisation.glow.intensity')).toBe(false)
    })

    it('should debounce rapid updates', async () => {
      vi.useFakeTimers()

      const { result } = renderHook(() => useSelectiveSettingsStore())

      const mockSettings = {
        visualisation: { glow: { intensity: 1.0 } }
      }

      vi.mocked(settingsApi.getSettingsByPaths).mockResolvedValue(mockSettings)
      vi.mocked(settingsApi.updateSettingsByPath).mockResolvedValue({
        updatedPaths: ['visualisation.glow.intensity'],
        newValues: { 'visualisation.glow.intensity': 3.0 }
      })

      await act(async () => {
        await result.current.loadSettingsPaths(['visualisation.glow.intensity'])
      })

      // Make rapid updates
      act(() => {
        result.current.updateSettingPathDebounced('visualisation.glow.intensity', 1.5)
        result.current.updateSettingPathDebounced('visualisation.glow.intensity', 2.0)
        result.current.updateSettingPathDebounced('visualisation.glow.intensity', 3.0)
      })

      // Should not call API yet
      expect(settingsApi.updateSettingsByPath).not.toHaveBeenCalled()

      // Advance timers
      act(() => {
        vi.advanceTimersByTime(1000)
      })

      // Should only make one API call with the final value
      expect(settingsApi.updateSettingsByPath).toHaveBeenCalledTimes(1)
      expect(settingsApi.updateSettingsByPath).toHaveBeenCalledWith([
        { path: 'visualisation.glow.intensity', value: 3.0 }
      ])

      vi.useRealTimers()
    })

    it('should handle memory management for unused paths', async () => {
      const { result } = renderHook(() => useSelectiveSettingsStore())

      const mockSettings = {
        visualisation: {
          glow: { intensity: 1.0, nodeGlowStrength: 2.0 }
        }
      }

      vi.mocked(settingsApi.getSettingsByPaths).mockResolvedValue(mockSettings)

      // Load some paths
      await act(async () => {
        await result.current.loadSettingsPaths([
          'visualisation.glow.intensity',
          'visualisation.glow.nodeGlowStrength'
        ])
      })

      expect(result.current.getLoadedPathsCount()).toBe(2)

      // Cleanup unused paths
      act(() => {
        result.current.cleanupUnusedPaths(['visualisation.glow.intensity'])
      })

      expect(result.current.getLoadedPathsCount()).toBe(1)
      expect(result.current.isPathLoaded('visualisation.glow.intensity')).toBe(true)
      expect(result.current.isPathLoaded('visualisation.glow.nodeGlowStrength')).toBe(false)
    })
  })

  describe('Integration with Components', () => {
    it('should support component-level selective subscriptions', async () => {
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

      const { result } = renderHook(() => useSelectiveSettingsStore())

      // Simulate component mounting and subscribing to specific paths
      const componentSubscriptions = [
        'visualisation.glow.intensity',
        'visualisation.glow.baseColor'
      ]

      await act(async () => {
        await result.current.subscribeComponent('GlowPanel', componentSubscriptions)
      })

      expect(settingsApi.getSettingsByPaths).toHaveBeenCalledWith(componentSubscriptions)

      // Component should only get updates for subscribed paths
      const glowIntensityUpdates: any[] = []
      const nodeGlowStrengthUpdates: any[] = []

      result.current.onPathUpdate('visualisation.glow.intensity', (value) => {
        glowIntensityUpdates.push(value)
      })

      result.current.onPathUpdate('visualisation.glow.nodeGlowStrength', (value) => {
        nodeGlowStrengthUpdates.push(value)
      })

      // Update subscribed path
      act(() => {
        result.current.handleExternalUpdate('visualisation.glow.intensity', 1.5)
      })

      // Update unsubscribed path
      act(() => {
        result.current.handleExternalUpdate('visualisation.glow.nodeGlowStrength', 3.0)
      })

      expect(glowIntensityUpdates).toHaveLength(1)
      expect(glowIntensityUpdates[0]).toBe(1.5)

      // nodeGlowStrength shouldn't trigger updates since component didn't subscribe
      expect(nodeGlowStrengthUpdates).toHaveLength(0)

      // Cleanup component subscriptions
      act(() => {
        result.current.unsubscribeComponent('GlowPanel')
      })

      expect(result.current.getComponentSubscriptions('GlowPanel')).toHaveLength(0)
    })

    it('should handle multiple components with overlapping subscriptions', async () => {
      const { result } = renderHook(() => useSelectiveSettingsStore())

      const mockSettings = {
        visualisation: { glow: { intensity: 1.0, nodeGlowStrength: 2.0 } }
      }

      vi.mocked(settingsApi.getSettingsByPaths).mockResolvedValue(mockSettings)

      // Component A subscribes to intensity and nodeGlowStrength
      await act(async () => {
        await result.current.subscribeComponent('ComponentA', [
          'visualisation.glow.intensity',
          'visualisation.glow.nodeGlowStrength'
        ])
      })

      // Component B subscribes to intensity only
      await act(async () => {
        await result.current.subscribeComponent('ComponentB', [
          'visualisation.glow.intensity'
        ])
      })

      expect(result.current.getActiveSubscriptionCount()).toBe(2) // unique paths
      expect(result.current.getComponentCount()).toBe(2)

      // Unsubscribe Component B
      act(() => {
        result.current.unsubscribeComponent('ComponentB')
      })

      // intensity should still be subscribed (Component A)
      expect(result.current.isSubscribed('visualisation.glow.intensity')).toBe(true)
      expect(result.current.getComponentCount()).toBe(1)

      // Unsubscribe Component A
      act(() => {
        result.current.unsubscribeComponent('ComponentA')
      })

      expect(result.current.getActiveSubscriptionCount()).toBe(0)
      expect(result.current.getComponentCount()).toBe(0)
    })
  })

  describe('Performance Characteristics', () => {
    it('should minimize API calls through intelligent caching', async () => {
      const { result } = renderHook(() => useSelectiveSettingsStore())

      const mockSettings = {
        visualisation: { glow: { intensity: 1.0 } }
      }

      vi.mocked(settingsApi.getSettingsByPaths).mockResolvedValue(mockSettings)

      // First access should trigger API call
      await act(async () => {
        await result.current.getSettingValueLazy('visualisation.glow.intensity')
      })

      expect(settingsApi.getSettingsByPaths).toHaveBeenCalledTimes(1)

      // Second access should use cache
      await act(async () => {
        await result.current.getSettingValueLazy('visualisation.glow.intensity')
      })

      expect(settingsApi.getSettingsByPaths).toHaveBeenCalledTimes(1) // Still 1
    })

    it('should handle memory pressure by evicting LRU paths', async () => {
      const { result } = renderHook(() => useSelectiveSettingsStore())

      // Configure small cache size for testing
      act(() => {
        result.current.setMaxCacheSize(2)
      })

      const mockSettings1 = { visualisation: { glow: { intensity: 1.0 } } }
      const mockSettings2 = { visualisation: { glow: { nodeGlowStrength: 2.0 } } }
      const mockSettings3 = { visualisation: { glow: { baseColor: '#ffffff' } } }

      vi.mocked(settingsApi.getSettingsByPaths)
        .mockResolvedValueOnce(mockSettings1)
        .mockResolvedValueOnce(mockSettings2)
        .mockResolvedValueOnce(mockSettings3)
        .mockResolvedValueOnce(mockSettings1) // Should be called again

      // Load 3 paths (exceeds cache size of 2)
      await act(async () => {
        await result.current.getSettingValueLazy('visualisation.glow.intensity')
      })

      await act(async () => {
        await result.current.getSettingValueLazy('visualisation.glow.nodeGlowStrength')
      })

      await act(async () => {
        await result.current.getSettingValueLazy('visualisation.glow.baseColor')
      })

      // First path should have been evicted
      expect(result.current.isPathLoaded('visualisation.glow.intensity')).toBe(false)

      // Accessing it again should trigger new API call
      await act(async () => {
        await result.current.getSettingValueLazy('visualisation.glow.intensity')
      })

      expect(settingsApi.getSettingsByPaths).toHaveBeenCalledTimes(4)
    })
  })
})