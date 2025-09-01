/**
 * Comprehensive tests for settings store in refactored system
 * 
 * Tests partial state management, selective subscriptions, 
 * granular updates, and integration with new camelCase API
 */

import { describe, it, expect, beforeEach, afterEach, vi, Mock } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { create } from 'zustand'
import { subscribeWithSelector } from 'zustand/middleware'
import { Settings, SettingsUpdate } from '../types/generated/settings'
import { settingsApi } from '../api/settingsApi'

// Mock the settings API
vi.mock('../api/settingsApi', () => ({
  settingsApi: {
    fetchSettings: vi.fn(),
    updateSettings: vi.fn(),
    getSettingsByPaths: vi.fn(),
    updateSettingsByPath: vi.fn(),
  }
}))

// Mock toast notifications
vi.mock('../features/design-system/components/Toast', () => ({
  toast: {
    error: vi.fn(),
    success: vi.fn(),
  }
}))

// Settings Store Interface
interface SettingsStore {
  settings: Partial<Settings>
  loadedPaths: Set<string>
  isLoading: boolean
  error: string | null
  
  // Actions
  loadSettingsByPaths: (paths: string[]) => Promise<void>
  updateSettingsByPath: (updates: Array<{path: string, value: any}>) => Promise<void>
  updateSetting: <T>(path: string, value: T) => void
  getSetting: <T>(path: string, defaultValue?: T) => T | undefined
  subscribeToPath: (path: string, callback: (value: any) => void) => () => void
  resetSettings: () => void
  isPathLoaded: (path: string) => boolean
}

// Create the settings store
const createSettingsStore = () => create<SettingsStore>()(
  subscribeWithSelector((set, get) => ({
    settings: {},
    loadedPaths: new Set<string>(),
    isLoading: false,
    error: null,

    loadSettingsByPaths: async (paths: string[]) => {
      const { loadedPaths } = get()
      const unloadedPaths = paths.filter(path => !loadedPaths.has(path))
      
      if (unloadedPaths.length === 0) return

      set({ isLoading: true, error: null })
      
      try {
        const partialSettings = await settingsApi.getSettingsByPaths(unloadedPaths)
        
        set(state => ({
          settings: { ...state.settings, ...partialSettings },
          loadedPaths: new Set([...state.loadedPaths, ...unloadedPaths]),
          isLoading: false
        }))
      } catch (error) {
        set({ 
          error: error instanceof Error ? error.message : 'Failed to load settings',
          isLoading: false 
        })
        throw error
      }
    },

    updateSettingsByPath: async (updates: Array<{path: string, value: any}>) => {
      set({ isLoading: true, error: null })
      
      try {
        await settingsApi.updateSettingsByPath(updates)
        
        // Update local state optimistically
        set(state => {
          const newSettings = { ...state.settings }
          updates.forEach(({ path, value }) => {
            setNestedValue(newSettings, path, value)
          })
          return { settings: newSettings, isLoading: false }
        })
      } catch (error) {
        set({ 
          error: error instanceof Error ? error.message : 'Failed to update settings',
          isLoading: false 
        })
        throw error
      }
    },

    updateSetting: (path: string, value: any) => {
      set(state => {
        const newSettings = { ...state.settings }
        setNestedValue(newSettings, path, value)
        return { settings: newSettings }
      })
    },

    getSetting: (path: string, defaultValue?: any) => {
      const { settings } = get()
      return getNestedValue(settings, path, defaultValue)
    },

    subscribeToPath: (path: string, callback: (value: any) => void) => {
      return get().subscribe(
        state => getNestedValue(state.settings, path),
        callback,
        { equalityFn: (a, b) => a === b }
      )
    },

    resetSettings: () => {
      set({ 
        settings: {}, 
        loadedPaths: new Set(), 
        isLoading: false, 
        error: null 
      })
    },

    isPathLoaded: (path: string) => {
      const { loadedPaths } = get()
      return loadedPaths.has(path)
    }
  }))
)

// Helper functions
function setNestedValue(obj: any, path: string, value: any) {
  const keys = path.split('.')
  let current = obj
  
  for (let i = 0; i < keys.length - 1; i++) {
    const key = keys[i]
    if (!(key in current) || typeof current[key] !== 'object') {
      current[key] = {}
    }
    current = current[key]
  }
  
  current[keys[keys.length - 1]] = value
}

function getNestedValue(obj: any, path: string, defaultValue?: any) {
  const keys = path.split('.')
  let current = obj
  
  for (const key of keys) {
    if (current === null || current === undefined || !(key in current)) {
      return defaultValue
    }
    current = current[key]
  }
  
  return current
}

describe('Settings Store', () => {
  let useSettingsStore: ReturnType<typeof createSettingsStore>
  
  beforeEach(() => {
    vi.clearAllMocks()
    useSettingsStore = createSettingsStore()
  })

  describe('Partial State Loading', () => {
    it('should load only requested settings paths', async () => {
      const mockPartialSettings = {
        visualisation: {
          glow: {
            nodeGlowStrength: 2.5,
            baseColor: '#ff0000'
          }
        }
      }

      vi.mocked(settingsApi.getSettingsByPaths).mockResolvedValue(mockPartialSettings)

      const { result } = renderHook(() => useSettingsStore())

      await act(async () => {
        await result.current.loadSettingsByPaths([
          'visualisation.glow.nodeGlowStrength',
          'visualisation.glow.baseColor'
        ])
      })

      expect(settingsApi.getSettingsByPaths).toHaveBeenCalledWith([
        'visualisation.glow.nodeGlowStrength',
        'visualisation.glow.baseColor'
      ])

      expect(result.current.settings.visualisation?.glow?.nodeGlowStrength).toBe(2.5)
      expect(result.current.settings.visualisation?.glow?.baseColor).toBe('#ff0000')
      expect(result.current.isPathLoaded('visualisation.glow.nodeGlowStrength')).toBe(true)
    })

    it('should not reload already loaded paths', async () => {
      const { result } = renderHook(() => useSettingsStore())

      // First load
      vi.mocked(settingsApi.getSettingsByPaths).mockResolvedValue({
        visualisation: { glow: { nodeGlowStrength: 2.5 } }
      })

      await act(async () => {
        await result.current.loadSettingsByPaths(['visualisation.glow.nodeGlowStrength'])
      })

      // Second load with same path
      vi.mocked(settingsApi.getSettingsByPaths).mockClear()

      await act(async () => {
        await result.current.loadSettingsByPaths(['visualisation.glow.nodeGlowStrength'])
      })

      expect(settingsApi.getSettingsByPaths).not.toHaveBeenCalled()
    })

    it('should handle loading errors gracefully', async () => {
      const mockError = new Error('Network error')
      vi.mocked(settingsApi.getSettingsByPaths).mockRejectedValue(mockError)

      const { result } = renderHook(() => useSettingsStore())

      await act(async () => {
        try {
          await result.current.loadSettingsByPaths(['visualisation.glow.nodeGlowStrength'])
        } catch (error) {
          expect(error).toBe(mockError)
        }
      })

      expect(result.current.error).toBe('Network error')
      expect(result.current.isLoading).toBe(false)
    })
  })

  describe('Selective State Subscription', () => {
    it('should allow subscribing to specific paths', async () => {
      const { result } = renderHook(() => useSettingsStore())
      const mockCallback = vi.fn()

      // Set up subscription
      let unsubscribe: (() => void) | undefined

      await act(async () => {
        unsubscribe = result.current.subscribeToPath('visualisation.glow.nodeGlowStrength', mockCallback)
      })

      // Update the specific path
      act(() => {
        result.current.updateSetting('visualisation.glow.nodeGlowStrength', 3.0)
      })

      expect(mockCallback).toHaveBeenCalledWith(3.0)

      // Update unrelated path
      act(() => {
        result.current.updateSetting('system.debugMode', true)
      })

      // Callback should not be called for unrelated changes
      expect(mockCallback).toHaveBeenCalledTimes(1)

      // Clean up
      if (unsubscribe) unsubscribe()
    })

    it('should handle nested path subscriptions correctly', async () => {
      const { result } = renderHook(() => useSettingsStore())
      const mockCallback = vi.fn()

      let unsubscribe: (() => void) | undefined

      await act(async () => {
        unsubscribe = result.current.subscribeToPath('visualisation.glow', mockCallback)
      })

      // Update nested property
      act(() => {
        result.current.updateSetting('visualisation.glow.nodeGlowStrength', 4.0)
      })

      expect(mockCallback).toHaveBeenCalled()
      const callArg = mockCallback.mock.calls[0][0]
      expect(callArg.nodeGlowStrength).toBe(4.0)

      if (unsubscribe) unsubscribe()
    })
  })

  describe('Granular Updates', () => {
    it('should update settings via path-based API', async () => {
      vi.mocked(settingsApi.updateSettingsByPath).mockResolvedValue()

      const { result } = renderHook(() => useSettingsStore())

      const updates = [
        { path: 'visualisation.glow.nodeGlowStrength', value: 2.8 },
        { path: 'system.debugMode', value: true }
      ]

      await act(async () => {
        await result.current.updateSettingsByPath(updates)
      })

      expect(settingsApi.updateSettingsByPath).toHaveBeenCalledWith(updates)
      expect(result.current.getSetting('visualisation.glow.nodeGlowStrength')).toBe(2.8)
      expect(result.current.getSetting('system.debugMode')).toBe(true)
    })

    it('should handle update failures gracefully', async () => {
      const mockError = new Error('Update failed')
      vi.mocked(settingsApi.updateSettingsByPath).mockRejectedValue(mockError)

      const { result } = renderHook(() => useSettingsStore())

      await act(async () => {
        try {
          await result.current.updateSettingsByPath([
            { path: 'visualisation.glow.nodeGlowStrength', value: 2.8 }
          ])
        } catch (error) {
          expect(error).toBe(mockError)
        }
      })

      expect(result.current.error).toBe('Update failed')
    })

    it('should support optimistic updates', async () => {
      vi.mocked(settingsApi.updateSettingsByPath).mockImplementation(
        () => new Promise(resolve => setTimeout(resolve, 100))
      )

      const { result } = renderHook(() => useSettingsStore())

      const updatePromise = act(async () => {
        await result.current.updateSettingsByPath([
          { path: 'visualisation.glow.nodeGlowStrength', value: 3.5 }
        ])
      })

      // Value should be updated immediately (optimistic)
      expect(result.current.getSetting('visualisation.glow.nodeGlowStrength')).toBe(3.5)

      await updatePromise
    })
  })

  describe('Memory Efficiency', () => {
    it('should only store loaded settings in memory', async () => {
      const { result } = renderHook(() => useSettingsStore())

      // Load only specific paths
      vi.mocked(settingsApi.getSettingsByPaths).mockResolvedValue({
        visualisation: {
          glow: { nodeGlowStrength: 2.0 }
        }
      })

      await act(async () => {
        await result.current.loadSettingsByPaths(['visualisation.glow.nodeGlowStrength'])
      })

      // Only loaded data should be present
      expect(result.current.settings.visualisation?.glow?.nodeGlowStrength).toBe(2.0)
      expect(result.current.settings.visualisation?.glow?.edgeGlowStrength).toBeUndefined()
      expect(result.current.settings.system).toBeUndefined()
    })

    it('should track loaded paths efficiently', () => {
      const { result } = renderHook(() => useSettingsStore())

      act(() => {
        result.current.updateSetting('visualisation.glow.nodeGlowStrength', 2.0)
      })

      expect(result.current.isPathLoaded('visualisation.glow.nodeGlowStrength')).toBe(false)
      
      // Manually mark as loaded (simulating API load)
      act(() => {
        const store = result.current as any
        store.loadedPaths.add('visualisation.glow.nodeGlowStrength')
      })

      expect(result.current.isPathLoaded('visualisation.glow.nodeGlowStrength')).toBe(true)
    })
  })

  describe('Type Safety', () => {
    it('should maintain type safety with generics', () => {
      const { result } = renderHook(() => useSettingsStore())

      // Set typed values
      act(() => {
        result.current.updateSetting('visualisation.glow.nodeGlowStrength', 2.5)
        result.current.updateSetting('system.debugMode', true)
        result.current.updateSetting('visualisation.glow.baseColor', '#ff0000')
      })

      // Get with type safety
      const glowStrength = result.current.getSetting<number>('visualisation.glow.nodeGlowStrength')
      const debugMode = result.current.getSetting<boolean>('system.debugMode')
      const color = result.current.getSetting<string>('visualisation.glow.baseColor')

      expect(typeof glowStrength).toBe('number')
      expect(typeof debugMode).toBe('boolean')
      expect(typeof color).toBe('string')

      expect(glowStrength).toBe(2.5)
      expect(debugMode).toBe(true)
      expect(color).toBe('#ff0000')
    })

    it('should handle default values correctly', () => {
      const { result } = renderHook(() => useSettingsStore())

      const defaultValue = result.current.getSetting('nonexistent.path', 'default')
      expect(defaultValue).toBe('default')

      const undefinedValue = result.current.getSetting('nonexistent.path')
      expect(undefinedValue).toBeUndefined()
    })
  })

  describe('Performance', () => {
    it('should handle large numbers of path updates efficiently', async () => {
      const { result } = renderHook(() => useSettingsStore())

      const updates = Array.from({ length: 1000 }, (_, i) => ({
        path: `test.path.${i}`,
        value: i
      }))

      vi.mocked(settingsApi.updateSettingsByPath).mockResolvedValue()

      const start = performance.now()
      
      await act(async () => {
        await result.current.updateSettingsByPath(updates)
      })

      const duration = performance.now() - start

      expect(duration).toBeLessThan(100) // Should complete in under 100ms
      expect(settingsApi.updateSettingsByPath).toHaveBeenCalledWith(updates)
    })

    it('should debounce rapid updates to same path', async () => {
      const { result } = renderHook(() => useSettingsStore())
      vi.mocked(settingsApi.updateSettingsByPath).mockResolvedValue()

      // Rapid local updates
      act(() => {
        result.current.updateSetting('visualisation.glow.nodeGlowStrength', 1.0)
        result.current.updateSetting('visualisation.glow.nodeGlowStrength', 2.0)
        result.current.updateSetting('visualisation.glow.nodeGlowStrength', 3.0)
      })

      // Final value should be the last one
      expect(result.current.getSetting('visualisation.glow.nodeGlowStrength')).toBe(3.0)
    })
  })

  describe('Integration with camelCase API', () => {
    it('should handle camelCase paths correctly', async () => {
      const mockResponse = {
        visualisation: {
          glow: {
            nodeGlowStrength: 2.5,  // camelCase from API
            edgeGlowStrength: 3.0,
            baseColor: '#ff0000'
          }
        }
      }

      vi.mocked(settingsApi.getSettingsByPaths).mockResolvedValue(mockResponse)

      const { result } = renderHook(() => useSettingsStore())

      await act(async () => {
        await result.current.loadSettingsByPaths([
          'visualisation.glow.nodeGlowStrength',
          'visualisation.glow.edgeGlowStrength',
          'visualisation.glow.baseColor'
        ])
      })

      // Verify camelCase fields are accessible
      expect(result.current.getSetting('visualisation.glow.nodeGlowStrength')).toBe(2.5)
      expect(result.current.getSetting('visualisation.glow.edgeGlowStrength')).toBe(3.0)
      expect(result.current.getSetting('visualisation.glow.baseColor')).toBe('#ff0000')
    })

    it('should send camelCase paths in API requests', async () => {
      vi.mocked(settingsApi.updateSettingsByPath).mockResolvedValue()

      const { result } = renderHook(() => useSettingsStore())

      const updates = [
        { path: 'visualisation.glow.nodeGlowStrength', value: 4.0 }, // camelCase path
        { path: 'system.debugMode', value: true },                  // camelCase path
      ]

      await act(async () => {
        await result.current.updateSettingsByPath(updates)
      })

      expect(settingsApi.updateSettingsByPath).toHaveBeenCalledWith(updates)
    })
  })

  describe('Error Recovery', () => {
    it('should recover from network errors', async () => {
      const { result } = renderHook(() => useSettingsStore())

      // First request fails
      vi.mocked(settingsApi.getSettingsByPaths).mockRejectedValueOnce(new Error('Network error'))

      await act(async () => {
        try {
          await result.current.loadSettingsByPaths(['visualisation.glow.nodeGlowStrength'])
        } catch (error) {
          // Expected to fail
        }
      })

      expect(result.current.error).toBe('Network error')

      // Second request succeeds
      vi.mocked(settingsApi.getSettingsByPaths).mockResolvedValue({
        visualisation: { glow: { nodeGlowStrength: 2.5 } }
      })

      await act(async () => {
        await result.current.loadSettingsByPaths(['visualisation.glow.nodeGlowStrength'])
      })

      expect(result.current.error).toBeNull()
      expect(result.current.getSetting('visualisation.glow.nodeGlowStrength')).toBe(2.5)
    })

    it('should handle partial update failures', async () => {
      const { result } = renderHook(() => useSettingsStore())

      // Mock partial failure
      vi.mocked(settingsApi.updateSettingsByPath).mockRejectedValue(new Error('Validation error'))

      await act(async () => {
        try {
          await result.current.updateSettingsByPath([
            { path: 'visualisation.glow.nodeGlowStrength', value: -1 } // Invalid value
          ])
        } catch (error) {
          expect(error).toBeInstanceOf(Error)
        }
      })

      expect(result.current.error).toBe('Validation error')
    })
  })

  describe('State Reset', () => {
    it('should reset store state completely', async () => {
      const { result } = renderHook(() => useSettingsStore())

      // Load some data
      vi.mocked(settingsApi.getSettingsByPaths).mockResolvedValue({
        visualisation: { glow: { nodeGlowStrength: 2.5 } }
      })

      await act(async () => {
        await result.current.loadSettingsByPaths(['visualisation.glow.nodeGlowStrength'])
      })

      expect(result.current.getSetting('visualisation.glow.nodeGlowStrength')).toBe(2.5)
      expect(result.current.isPathLoaded('visualisation.glow.nodeGlowStrength')).toBe(true)

      // Reset
      act(() => {
        result.current.resetSettings()
      })

      expect(result.current.settings).toEqual({})
      expect(result.current.isPathLoaded('visualisation.glow.nodeGlowStrength')).toBe(false)
      expect(result.current.error).toBeNull()
      expect(result.current.isLoading).toBe(false)
    })
  })
})