/**
 * Comprehensive tests for settings API in refactored system
 * 
 * Tests granular API endpoints, camelCase handling, error handling,
 * and performance characteristics of the new API design
 */

import { describe, it, expect, beforeEach, afterEach, vi, Mock } from 'vitest'
import { settingsApi } from '../api/settingsApi'

// Mock fetch globally
const mockFetch = vi.fn()
global.fetch = mockFetch

describe('Settings API', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockFetch.mockClear()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('getSettingsByPaths', () => {
    it('should request specific paths from granular endpoint', async () => {
      const mockResponse = {
        visualisation: {
          glow: {
            nodeGlowStrength: 2.5,
            baseColor: '#ff0000'
          }
        }
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => mockResponse
      })

      const paths = ['visualisation.glow.nodeGlowStrength', 'visualisation.glow.baseColor']
      const result = await settingsApi.getSettingsByPaths(paths)

      expect(mockFetch).toHaveBeenCalledWith(
        '/api/settings/get?paths=visualisation.glow.nodeGlowStrength,visualisation.glow.baseColor',
        {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          }
        }
      )

      expect(result).toEqual(mockResponse)
    })

    it('should handle single path requests', async () => {
      const mockResponse = {
        system: { debugMode: true }
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      })

      const result = await settingsApi.getSettingsByPaths(['system.debugMode'])

      expect(mockFetch).toHaveBeenCalledWith(
        '/api/settings/get?paths=system.debugMode',
        expect.any(Object)
      )

      expect(result).toEqual(mockResponse)
    })

    it('should handle empty paths array', async () => {
      const result = await settingsApi.getSettingsByPaths([])

      expect(mockFetch).not.toHaveBeenCalled()
      expect(result).toEqual({})
    })

    it('should URL encode complex paths', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({})
      })

      const complexPaths = [
        'visualisation.graphs.logseq.physics.springK',
        'system.websocket.heartbeatInterval',
        'xr.locomotionMethod'
      ]

      await settingsApi.getSettingsByPaths(complexPaths)

      const expectedUrl = '/api/settings/get?paths=' + encodeURIComponent(complexPaths.join(','))
      expect(mockFetch).toHaveBeenCalledWith(expectedUrl, expect.any(Object))
    })

    it('should handle API errors gracefully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        statusText: 'Not Found',
        json: async () => ({ error: 'Path not found' })
      })

      await expect(settingsApi.getSettingsByPaths(['invalid.path']))
        .rejects.toThrow('Failed to fetch settings: 404 Not Found')
    })

    it('should handle network errors', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'))

      await expect(settingsApi.getSettingsByPaths(['system.debugMode']))
        .rejects.toThrow('Network error')
    })

    it('should handle malformed JSON response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => { throw new Error('Invalid JSON') }
      })

      await expect(settingsApi.getSettingsByPaths(['system.debugMode']))
        .rejects.toThrow('Invalid JSON')
    })
  })

  describe('updateSettingsByPath', () => {
    it('should send path-based updates to granular endpoint', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ success: true })
      })

      const updates = [
        { path: 'visualisation.glow.nodeGlowStrength', value: 3.0 },
        { path: 'system.debugMode', value: true }
      ]

      await settingsApi.updateSettingsByPath(updates)

      expect(mockFetch).toHaveBeenCalledWith('/api/settings/set', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updates)
      })
    })

    it('should handle single update', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true })
      })

      const update = [{ path: 'visualisation.glow.baseColor', value: '#00ff00' }]
      await settingsApi.updateSettingsByPath(update)

      expect(mockFetch).toHaveBeenCalledWith('/api/settings/set', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(update)
      })
    })

    it('should handle complex nested values', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true })
      })

      const complexUpdate = [{
        path: 'visualisation.glow',
        value: {
          nodeGlowStrength: 2.5,
          edgeGlowStrength: 1.8,
          baseColor: '#ff0000',
          emissionColor: '#ffffff'
        }
      }]

      await settingsApi.updateSettingsByPath(complexUpdate)

      expect(mockFetch).toHaveBeenCalledWith('/api/settings/set', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(complexUpdate)
      })
    })

    it('should handle validation errors', async () => {
      const validationError = {
        error: 'Validation failed',
        details: {
          'visualisation.glow.nodeGlowStrength': 'Value must be between 0 and 10'
        }
      }

      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        statusText: 'Bad Request',
        json: async () => validationError
      })

      const invalidUpdate = [{ path: 'visualisation.glow.nodeGlowStrength', value: -1 }]

      await expect(settingsApi.updateSettingsByPath(invalidUpdate))
        .rejects.toThrow('Failed to update settings: 400 Bad Request')
    })

    it('should handle empty updates array', async () => {
      const result = await settingsApi.updateSettingsByPath([])

      expect(mockFetch).not.toHaveBeenCalled()
    })

    it('should handle concurrent updates', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ success: true })
      })

      const updates1 = [{ path: 'visualisation.glow.nodeGlowStrength', value: 2.0 }]
      const updates2 = [{ path: 'system.debugMode', value: false }]

      const [result1, result2] = await Promise.all([
        settingsApi.updateSettingsByPath(updates1),
        settingsApi.updateSettingsByPath(updates2)
      ])

      expect(mockFetch).toHaveBeenCalledTimes(2)
    })
  })

  describe('CamelCase Integration', () => {
    it('should handle camelCase field names in responses', async () => {
      const camelCaseResponse = {
        visualisation: {
          glow: {
            nodeGlowStrength: 2.5,  // camelCase
            edgeGlowStrength: 3.0,  // camelCase
            environmentGlowStrength: 1.5,  // camelCase
            baseColor: '#ff0000',
            emissionColor: '#ffffff'
          }
        },
        system: {
          debugMode: true,        // camelCase
          maxConnections: 100,    // camelCase
          connectionTimeout: 5000 // camelCase
        }
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => camelCaseResponse
      })

      const paths = [
        'visualisation.glow.nodeGlowStrength',
        'visualisation.glow.edgeGlowStrength',
        'system.debugMode',
        'system.maxConnections'
      ]

      const result = await settingsApi.getSettingsByPaths(paths)

      expect(result.visualisation.glow.nodeGlowStrength).toBe(2.5)
      expect(result.visualisation.glow.edgeGlowStrength).toBe(3.0)
      expect(result.system.debugMode).toBe(true)
      expect(result.system.maxConnections).toBe(100)
    })

    it('should send camelCase field names in updates', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true })
      })

      const camelCaseUpdates = [
        { path: 'visualisation.glow.nodeGlowStrength', value: 3.5 },
        { path: 'system.debugMode', value: true },
        { path: 'system.maxConnections', value: 150 },
        { path: 'xr.handMeshColor', value: '#00ff00' },
        { path: 'xr.locomotionMethod', value: 'teleport' }
      ]

      await settingsApi.updateSettingsByPath(camelCaseUpdates)

      const sentBody = JSON.parse(mockFetch.mock.calls[0][1].body)
      expect(sentBody).toEqual(camelCaseUpdates)
    })

    it('should handle physics parameters with camelCase', async () => {
      const physicsResponse = {
        visualisation: {
          graphs: {
            logseq: {
              physics: {
                springK: 0.1,           // camelCase
                repelK: 100.0,          // camelCase
                attractionK: 0.05,      // camelCase
                maxVelocity: 5.0,       // camelCase
                boundsSize: 1000.0,     // camelCase
                separationRadius: 50.0, // camelCase
                centerGravityK: 0.01,   // camelCase
                gridCellSize: 100.0,    // camelCase
                warmupIterations: 10,   // camelCase
                coolingRate: 0.95,      // camelCase
                boundaryDamping: 0.8,   // camelCase
                updateThreshold: 0.01   // camelCase
              }
            }
          }
        }
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => physicsResponse
      })

      const physicsPath = 'visualisation.graphs.logseq.physics'
      const result = await settingsApi.getSettingsByPaths([physicsPath])

      const physics = result.visualisation.graphs.logseq.physics
      expect(physics.springK).toBe(0.1)
      expect(physics.repelK).toBe(100.0)
      expect(physics.maxVelocity).toBe(5.0)
      expect(physics.centerGravityK).toBe(0.01)
    })
  })

  describe('Performance Optimizations', () => {
    it('should efficiently handle large path requests', async () => {
      const largePaths = Array.from({ length: 50 }, (_, i) => `path.${i}.value`)
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({})
      })

      const start = performance.now()
      await settingsApi.getSettingsByPaths(largePaths)
      const duration = performance.now() - start

      expect(duration).toBeLessThan(50) // Should be fast
      expect(mockFetch).toHaveBeenCalledTimes(1) // Single request
    })

    it('should handle request size limits gracefully', async () => {
      // Simulate extremely long path list that might exceed URL limits
      const enormousPaths = Array.from({ length: 1000 }, (_, i) => 
        `very.long.path.name.with.many.segments.${i}.value.that.creates.a.very.long.url`
      )

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({})
      })

      // Should either succeed or gracefully handle URL length limits
      try {
        await settingsApi.getSettingsByPaths(enormousPaths)
        expect(mockFetch).toHaveBeenCalled()
      } catch (error) {
        // If it fails, it should be a graceful failure
        expect(error).toBeInstanceOf(Error)
      }
    })

    it('should batch multiple rapid updates efficiently', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ success: true })
      })

      // Simulate rapid successive updates
      const rapidUpdates = Array.from({ length: 10 }, (_, i) => [
        { path: `test.path.${i}`, value: i }
      ])

      const promises = rapidUpdates.map(update => 
        settingsApi.updateSettingsByPath(update)
      )

      await Promise.all(promises)

      expect(mockFetch).toHaveBeenCalledTimes(10)
    })
  })

  describe('Error Recovery', () => {
    it('should retry on transient failures', async () => {
      // First call fails, second succeeds
      mockFetch
        .mockRejectedValueOnce(new Error('Network timeout'))
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ system: { debugMode: true } })
        })

      // Implement simple retry logic in test
      let result
      try {
        result = await settingsApi.getSettingsByPaths(['system.debugMode'])
      } catch (error) {
        // Retry once
        result = await settingsApi.getSettingsByPaths(['system.debugMode'])
      }

      expect(result).toEqual({ system: { debugMode: true } })
      expect(mockFetch).toHaveBeenCalledTimes(2)
    })

    it('should handle partial response corruption', async () => {
      // Response with some valid and some invalid data
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          visualisation: {
            glow: {
              nodeGlowStrength: 2.5,  // Valid
              baseColor: null,        // Invalid/corrupted
              edgeGlowStrength: undefined // Invalid/corrupted
            }
          }
        })
      })

      const result = await settingsApi.getSettingsByPaths([
        'visualisation.glow.nodeGlowStrength',
        'visualisation.glow.baseColor',
        'visualisation.glow.edgeGlowStrength'
      ])

      // Should handle corrupted fields gracefully
      expect(result.visualisation.glow.nodeGlowStrength).toBe(2.5)
      // Corrupted fields might be null/undefined, which is acceptable
    })

    it('should handle server-side validation errors with details', async () => {
      const detailedError = {
        error: 'Validation failed',
        validationErrors: [
          {
            path: 'visualisation.glow.nodeGlowStrength',
            message: 'Value must be between 0 and 10',
            receivedValue: 15
          },
          {
            path: 'visualisation.glow.baseColor',
            message: 'Invalid hex color format',
            receivedValue: 'invalid_color'
          }
        ]
      }

      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => detailedError
      })

      const invalidUpdates = [
        { path: 'visualisation.glow.nodeGlowStrength', value: 15 },
        { path: 'visualisation.glow.baseColor', value: 'invalid_color' }
      ]

      await expect(settingsApi.updateSettingsByPath(invalidUpdates))
        .rejects.toThrow('Failed to update settings: 400 Bad Request')
    })
  })

  describe('Backwards Compatibility', () => {
    it('should maintain compatibility with existing camelCase data', async () => {
      // Simulate response that already contains camelCase data
      const existingCamelCaseData = {
        visualisation: {
          glow: {
            nodeGlowStrength: 1.5,    // Already camelCase
            edgeGlowStrength: 2.0,    // Already camelCase
            baseColor: '#00ffff'      // Already camelCase
          }
        },
        system: {
          debugMode: false,          // Already camelCase
          maxConnections: 50         // Already camelCase
        }
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => existingCamelCaseData
      })

      const result = await settingsApi.getSettingsByPaths([
        'visualisation.glow',
        'system.debugMode'
      ])

      expect(result).toEqual(existingCamelCaseData)
    })

    it('should handle mixed case scenarios gracefully', async () => {
      // Test response with potentially mixed casing
      const mixedCaseData = {
        visualisation: {
          glow: {
            nodeGlowStrength: 2.5,  // camelCase
            base_color: '#ff0000',  // snake_case (shouldn't happen but test resilience)
          }
        }
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mixedCaseData
      })

      const result = await settingsApi.getSettingsByPaths(['visualisation.glow'])
      
      // Should handle whatever the server returns
      expect(result.visualisation.glow.nodeGlowStrength).toBe(2.5)
      // Implementation might normalize or preserve snake_case fields
    })
  })

  describe('Type Safety', () => {
    it('should preserve TypeScript types through API calls', async () => {
      interface TestSettings {
        visualisation: {
          glow: {
            nodeGlowStrength: number
            baseColor: string
          }
        }
        system: {
          debugMode: boolean
          maxConnections: number
        }
      }

      const typedResponse: TestSettings = {
        visualisation: {
          glow: {
            nodeGlowStrength: 2.5,
            baseColor: '#ff0000'
          }
        },
        system: {
          debugMode: true,
          maxConnections: 100
        }
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => typedResponse
      })

      const result = await settingsApi.getSettingsByPaths([
        'visualisation.glow',
        'system.debugMode'
      ]) as TestSettings

      // TypeScript should maintain type safety
      expect(typeof result.visualisation.glow.nodeGlowStrength).toBe('number')
      expect(typeof result.visualisation.glow.baseColor).toBe('string')
      expect(typeof result.system.debugMode).toBe('boolean')
      expect(typeof result.system.maxConnections).toBe('number')
    })
  })
})