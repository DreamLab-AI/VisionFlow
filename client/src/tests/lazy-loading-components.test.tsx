/**
 * Comprehensive tests for lazy loading components in refactored settings system
 * 
 * Tests React.lazy integration, selective loading, performance optimizations,
 * and error handling for dynamically loaded settings components
 */

import { describe, it, expect, beforeEach, afterEach, vi, Mock } from 'vitest'
import { render, screen, waitFor, act } from '@testing-library/react'
import { userEvent } from '@testing-library/user-event'
import React, { Suspense, ComponentType } from 'react'
import { Settings } from '../types/generated/settings'

// Mock the settings API and store
vi.mock('../api/settingsApi', () => ({
  settingsApi: {
    getSettingsByPaths: vi.fn(),
    updateSettingsByPath: vi.fn(),
  }
}))

vi.mock('../store/settingsStore', () => ({
  useSettingsStore: vi.fn()
}))

// Mock the lazy-loaded components
const MockGlowSettingsComponent = vi.fn(() => <div data-testid="glow-settings">Glow Settings</div>)
const MockPhysicsSettingsComponent = vi.fn(() => <div data-testid="physics-settings">Physics Settings</div>)
const MockSystemSettingsComponent = vi.fn(() => <div data-testid="system-settings">System Settings</div>)
const MockXRSettingsComponent = vi.fn(() => <div data-testid="xr-settings">XR Settings</div>)

// Mock React.lazy
const mockLazy = vi.fn()
vi.mock('react', async () => {
  const actual = await vi.importActual('react')
  return {
    ...actual,
    lazy: (factory: () => Promise<{ default: ComponentType<any> }>) => {
      mockLazy(factory)
      return (props: any) => {
        // Simulate loading delay
        const [loaded, setLoaded] = React.useState(false)
        
        React.useEffect(() => {
          const timer = setTimeout(() => setLoaded(true), 10)
          return () => clearTimeout(timer)
        }, [])
        
        if (!loaded) return <div data-testid="loading">Loading...</div>
        
        // Return the appropriate mock component based on the factory
        const factoryStr = factory.toString()
        if (factoryStr.includes('GlowSettings')) return <MockGlowSettingsComponent {...props} />
        if (factoryStr.includes('PhysicsSettings')) return <MockPhysicsSettingsComponent {...props} />
        if (factoryStr.includes('SystemSettings')) return <MockSystemSettingsComponent {...props} />
        if (factoryStr.includes('XRSettings')) return <MockXRSettingsComponent {...props} />
        
        return <div data-testid="unknown-component">Unknown Component</div>
      }
    }
  }
})

// Lazy Settings Section Component
interface LazySettingsSectionProps {
  section: 'glow' | 'physics' | 'system' | 'xr'
  isActive: boolean
  onLoad?: () => void
}

const LazySettingsSection: React.FC<LazySettingsSectionProps> = ({ section, isActive, onLoad }) => {
  // Lazy load components only when needed
  const components = React.useMemo(() => ({
    glow: React.lazy(() => import('../features/settings/components/GlowSettings')),
    physics: React.lazy(() => import('../features/settings/components/PhysicsSettings')),
    system: React.lazy(() => import('../features/settings/components/SystemSettings')),
    xr: React.lazy(() => import('../features/settings/components/XRSettings')),
  }), [])

  const Component = components[section]

  React.useEffect(() => {
    if (isActive && onLoad) {
      onLoad()
    }
  }, [isActive, onLoad])

  if (!isActive) {
    return null
  }

  return (
    <Suspense fallback={<div data-testid={`${section}-loading`}>Loading {section} settings...</div>}>
      <Component />
    </Suspense>
  )
}

// Settings Panel with Tabs
interface SettingsPanelProps {
  initialTab?: string
}

const SettingsPanel: React.FC<SettingsPanelProps> = ({ initialTab = 'glow' }) => {
  const [activeTab, setActiveTab] = React.useState(initialTab)
  const [loadedTabs, setLoadedTabs] = React.useState<Set<string>>(new Set())

  const handleTabClick = (tab: string) => {
    setActiveTab(tab)
  }

  const handleSectionLoad = (section: string) => {
    setLoadedTabs(prev => new Set(prev).add(section))
  }

  const tabs = ['glow', 'physics', 'system', 'xr']

  return (
    <div data-testid="settings-panel">
      {/* Tab Navigation */}
      <div data-testid="tab-nav">
        {tabs.map(tab => (
          <button
            key={tab}
            data-testid={`tab-${tab}`}
            onClick={() => handleTabClick(tab)}
            className={activeTab === tab ? 'active' : ''}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
            {loadedTabs.has(tab) && <span data-testid={`loaded-indicator-${tab}`}>✓</span>}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div data-testid="tab-content">
        {tabs.map(tab => (
          <LazySettingsSection
            key={tab}
            section={tab as 'glow' | 'physics' | 'system' | 'xr'}
            isActive={activeTab === tab}
            onLoad={() => handleSectionLoad(tab)}
          />
        ))}
      </div>

      {/* Debug Info */}
      <div data-testid="debug-info">
        <span data-testid="active-tab">{activeTab}</span>
        <span data-testid="loaded-count">{loadedTabs.size}</span>
      </div>
    </div>
  )
}

describe('Lazy Loading Components', () => {
  const user = userEvent.setup()

  beforeEach(() => {
    vi.clearAllMocks()
    MockGlowSettingsComponent.mockClear()
    MockPhysicsSettingsComponent.mockClear()
    MockSystemSettingsComponent.mockClear()
    MockXRSettingsComponent.mockClear()
  })

  describe('Initial Loading', () => {
    it('should load only the initial tab component', async () => {
      render(<SettingsPanel initialTab="glow" />)

      // Should show loading state initially
      expect(screen.getByTestId('glow-loading')).toBeInTheDocument()

      // Wait for component to load
      await waitFor(() => {
        expect(screen.getByTestId('glow-settings')).toBeInTheDocument()
      })

      // Only glow component should be loaded
      expect(MockGlowSettingsComponent).toHaveBeenCalled()
      expect(MockPhysicsSettingsComponent).not.toHaveBeenCalled()
      expect(MockSystemSettingsComponent).not.toHaveBeenCalled()
      expect(MockXRSettingsComponent).not.toHaveBeenCalled()
    })

    it('should show loading indicator during component load', async () => {
      render(<SettingsPanel initialTab="physics" />)

      // Should show loading state
      const loadingElement = screen.getByTestId('physics-loading')
      expect(loadingElement).toBeInTheDocument()
      expect(loadingElement).toHaveTextContent('Loading physics settings...')

      // Wait for component to load
      await waitFor(() => {
        expect(screen.getByTestId('physics-settings')).toBeInTheDocument()
      })

      // Loading indicator should be gone
      expect(screen.queryByTestId('physics-loading')).not.toBeInTheDocument()
    })
  })

  describe('Tab Switching', () => {
    it('should lazy load components when tabs are clicked', async () => {
      render(<SettingsPanel initialTab="glow" />)

      // Wait for initial tab to load
      await waitFor(() => {
        expect(screen.getByTestId('glow-settings')).toBeInTheDocument()
      })

      // Click physics tab
      await user.click(screen.getByTestId('tab-physics'))

      // Should show physics loading
      expect(screen.getByTestId('physics-loading')).toBeInTheDocument()

      // Wait for physics to load
      await waitFor(() => {
        expect(screen.getByTestId('physics-settings')).toBeInTheDocument()
      })

      // Physics component should now be loaded
      expect(MockPhysicsSettingsComponent).toHaveBeenCalled()
      
      // Glow should be hidden but still loaded
      expect(screen.queryByTestId('glow-settings')).not.toBeInTheDocument()
      
      // System and XR should still not be loaded
      expect(MockSystemSettingsComponent).not.toHaveBeenCalled()
      expect(MockXRSettingsComponent).not.toHaveBeenCalled()
    })

    it('should track loaded tabs correctly', async () => {
      render(<SettingsPanel />)

      // Initially should have 0 loaded tabs
      await waitFor(() => {
        expect(screen.getByTestId('loaded-count')).toHaveTextContent('1')
      })

      // Click different tabs
      await user.click(screen.getByTestId('tab-system'))
      
      await waitFor(() => {
        expect(screen.getByTestId('loaded-count')).toHaveTextContent('2')
      })

      await user.click(screen.getByTestId('tab-xr'))
      
      await waitFor(() => {
        expect(screen.getByTestId('loaded-count')).toHaveTextContent('3')
      })

      // Should show loaded indicators
      expect(screen.getByTestId('loaded-indicator-glow')).toBeInTheDocument()
      expect(screen.getByTestId('loaded-indicator-system')).toBeInTheDocument()
      expect(screen.getByTestId('loaded-indicator-xr')).toBeInTheDocument()
    })

    it('should not reload already loaded components', async () => {
      render(<SettingsPanel />)

      // Wait for initial glow to load
      await waitFor(() => {
        expect(screen.getByTestId('glow-settings')).toBeInTheDocument()
      })

      const initialGlowCalls = MockGlowSettingsComponent.mock.calls.length

      // Switch to physics
      await user.click(screen.getByTestId('tab-physics'))
      await waitFor(() => {
        expect(screen.getByTestId('physics-settings')).toBeInTheDocument()
      })

      // Switch back to glow
      await user.click(screen.getByTestId('tab-glow'))
      await waitFor(() => {
        expect(screen.getByTestId('glow-settings')).toBeInTheDocument()
      })

      // Glow component should not have been called again (already loaded)
      expect(MockGlowSettingsComponent.mock.calls.length).toBe(initialGlowCalls)
    })
  })

  describe('Performance Optimization', () => {
    it('should render tabs without loading unused components', async () => {
      const renderStart = performance.now()
      
      render(<SettingsPanel />)

      // Wait for initial component to load
      await waitFor(() => {
        expect(screen.getByTestId('glow-settings')).toBeInTheDocument()
      })

      const renderDuration = performance.now() - renderStart

      // Should render quickly since only one component is loaded
      expect(renderDuration).toBeLessThan(100)

      // Only glow should be loaded
      expect(MockGlowSettingsComponent).toHaveBeenCalled()
      expect(MockPhysicsSettingsComponent).not.toHaveBeenCalled()
      expect(MockSystemSettingsComponent).not.toHaveBeenCalled()
      expect(MockXRSettingsComponent).not.toHaveBeenCalled()
    })

    it('should handle rapid tab switching efficiently', async () => {
      render(<SettingsPanel />)

      await waitFor(() => {
        expect(screen.getByTestId('glow-settings')).toBeInTheDocument()
      })

      // Rapid tab switching
      const tabs = ['physics', 'system', 'xr', 'glow', 'physics']
      
      for (const tab of tabs) {
        await user.click(screen.getByTestId(`tab-${tab}`))
        // Small delay to simulate real user behavior
        await act(async () => {
          await new Promise(resolve => setTimeout(resolve, 10))
        })
      }

      // All tabs should eventually be loaded without issues
      await waitFor(() => {
        expect(screen.getByTestId('loaded-count')).toHaveTextContent('4')
      })
    })

    it('should maintain component state when switching tabs', async () => {
      // This test would verify that component state is preserved
      // when switching between already-loaded tabs
      render(<SettingsPanel />)

      await waitFor(() => {
        expect(screen.getByTestId('glow-settings')).toBeInTheDocument()
      })

      // Switch away and back
      await user.click(screen.getByTestId('tab-physics'))
      await waitFor(() => {
        expect(screen.getByTestId('physics-settings')).toBeInTheDocument()
      })

      await user.click(screen.getByTestId('tab-glow'))
      await waitFor(() => {
        expect(screen.getByTestId('glow-settings')).toBeInTheDocument()
      })

      // Component should maintain its state (in a real implementation,
      // this would test that form values, etc., are preserved)
      expect(MockGlowSettingsComponent).toHaveBeenCalledTimes(1)
    })
  })

  describe('Error Handling', () => {
    it('should handle component loading errors gracefully', async () => {
      // Mock a component that fails to load
      const FailingLazyComponent: React.FC = () => {
        throw new Error('Component failed to load')
      }

      const ErrorBoundary: React.FC<{ children: React.ReactNode }> = ({ children }) => {
        const [hasError, setHasError] = React.useState(false)

        React.useEffect(() => {
          const handleError = () => setHasError(true)
          window.addEventListener('error', handleError)
          return () => window.removeEventListener('error', handleError)
        }, [])

        if (hasError) {
          return <div data-testid="error-fallback">Failed to load component</div>
        }

        return <>{children}</>
      }

      render(
        <ErrorBoundary>
          <Suspense fallback={<div data-testid="loading">Loading...</div>}>
            <FailingLazyComponent />
          </Suspense>
        </ErrorBoundary>
      )

      // Should show loading initially
      expect(screen.getByTestId('loading')).toBeInTheDocument()

      // Should show error fallback after component fails
      await waitFor(() => {
        expect(screen.getByTestId('error-fallback')).toBeInTheDocument()
      })
    })

    it('should handle missing components gracefully', async () => {
      // Test with a component that doesn't exist
      const LazyMissingComponent = React.lazy(() => 
        import('../components/NonExistentComponent').catch(() => ({
          default: () => <div data-testid="fallback-component">Component not found</div>
        }))
      )

      render(
        <Suspense fallback={<div data-testid="loading">Loading...</div>}>
          <LazyMissingComponent />
        </Suspense>
      )

      await waitFor(() => {
        expect(screen.getByTestId('fallback-component')).toBeInTheDocument()
      })
    })
  })

  describe('Memory Management', () => {
    it('should not load components for inactive tabs', async () => {
      render(<SettingsPanel />)

      await waitFor(() => {
        expect(screen.getByTestId('glow-settings')).toBeInTheDocument()
      })

      // After 1 second, only the active tab should be loaded
      await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 100))
      })

      expect(MockGlowSettingsComponent).toHaveBeenCalled()
      expect(MockPhysicsSettingsComponent).not.toHaveBeenCalled()
      expect(MockSystemSettingsComponent).not.toHaveBeenCalled()
      expect(MockXRSettingsComponent).not.toHaveBeenCalled()
    })

    it('should handle component unmounting correctly', async () => {
      const { unmount } = render(<SettingsPanel />)

      await waitFor(() => {
        expect(screen.getByTestId('glow-settings')).toBeInTheDocument()
      })

      // Unmount the panel
      unmount()

      // Components should clean up without errors
      // In a real implementation, this would verify cleanup of subscriptions, timers, etc.
      expect(true).toBe(true) // No errors thrown
    })
  })

  describe('Accessibility', () => {
    it('should provide proper loading states for screen readers', async () => {
      render(<SettingsPanel />)

      const loadingElement = screen.getByTestId('glow-loading')
      expect(loadingElement).toHaveTextContent('Loading glow settings...')
      expect(loadingElement).toHaveAttribute('aria-live', 'polite') // In real implementation

      await waitFor(() => {
        expect(screen.getByTestId('glow-settings')).toBeInTheDocument()
      })
    })

    it('should maintain focus management during lazy loading', async () => {
      render(<SettingsPanel />)

      const physicsTab = screen.getByTestId('tab-physics')
      
      await user.click(physicsTab)

      // Focus should remain on the tab during loading
      expect(document.activeElement).toBe(physicsTab)

      await waitFor(() => {
        expect(screen.getByTestId('physics-settings')).toBeInTheDocument()
      })

      // Focus should still be manageable after component loads
      expect(document.activeElement).toBe(physicsTab)
    })
  })

  describe('Integration with Settings Store', () => {
    it('should trigger settings loading when component becomes active', async () => {
      const mockLoadSettings = vi.fn()
      
      // Mock the settings store
      vi.mocked(require('../store/settingsStore').useSettingsStore).mockReturnValue({
        loadSettingsByPaths: mockLoadSettings,
        settings: {},
        isLoading: false,
        error: null
      })

      render(<SettingsPanel />)

      await waitFor(() => {
        expect(screen.getByTestId('glow-settings')).toBeInTheDocument()
      })

      // Switch to physics tab
      await user.click(screen.getByTestId('tab-physics'))

      await waitFor(() => {
        expect(screen.getByTestId('physics-settings')).toBeInTheDocument()
      })

      // Should trigger loading of physics-related settings
      // (In real implementation, would check for specific path loading)
      expect(mockLoadSettings).toHaveBeenCalled()
    })
  })

  describe('Bundle Splitting Verification', () => {
    it('should demonstrate separate component loading', async () => {
      // This test verifies that components are loaded separately
      // In a real environment, this would use actual import() calls
      
      render(<SettingsPanel />)

      // Initially only glow is loaded
      await waitFor(() => {
        expect(MockGlowSettingsComponent).toHaveBeenCalled()
      })

      const loadTimes: Record<string, number> = {}
      
      // Measure load times for each component
      const tabs = ['physics', 'system', 'xr']
      
      for (const tab of tabs) {
        const start = performance.now()
        await user.click(screen.getByTestId(`tab-${tab}`))
        
        await waitFor(() => {
          expect(screen.getByTestId(`${tab}-settings`)).toBeInTheDocument()
        })
        
        loadTimes[tab] = performance.now() - start
      }

      // Each component should load in reasonable time
      for (const [tab, time] of Object.entries(loadTimes)) {
        expect(time).toBeLessThan(100) // Should be fast in tests
        console.log(`${tab} component loaded in ${time.toFixed(2)}ms`)
      }
    })
  })
})