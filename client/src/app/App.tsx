import { useEffect, Component, ReactNode, useCallback, useState } from 'react'
import AppInitializer from './AppInitializer'
import { ApplicationModeProvider } from '../contexts/ApplicationModeContext';
import XRCoreProvider from '../features/xr/providers/XRCoreProvider';
import { useSettingsStore } from '../store/settingsStore';
import { createLogger, createErrorMetadata } from '../utils/logger';
import Quest3AR from './Quest3AR';
import MainLayout from './MainLayout';
import { useQuest3Integration } from '../hooks/useQuest3Integration';
import { CommandPalette } from '../features/command-palette/components/CommandPalette';
import { initializeCommandPalette } from '../features/command-palette/defaultCommands';
import { HelpProvider } from '../features/help/components/HelpProvider';
import { registerSettingsHelp } from '../features/help/settingsHelp';
import { OnboardingProvider } from '../features/onboarding/components/OnboardingProvider';
import { registerOnboardingCommands } from '../features/onboarding/flows/defaultFlows';
import { TooltipProvider } from '../features/design-system/components/Tooltip';
import { useBotsWebSocketIntegration } from '../features/bots/hooks/useBotsWebSocketIntegration';
import { DebugControlPanel } from '../components/DebugControlPanel';
import { ConnectionWarning } from '../components/ConnectionWarning';
import { useAutoBalanceNotifications } from '../hooks/useAutoBalanceNotifications';
const logger = createLogger('App')

// Error boundary component to catch rendering errors
interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
}

class ErrorBoundary extends Component<ErrorBoundaryProps, { hasError: boolean; error: Error | null; errorInfo: any }> {
  state = { hasError: false, error: null, errorInfo: null };

  static getDerivedStateFromError(error: any) {
    return { hasError: true, error };
  }

  componentDidCatch(error: any, errorInfo: any) {
    logger.error('React error boundary caught error:', {
      ...createErrorMetadata(error),
      component: errorInfo?.componentStack
        ? errorInfo.componentStack.split('\n')[1]?.trim()
        : 'Unknown component'
    });
    this.setState({ errorInfo });
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div className="p-4 bg-destructive text-destructive-foreground rounded-md">
          <h2 className="text-xl font-bold mb-2">Something went wrong</h2>
          <p className="mb-4">The application encountered an error. Try refreshing the page.</p>
          {process.env.NODE_ENV === 'development' && (
            <pre className="bg-muted p-2 rounded text-sm overflow-auto">
              {this.state.error
                ? (this.state.error.message || String(this.state.error))
                : 'No error details available'}
            </pre>
          )}
        </div>
      );
    }
    return this.props.children;
  }
}

function App() {
  const [initializationState, setInitializationState] = useState<'loading' | 'initialized' | 'error'>('loading');
  const [initializationError, setInitializationError] = useState<Error | null>(null);
  const initialized = useSettingsStore(state => state.initialized);

  const { shouldUseQuest3Layout, isQuest3Detected, autoStartSuccessful } = useQuest3Integration({
    enableAutoStart: false
  });

  // Initialize bots WebSocket integration
  const botsConnectionStatus = useBotsWebSocketIntegration();
  
  // Initialize auto-balance notifications polling
  useAutoBalanceNotifications();

  // Check if we should use Quest 3 AR mode (unified approach)
  const shouldUseQuest3AR = () => {
    const userAgent = navigator.userAgent.toLowerCase();
    const isQuest3Browser = userAgent.includes('quest 3') || 
                            userAgent.includes('meta quest 3') ||
                            (userAgent.includes('oculus') && userAgent.includes('quest'));
    
    // Check for force parameter
    const forceQuest3 = window.location.search.includes('force=quest3') ||
                        window.location.search.includes('directar=true');
    
    return (isQuest3Browser || forceQuest3 || shouldUseQuest3Layout) && initialized;
  };

  useEffect(() => {
    // Initialize command palette, help system, and onboarding on first load
    if (initialized) {
      initializeCommandPalette();
      registerSettingsHelp();
      registerOnboardingCommands();

      const hasVisited = localStorage.getItem('hasVisited');
      if (!hasVisited) {
        localStorage.setItem('hasVisited', 'true');
        setTimeout(() => {
          window.dispatchEvent(new CustomEvent('start-onboarding', {
            detail: { flowId: 'welcome' }
          }));
        }, 1000);
      }
    }
  }, [initialized])

  const handleInitialized = useCallback(() => {
    setInitializationState('initialized');
    const settings = useSettingsStore.getState().settings;
    const debugEnabled = settings?.system?.debug?.enabled === true;
    if (debugEnabled) {
      logger.debug('Application initialized');
      logger.debug('Bots WebSocket connection status:', botsConnectionStatus);
    }
  }, [botsConnectionStatus]);

  const handleInitializationError = useCallback((error: Error) => {
    setInitializationError(error);
    setInitializationState('error');
  }, []);

  const renderContent = () => {
    switch (initializationState) {
      case 'loading':
        return <div>Connecting to server...</div>;
      case 'error':
        return (
          <div>
            <h2>Error Initializing Application</h2>
            <p>{initializationError?.message || 'An unknown error occurred.'}</p>
            <button onClick={() => window.location.reload()}>Retry</button>
          </div>
        );
      case 'initialized':
        return shouldUseQuest3AR() ? <Quest3AR /> : <MainLayout />;
    }
  };

  return (
    <TooltipProvider delayDuration={300} skipDelayDuration={100}>
      <HelpProvider>
        <OnboardingProvider>
          <ErrorBoundary>
            <ApplicationModeProvider>
              <XRCoreProvider>
                {renderContent()}
                {initializationState === 'loading' && (
                  <AppInitializer onInitialized={handleInitialized} onError={handleInitializationError} />
                )}
                {initializationState === 'initialized' && (
                  <>
                    <ConnectionWarning />
                    <CommandPalette />
                    <DebugControlPanel />
                  </>
                )}
              </XRCoreProvider>
            </ApplicationModeProvider>
          </ErrorBoundary>
        </OnboardingProvider>
      </HelpProvider>
    </TooltipProvider>
  );
}

export default App
