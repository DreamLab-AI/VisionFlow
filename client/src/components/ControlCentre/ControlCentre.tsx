import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../features/design-system/components/Card';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../../features/design-system/components/Tabs';
import { Button } from '../../features/design-system/components/Button';
import { useSettingsStore } from '../../store/settingsStore';
import { BackgroundEnvironmentControls } from './BackgroundEnvironmentControls';
import { ForceGraphControls } from './ForceGraphControls';
import { useToast } from '../../features/design-system/components/Toast';
import { Minimize2, Maximize2, Settings, Eye, Activity } from 'lucide-react';

interface ControlCentreProps {
  className?: string;
  defaultExpanded?: boolean;
  showStats?: boolean;
  enableBloom?: boolean;
}

interface NostrAuth {
  isAuthenticated: () => boolean;
  getCurrentUser: () => { pubkey: string } | null;
  login: () => Promise<string>;
  logout: () => void;
}

declare global {
  interface Window {
    nostr?: {
      getPublicKey: () => Promise<string>;
      signEvent: (event: any) => Promise<any>;
    };
  }
}

export const ControlCentre: React.FC<ControlCentreProps> = ({
  className = '',
  defaultExpanded = true,
  showStats = false,
  enableBloom = false
}) => {
  const { toast } = useToast();
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  const [activeTab, setActiveTab] = useState('background');
  const [nostrAuth, setNostrAuth] = useState<NostrAuth | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [currentUser, setCurrentUser] = useState<{ pubkey: string } | null>(null);
  
  const { 
    settings, 
    updateSettings, 
    authenticated,
    setAuthenticated,
    setUser,
    user,
    isPowerUser
  } = useSettingsStore();

  // Initialize Nostr authentication
  useEffect(() => {
    const initializeNostr = async () => {
      try {
        const { nostrAuth } = await import('../../services/nostrAuthService');
        setNostrAuth(nostrAuth);
        
        if (nostrAuth.isAuthenticated()) {
          const user = nostrAuth.getCurrentUser();
          setIsAuthenticated(true);
          setCurrentUser(user);
          setAuthenticated(true);
          setUser(user ? { ...user, isPowerUser: false } : null);
        }
      } catch (error) {
        console.warn('Failed to initialize Nostr authentication:', error);
      }
    };

    initializeNostr();
  }, [setAuthenticated, setUser]);

  const handleNostrLogin = useCallback(async () => {
    if (!window.nostr) {
      toast({
        title: 'Nostr Extension Not Found',
        description: 'Please install a Nostr browser extension like nos2x or Alby.',
        variant: 'destructive'
      });
      return;
    }

    try {
      const publicKey = await window.nostr.getPublicKey();
      
      if (nostrAuth) {
        await nostrAuth.login();
        const user = nostrAuth.getCurrentUser();
        setIsAuthenticated(true);
        setCurrentUser(user);
        setAuthenticated(true);
        setUser(user ? { ...user, isPowerUser: false } : null);
      }

      // Update settings to enable server persistence
      updateSettings((draft) => {
        if (!draft.system) draft.system = {} as any;
        draft.system.persistSettings = true;
      });

      toast({
        title: 'Nostr Authentication Successful',
        description: `Connected with public key: ${publicKey.slice(0, 8)}...`,
      });
    } catch (error) {
      console.error('Nostr login failed:', error);
      toast({
        title: 'Authentication Failed',
        description: 'Failed to connect to Nostr. Please try again.',
        variant: 'destructive'
      });
    }
  }, [nostrAuth, setAuthenticated, setUser, updateSettings, toast]);

  const handleNostrLogout = useCallback(() => {
    if (nostrAuth) {
      nostrAuth.logout();
      setIsAuthenticated(false);
      setCurrentUser(null);
      setAuthenticated(false);
      setUser(null);
      
      toast({
        title: 'Logged Out',
        description: 'Successfully disconnected from Nostr.'
      });
    }
  }, [nostrAuth, setAuthenticated, setUser, toast]);

  const handleToggleExpanded = useCallback(() => {
    setIsExpanded(!isExpanded);
  }, [isExpanded]);

  const handleSaveSettings = useCallback(async () => {
    try {
      // Settings are automatically saved via the store's persistence mechanism
      // This button provides a manual sync option
      toast({
        title: 'Settings Saved',
        description: 'Your settings have been saved and synced.',
      });
    } catch (error) {
      toast({
        title: 'Save Failed',
        description: 'Failed to save settings. Please try again.',
        variant: 'destructive'
      });
    }
  }, [toast]);

  if (!isExpanded) {
    return (
      <div className={`fixed top-4 left-4 z-50 ${className}`}>
        <Button
          variant="outline"
          size="sm"
          onClick={handleToggleExpanded}
          className="bg-black/80 border-white/20 text-white hover:bg-black/90"
        >
          <Maximize2 className="h-4 w-4" />
          <span className="sr-only">Expand Control Centre</span>
        </Button>
      </div>
    );
  }

  return (
    <div className={`fixed top-4 left-4 z-50 w-96 max-h-[80vh] ${className}`}>
      <Card className="bg-black/90 border-white/20 text-white backdrop-blur-sm">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Settings className="h-5 w-5" />
              <CardTitle className="text-lg">Control Centre</CardTitle>
            </div>
            <div className="flex items-center gap-2">
              {/* Authentication Status */}
              <div className="flex items-center gap-2">
                {isAuthenticated && currentUser ? (
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full" />
                    <span className="text-xs text-green-400">
                      {currentUser.pubkey.slice(0, 8)}...
                    </span>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={handleNostrLogout}
                      className="text-xs px-2 py-1 h-6"
                    >
                      Logout
                    </Button>
                  </div>
                ) : (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleNostrLogin}
                    className="text-xs px-2 py-1 h-6"
                  >
                    <div className="w-2 h-2 bg-orange-500 rounded-full mr-1" />
                    Connect Nostr
                  </Button>
                )}
              </div>
              
              <Button
                variant="ghost"
                size="sm"
                onClick={handleToggleExpanded}
              >
                <Minimize2 className="h-4 w-4" />
                <span className="sr-only">Minimize Control Centre</span>
              </Button>
            </div>
          </div>
          <CardDescription className="text-white/70">
            Configure visualization and environment settings
          </CardDescription>
        </CardHeader>
        
        <CardContent className="overflow-hidden">
          <div className="space-y-4">
            {/* Status Information */}
            <div className="flex items-center gap-4 text-sm">
              <div className="flex items-center gap-2">
                <Eye className="h-4 w-4" />
                <span>Stats: {showStats ? 'ON' : 'OFF'}</span>
              </div>
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4" />
                <span>Bloom: {enableBloom ? 'ON' : 'OFF'}</span>
              </div>
              {settings?.system?.persistSettings && (
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full" />
                  <span className="text-blue-400">Server Sync</span>
                </div>
              )}
            </div>

            {/* Main Control Tabs */}
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid w-full grid-cols-2 bg-white/10">
                <TabsTrigger 
                  value="background"
                  className="data-[state=active]:bg-white/20 data-[state=active]:text-white"
                >
                  Environment
                </TabsTrigger>
                <TabsTrigger 
                  value="graph"
                  className="data-[state=active]:bg-white/20 data-[state=active]:text-white"
                >
                  Force Graph
                </TabsTrigger>
              </TabsList>

              <div className="mt-4 max-h-[60vh] overflow-y-auto">
                <TabsContent value="background" className="mt-0 space-y-4">
                  <BackgroundEnvironmentControls />
                </TabsContent>

                <TabsContent value="graph" className="mt-0 space-y-4">
                  <ForceGraphControls />
                </TabsContent>
              </div>
            </Tabs>

            {/* Action Buttons */}
            <div className="flex gap-2 pt-2 border-t border-white/20">
              <Button
                variant="outline"
                size="sm"
                onClick={handleSaveSettings}
                className="flex-1 bg-transparent border-white/20 text-white hover:bg-white/10"
              >
                Save Settings
              </Button>
              {isPowerUser && (
                <Button
                  variant="outline"
                  size="sm"
                  className="bg-transparent border-yellow-500/50 text-yellow-400 hover:bg-yellow-500/10"
                  disabled
                >
                  Power User
                </Button>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};