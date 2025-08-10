import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { Switch } from '@/features/design-system/components/Switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/features/design-system/components/Select';
import { Label } from '@/features/design-system/components/Label';
import { Button } from '@/features/design-system/components/Button';
import { Input } from '@/features/design-system/components/Input';
import { Shield, Key, LogIn, LogOut, User } from 'lucide-react';
import { useToast } from '@/features/design-system/components/Toast';
import { Badge } from '@/features/design-system/components/Badge';

interface AuthSettings {
  enabled: boolean;
  provider: string;
  required: boolean;
}

interface AuthStatus {
  authenticated: boolean;
  provider?: string;
  user?: {
    id: string;
    username?: string;
    pubkey?: string;
  };
}

export function AuthPanel() {
  const { toast } = useToast();
  const [authSettings, setAuthSettings] = useState<AuthSettings>({
    enabled: false,
    provider: 'nostr',
    required: false,
  });
  
  const [authStatus, setAuthStatus] = useState<AuthStatus>({
    authenticated: false,
  });
  
  const [nostrKey, setNostrKey] = useState('');
  const [loading, setLoading] = useState(false);

  // Load auth settings and status
  useEffect(() => {
    loadAuthSettings();
    checkAuthStatus();
  }, []);

  const loadAuthSettings = async () => {
    try {
      const response = await fetch('/api/settings');
      if (response.ok) {
        const settings = await response.json();
        if (settings.auth) {
          setAuthSettings(settings.auth);
        }
      }
    } catch (error) {
      console.error('Failed to load auth settings:', error);
    }
  };

  const checkAuthStatus = async () => {
    try {
      const response = await fetch('/api/auth/status');
      if (response.ok) {
        const status = await response.json();
        setAuthStatus(status);
      }
    } catch (error) {
      console.error('Failed to check auth status:', error);
    }
  };

  const handleToggleAuth = async (enabled: boolean) => {
    try {
      const response = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          auth: { ...authSettings, enabled }
        }),
      });
      
      if (response.ok) {
        setAuthSettings({ ...authSettings, enabled });
        toast({
          title: 'Authentication Settings Updated',
          description: `Authentication ${enabled ? 'enabled' : 'disabled'}`,
        });
      }
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to update authentication settings',
        variant: 'destructive',
      });
    }
  };

  const handleProviderChange = async (provider: string) => {
    try {
      const response = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          auth: { ...authSettings, provider }
        }),
      });
      
      if (response.ok) {
        setAuthSettings({ ...authSettings, provider });
        toast({
          title: 'Provider Updated',
          description: `Authentication provider set to ${provider}`,
        });
      }
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to update provider',
        variant: 'destructive',
      });
    }
  };

  const handleNostrLogin = async () => {
    if (!nostrKey) {
      toast({
        title: 'Error',
        description: 'Please enter your Nostr private key or nsec',
        variant: 'destructive',
      });
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('/api/auth/nostr/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ privateKey: nostrKey }),
      });
      
      if (response.ok) {
        const result = await response.json();
        setAuthStatus({
          authenticated: true,
          provider: 'nostr',
          user: result.user,
        });
        setNostrKey(''); // Clear the key
        toast({
          title: 'Login Successful',
          description: 'You are now authenticated with Nostr',
        });
      } else {
        throw new Error('Login failed');
      }
    } catch (error) {
      toast({
        title: 'Login Failed',
        description: 'Invalid key or authentication error',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/auth/logout', {
        method: 'POST',
      });
      
      if (response.ok) {
        setAuthStatus({ authenticated: false });
        toast({
          title: 'Logged Out',
          description: 'You have been logged out successfully',
        });
      }
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to logout',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* Authentication Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Authentication Status
          </CardTitle>
          <CardDescription>
            Current authentication state and user information
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Status</span>
              <Badge variant={authStatus.authenticated ? 'default' : 'secondary'}>
                {authStatus.authenticated ? 'Authenticated' : 'Not Authenticated'}
              </Badge>
            </div>
            
            {authStatus.authenticated && authStatus.user && (
              <>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Provider</span>
                  <Badge variant="outline">{authStatus.provider}</Badge>
                </div>
                
                {authStatus.user.username && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Username</span>
                    <span className="text-sm text-muted-foreground">{authStatus.user.username}</span>
                  </div>
                )}
                
                {authStatus.user.pubkey && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Public Key</span>
                    <span className="text-xs text-muted-foreground font-mono">
                      {authStatus.user.pubkey.slice(0, 8)}...{authStatus.user.pubkey.slice(-8)}
                    </span>
                  </div>
                )}
                
                <Button 
                  onClick={handleLogout} 
                  variant="outline" 
                  className="w-full"
                  disabled={loading}
                >
                  <LogOut className="mr-2 h-4 w-4" />
                  Logout
                </Button>
              </>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Authentication Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Key className="h-5 w-5" />
            Authentication Settings
          </CardTitle>
          <CardDescription>
            Configure authentication requirements and providers
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <Label htmlFor="auth-enabled">Enable Authentication</Label>
            <Switch
              id="auth-enabled"
              checked={authSettings.enabled}
              onCheckedChange={handleToggleAuth}
            />
          </div>
          
          {authSettings.enabled && (
            <>
              <div className="space-y-2">
                <Label htmlFor="auth-provider">Provider</Label>
                <Select value={authSettings.provider} onValueChange={handleProviderChange}>
                  <SelectTrigger id="auth-provider">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="nostr">Nostr</SelectItem>
                    <SelectItem value="oauth" disabled>OAuth (Coming Soon)</SelectItem>
                    <SelectItem value="webauthn" disabled>WebAuthn (Coming Soon)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="auth-required">Require Authentication</Label>
                  <p className="text-xs text-muted-foreground">
                    Block access to protected resources
                  </p>
                </div>
                <Switch
                  id="auth-required"
                  checked={authSettings.required}
                  onCheckedChange={async (required) => {
                    const response = await fetch('/api/settings', {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({
                        auth: { ...authSettings, required }
                      }),
                    });
                    if (response.ok) {
                      setAuthSettings({ ...authSettings, required });
                    }
                  }}
                />
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {/* Nostr Login */}
      {authSettings.enabled && authSettings.provider === 'nostr' && !authStatus.authenticated && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <LogIn className="h-5 w-5" />
              Nostr Login
            </CardTitle>
            <CardDescription>
              Sign in with your Nostr private key or nsec
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="nostr-key">Private Key (hex or nsec)</Label>
              <Input
                id="nostr-key"
                type="password"
                placeholder="Enter your private key..."
                value={nostrKey}
                onChange={(e) => setNostrKey(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleNostrLogin();
                }}
              />
              <p className="text-xs text-muted-foreground">
                Your key is never stored and is only used for signing
              </p>
            </div>
            
            <Button 
              onClick={handleNostrLogin} 
              className="w-full"
              disabled={loading || !nostrKey}
            >
              <User className="mr-2 h-4 w-4" />
              Sign In with Nostr
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
}