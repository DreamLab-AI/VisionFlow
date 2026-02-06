import React, { useState, useEffect } from 'react';
import { useSettingsStore } from '@/store/settingsStore';
import { Button } from '@/features/design-system/components/Button';
import { Input } from '@/features/design-system/components/Input';
import { Label } from '@/features/design-system/components/Label';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { webSocketService } from '@/store/websocketStore';
import { createLogger } from '@/utils/loggerConfig';

const logger = createLogger('BackendUrlSetting');

export function BackendUrlSetting() {
  const { get: getSetting, set: setSetting } = useSettingsStore();
  const [backendUrl, setBackendUrl] = useState<string>('');
  const [isConnected, setIsConnected] = useState<boolean>(false);


  useEffect(() => {
    const storedUrl = getSetting('system.customBackendUrl') as string;
    setBackendUrl(storedUrl || '');


    setIsConnected(webSocketService.isReady());


    const unsubscribe = webSocketService.onConnectionStatusChange((connected) => {
      setIsConnected(connected);
    });

    return () => {
      unsubscribe();
    };
  }, [getSetting]);

  const handleSave = () => {

    setSetting('system.customBackendUrl', backendUrl);


    webSocketService.setCustomBackendUrl(backendUrl || null);

    logger.info(`Backend URL set to: ${backendUrl || 'default'}`);
  };

  const handleReset = () => {
    setBackendUrl('');
    setSetting('system.customBackendUrl', '');


    webSocketService.setCustomBackendUrl(null);
    
    logger.info('Backend URL reset to default');
  };
  
  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Backend Connection</CardTitle>
        <CardDescription>
          Configure the connection to the backend server
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid gap-4">
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
          </div>
          
          <div className="grid gap-2">
            <Label htmlFor="backendUrl">Backend URL</Label>
            <Input
              id="backendUrl"
              placeholder="e.g., http://192.168.0.51:8000"
              value={backendUrl}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setBackendUrl(e.target.value)}
            />
            <p className="text-sm text-muted-foreground">
              Leave empty to use the default backend URL. Changes require reconnection.
            </p>
          </div>
        </div>
      </CardContent>
      <CardFooter className="flex justify-between">
        <Button variant="outline" onClick={handleReset}>
          Reset to Default
        </Button>
        <Button onClick={handleSave}>
          Save & Reconnect
        </Button>
      </CardFooter>
    </Card>
  );
}

export default BackendUrlSetting;
