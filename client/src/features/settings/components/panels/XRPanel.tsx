import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/features/design-system/components/Card';

export const XRPanel: React.FC = () => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>XR Settings</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-muted-foreground">Settings for XR/VR (Quest 3, WebXR, spatial audio) are managed via settings.yaml on the server. A UI for these settings is planned for a future release.</p>
      </CardContent>
    </Card>
  );
};
