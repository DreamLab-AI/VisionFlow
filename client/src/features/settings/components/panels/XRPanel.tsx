import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/features/design-system/components/Card';

export const XRPanel: React.FC = () => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>XR Settings</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-muted-foreground">XR/VR settings coming soon.</p>
      </CardContent>
    </Card>
  );
};
