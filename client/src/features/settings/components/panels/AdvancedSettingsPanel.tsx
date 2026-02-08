import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/features/design-system/components/Card';

export const AdvancedSettingsPanel: React.FC = () => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Advanced Settings</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-muted-foreground">Advanced settings (networking, security, performance tuning) are managed via settings.yaml on the server. A UI for these settings is planned for a future release.</p>
      </CardContent>
    </Card>
  );
};
