import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/features/design-system/components/Card';

export const VisualisationPanel: React.FC = () => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Visualisation Settings</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-muted-foreground">Visualisation settings coming soon.</p>
      </CardContent>
    </Card>
  );
};
