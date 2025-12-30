import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/features/design-system/components/Card';

export const AIPanel: React.FC = () => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>AI Settings</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-muted-foreground">AI configuration coming soon.</p>
      </CardContent>
    </Card>
  );
};
