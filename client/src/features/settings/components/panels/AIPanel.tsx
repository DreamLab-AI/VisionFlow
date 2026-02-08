import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/features/design-system/components/Card';

export const AIPanel: React.FC = () => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>AI Settings</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-muted-foreground">Settings for AI services (Perplexity, RAGFlow, OpenAI) are managed via settings.yaml on the server. A UI for these settings is planned for a future release.</p>
      </CardContent>
    </Card>
  );
};
