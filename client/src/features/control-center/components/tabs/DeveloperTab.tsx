// Developer Tab - Debug tools, API testing & experimental features
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { Code, Bug, TestTube, Beaker } from 'lucide-react';

interface DeveloperTabProps {
  searchQuery?: string;
}

export const DeveloperTab: React.FC<DeveloperTabProps> = ({ searchQuery = '' }) => {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Code className="w-5 h-5" />
            Developer Tools
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">
            Debug tools, API testing, and experimental features for power users and developers.
          </p>
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="p-4 border rounded-lg">
              <Bug className="w-8 h-8 text-red-500 mb-2" />
              <h4 className="font-semibold">Debug Tools</h4>
              <p className="text-sm text-muted-foreground">Console, inspector, and GPU debugger</p>
            </div>
            <div className="p-4 border rounded-lg">
              <TestTube className="w-8 h-8 text-blue-500 mb-2" />
              <h4 className="font-semibold">API Testing</h4>
              <p className="text-sm text-muted-foreground">Endpoint tester and WebSocket monitor</p>
            </div>
            <div className="p-4 border rounded-lg">
              <Beaker className="w-8 h-8 text-green-500 mb-2" />
              <h4 className="font-semibold">Experimental</h4>
              <p className="text-sm text-muted-foreground">Beta toggles and feature flags</p>
            </div>
            <div className="p-4 border rounded-lg">
              <Code className="w-8 h-8 text-purple-500 mb-2" />
              <h4 className="font-semibold">Advanced</h4>
              <p className="text-sm text-muted-foreground">A/B testing and custom scripts</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default DeveloperTab;