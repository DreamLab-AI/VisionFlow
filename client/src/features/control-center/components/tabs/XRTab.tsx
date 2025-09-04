// XR/AR Tab - Quest 3, spatial computing, and immersive controls
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { Headphones, Hand, Eye, Compass } from 'lucide-react';

interface XRTabProps {
  searchQuery?: string;
}

export const XRTab: React.FC<XRTabProps> = ({ searchQuery = '' }) => {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Headphones className="w-5 h-5" />
            XR/AR Settings
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">
            Extended Reality settings for Quest 3, spatial computing, and immersive interactions.
          </p>
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="p-4 border rounded-lg">
              <Headphones className="w-8 h-8 text-blue-500 mb-2" />
              <h4 className="font-semibold">Quest 3 Settings</h4>
              <p className="text-sm text-muted-foreground">AR mode and hand tracking</p>
            </div>
            <div className="p-4 border rounded-lg">
              <Hand className="w-8 h-8 text-green-500 mb-2" />
              <h4 className="font-semibold">Immersive Controls</h4>
              <p className="text-sm text-muted-foreground">Movement and interaction settings</p>
            </div>
            <div className="p-4 border rounded-lg">
              <Compass className="w-8 h-8 text-purple-500 mb-2" />
              <h4 className="font-semibold">Spatial Computing</h4>
              <p className="text-sm text-muted-foreground">Anchor management and environment</p>
            </div>
            <div className="p-4 border rounded-lg">
              <Eye className="w-8 h-8 text-orange-500 mb-2" />
              <h4 className="font-semibold">Passthrough</h4>
              <p className="text-sm text-muted-foreground">AR passthrough settings</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default XRTab;