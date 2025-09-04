// Data Management Tab - Import/export, streaming & persistence
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { Database, Upload, Download, Wifi } from 'lucide-react';

interface DataManagementTabProps {
  searchQuery?: string;
}

export const DataManagementTab: React.FC<DataManagementTabProps> = ({ searchQuery = '' }) => {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="w-5 h-5" />
            Data Management
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">
            Data import/export, streaming connections, and persistence settings.
          </p>
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="p-4 border rounded-lg">
              <Database className="w-8 h-8 text-blue-500 mb-2" />
              <h4 className="font-semibold">Graph Data</h4>
              <p className="text-sm text-muted-foreground">Import/export and data sources</p>
            </div>
            <div className="p-4 border rounded-lg">
              <Upload className="w-8 h-8 text-green-500 mb-2" />
              <h4 className="font-semibold">Persistence</h4>
              <p className="text-sm text-muted-foreground">Save/load states and presets</p>
            </div>
            <div className="p-4 border rounded-lg">
              <Wifi className="w-8 h-8 text-purple-500 mb-2" />
              <h4 className="font-semibold">Streaming</h4>
              <p className="text-sm text-muted-foreground">WebSocket and protocol settings</p>
            </div>
            <div className="p-4 border rounded-lg">
              <Download className="w-8 h-8 text-orange-500 mb-2" />
              <h4 className="font-semibold">Backup & Restore</h4>
              <p className="text-sm text-muted-foreground">Backup management</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default DataManagementTab;