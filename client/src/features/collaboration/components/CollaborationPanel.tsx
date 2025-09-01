import React, { useState, useMemo } from 'react';
import { useSelectiveSetting, useSelectiveSettings, useSettingSetter } from '@/hooks/useSelectiveSettingsStore';
import { Card, CardHeader, CardTitle, CardContent } from '@/features/design-system/components/Card';
import { Button } from '@/features/design-system/components/Button';
import { Badge } from '@/features/design-system/components/Badge';
import { Avatar } from '@/features/design-system/components/Avatar';
import { ScrollArea } from '@/features/design-system/components/ScrollArea';
import { Users, UserPlus, MessageSquare, Video, Share2, Eye } from 'lucide-react';
import { createLogger } from '@/utils/logger';

const logger = createLogger('CollaborationPanel');

interface CollaborationPanelProps {
  className?: string;
}

interface CollaboratorUser {
  id: string;
  name: string;
  email: string;
  avatar?: string;
  status: 'online' | 'away' | 'offline';
  role: 'owner' | 'admin' | 'editor' | 'viewer';
  lastSeen: Date;
  currentView?: string;
}

export const CollaborationPanel: React.FC<CollaborationPanelProps> = ({ className }) => {
  const { set } = useSettingSetter();
  
  // Subscribe only to collaboration-related settings
  const collaborationSettings = useSelectiveSettings({
    enabled: 'collaboration.enabled',
    allowGuests: 'collaboration.allowGuests',
    shareByLink: 'collaboration.shareByLink',
    realTimeSync: 'collaboration.realTimeSync.enabled',
    showCursors: 'collaboration.cursors.enabled',
    notifications: 'collaboration.notifications.enabled',
    autoSave: 'collaboration.autoSave.enabled',
    conflictResolution: 'collaboration.conflicts.resolution'
  });
  
  // Mock collaborators data
  const collaborators: CollaboratorUser[] = useMemo(() => [
    {
      id: '1',
      name: 'Alice Johnson',
      email: 'alice@company.com',
      status: 'online',
      role: 'admin',
      lastSeen: new Date(),
      currentView: 'Graph View'
    },
    {
      id: '2',
      name: 'Bob Smith',
      email: 'bob@company.com',
      status: 'online',
      role: 'editor',
      lastSeen: new Date(Date.now() - 2 * 60 * 1000),
      currentView: 'Data Panel'
    },
    {
      id: '3',
      name: 'Carol Davis',
      email: 'carol@external.com',
      status: 'away',
      role: 'viewer',
      lastSeen: new Date(Date.now() - 15 * 60 * 1000)
    }
  ], []);
  
  const getStatusColor = (status: CollaboratorUser['status']) => {
    switch (status) {
      case 'online': return 'bg-green-500';
      case 'away': return 'bg-yellow-500';
      case 'offline': return 'bg-gray-400';
    }
  };
  
  const getRoleColor = (role: CollaboratorUser['role']) => {
    switch (role) {
      case 'owner': return 'bg-purple-100 text-purple-800';
      case 'admin': return 'bg-blue-100 text-blue-800';
      case 'editor': return 'bg-green-100 text-green-800';
      case 'viewer': return 'bg-gray-100 text-gray-800';
    }
  };
  
  if (!collaborationSettings.enabled) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users size={20} />
            Collaboration
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <Users size={48} className="mx-auto mb-4 text-gray-400" />
            <p className="text-muted-foreground">Collaboration is disabled</p>
            <p className="text-sm text-muted-foreground mt-2">
              Enable collaboration to work with team members in real-time
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Users size={20} />
            Collaboration
            <Badge className="bg-green-100 text-green-800">
              {collaborators.filter(u => u.status === 'online').length} online
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <Button size="sm" variant="outline">
              <UserPlus size={16} className="mr-1" />
              Invite
            </Button>
            <Button size="sm">
              <Share2 size={16} className="mr-1" />
              Share
            </Button>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Active Collaborators */}
        <div>
          <h3 className="font-medium mb-3">Active Collaborators</h3>
          <ScrollArea className="h-[200px]">
            <div className="space-y-2">
              {collaborators.map((user) => (
                <div key={user.id} className="flex items-center gap-3 p-2 rounded hover:bg-gray-50">
                  <div className="relative">
                    <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white text-sm font-medium">
                      {user.name.split(' ').map(n => n[0]).join('')}
                    </div>
                    <div className={`absolute -bottom-1 -right-1 w-3 h-3 rounded-full border-2 border-white ${getStatusColor(user.status)}`} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm truncate">{user.name}</span>
                      <Badge className={getRoleColor(user.role)} size="sm">
                        {user.role}
                      </Badge>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {user.currentView || user.email}
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    {user.status === 'online' && (
                      <Button size="sm" variant="ghost" className="h-6 w-6 p-0">
                        <MessageSquare size={12} />
                      </Button>
                    )}
                    <Button size="sm" variant="ghost" className="h-6 w-6 p-0">
                      <Eye size={12} />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        </div>
        
        {/* Quick Actions */}
        <div>
          <h3 className="font-medium mb-3">Quick Actions</h3>
          <div className="grid grid-cols-2 gap-2">
            <Button variant="outline" className="h-16 flex-col">
              <Video size={20} className="mb-1" />
              <span className="text-xs">Start Call</span>
            </Button>
            <Button variant="outline" className="h-16 flex-col">
              <MessageSquare size={20} className="mb-1" />
              <span className="text-xs">Group Chat</span>
            </Button>
            <Button variant="outline" className="h-16 flex-col">
              <Share2 size={20} className="mb-1" />
              <span className="text-xs">Share Screen</span>
            </Button>
            <Button variant="outline" className="h-16 flex-col">
              <UserPlus size={20} className="mb-1" />
              <span className="text-xs">Invite More</span>
            </Button>
          </div>
        </div>
        
        {/* Collaboration Settings */}
        <div>
          <h3 className="font-medium mb-3">Settings</h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm">Real-time Sync</span>
              <Button
                variant={collaborationSettings.realTimeSync ? 'default' : 'outline'}
                size="sm"
                onClick={() => set('collaboration.realTimeSync.enabled', !collaborationSettings.realTimeSync)}
              >
                {collaborationSettings.realTimeSync ? 'ON' : 'OFF'}
              </Button>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">Show Cursors</span>
              <Button
                variant={collaborationSettings.showCursors ? 'default' : 'outline'}
                size="sm"
                onClick={() => set('collaboration.cursors.enabled', !collaborationSettings.showCursors)}
              >
                {collaborationSettings.showCursors ? 'ON' : 'OFF'}
              </Button>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">Notifications</span>
              <Button
                variant={collaborationSettings.notifications ? 'default' : 'outline'}
                size="sm"
                onClick={() => set('collaboration.notifications.enabled', !collaborationSettings.notifications)}
              >
                {collaborationSettings.notifications ? 'ON' : 'OFF'}
              </Button>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default CollaborationPanel;