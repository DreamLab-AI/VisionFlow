import React, { useState, useMemo } from 'react';
import { useSelectiveSetting, useSelectiveSettings, useSettingSetter } from '@/hooks/useSelectiveSettingsStore';
import { Card, CardHeader, CardTitle, CardContent } from '@/features/design-system/components/Card';
import { Button } from '@/features/design-system/components/Button';
import { Badge } from '@/features/design-system/components/Badge';
import { ScrollArea } from '@/features/design-system/components/ScrollArea';
import { Bell, CheckCircle, AlertTriangle, Info, X, Settings } from 'lucide-react';
import { createLogger } from '@/utils/logger';

const logger = createLogger('NotificationCenter');

interface NotificationCenterProps {
  className?: string;
}

interface Notification {
  id: string;
  title: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
  timestamp: Date;
  read: boolean;
  actionable?: boolean;
  action?: string;
}

export const NotificationCenter: React.FC<NotificationCenterProps> = ({ className }) => {
  const { set } = useSettingSetter();
  
  // Subscribe only to notification settings
  const notificationSettings = useSelectiveSettings({
    enabled: 'notifications.enabled',
    showDesktop: 'notifications.desktop.enabled',
    showInApp: 'notifications.inApp.enabled',
    playSound: 'notifications.sound.enabled',
    groupSimilar: 'notifications.groupSimilar',
    retentionDays: 'notifications.retentionDays',
    types: {
      system: 'notifications.types.system',
      data: 'notifications.types.data',
      user: 'notifications.types.user',
      security: 'notifications.types.security'
    }
  });
  
  // Mock notifications
  const [notifications, setNotifications] = useState<Notification[]>([
    {
      id: '1',
      title: 'Data Export Complete',
      message: 'Your analytics data export (2.4MB) has been successfully generated and is ready for download.',
      type: 'success',
      timestamp: new Date(Date.now() - 5 * 60 * 1000),
      read: false,
      actionable: true,
      action: 'Download'
    },
    {
      id: '2',
      title: 'System Update Available',
      message: 'Version 2.1.4 is available with performance improvements and bug fixes.',
      type: 'info',
      timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000),
      read: false,
      actionable: true,
      action: 'Update Now'
    },
    {
      id: '3',
      title: 'High Memory Usage Detected',
      message: 'System memory usage has exceeded 85%. Consider closing unused applications or adding more RAM.',
      type: 'warning',
      timestamp: new Date(Date.now() - 3 * 60 * 60 * 1000),
      read: true
    },
    {
      id: '4',
      title: 'Bot Processing Failed',
      message: 'The data analysis bot encountered an error and stopped processing. Check logs for details.',
      type: 'error',
      timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000),
      read: true,
      actionable: true,
      action: 'View Logs'
    },
    {
      id: '5',
      title: 'New User Joined',
      message: 'Alice Johnson has joined the workspace as an Editor.',
      type: 'info',
      timestamp: new Date(Date.now() - 6 * 60 * 60 * 1000),
      read: true
    }
  ]);
  
  const unreadCount = notifications.filter(n => !n.read).length;
  
  const getTypeIcon = (type: Notification['type']) => {
    switch (type) {
      case 'success': return <CheckCircle size={20} className="text-green-600" />;
      case 'warning': return <AlertTriangle size={20} className="text-yellow-600" />;
      case 'error': return <X size={20} className="text-red-600" />;
      default: return <Info size={20} className="text-blue-600" />;
    }
  };
  
  const getTypeColor = (type: Notification['type']) => {
    switch (type) {
      case 'success': return 'border-l-green-500 bg-green-50';
      case 'warning': return 'border-l-yellow-500 bg-yellow-50';
      case 'error': return 'border-l-red-500 bg-red-50';
      default: return 'border-l-blue-500 bg-blue-50';
    }
  };
  
  const markAsRead = (id: string) => {
    setNotifications(prev => prev.map(n => 
      n.id === id ? { ...n, read: true } : n
    ));
  };
  
  const markAllAsRead = () => {
    setNotifications(prev => prev.map(n => ({ ...n, read: true })));
  };
  
  const removeNotification = (id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };
  
  const clearAll = () => {
    setNotifications([]);
  };
  
  const handleAction = (notification: Notification) => {
    logger.info('Notification action clicked', { 
      notificationId: notification.id, 
      action: notification.action 
    });
    markAsRead(notification.id);
  };
  
  const formatTimeAgo = (timestamp: Date) => {
    const diffMs = Date.now() - timestamp.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${diffDays}d ago`;
  };
  
  if (!notificationSettings.enabled) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Bell size={20} />
            Notifications
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <Bell size={48} className="mx-auto mb-4 text-gray-400" />
            <p className="text-muted-foreground">Notifications are disabled</p>
            <p className="text-sm text-muted-foreground mt-2">
              Enable notifications in settings to stay updated on system events
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
            <Bell size={20} />
            Notification Center
            {unreadCount > 0 && (
              <Badge className="bg-red-100 text-red-800">
                {unreadCount} unread
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-2">
            {unreadCount > 0 && (
              <Button size="sm" variant="outline" onClick={markAllAsRead}>
                Mark All Read
              </Button>
            )}
            <Button size="sm" variant="outline" onClick={clearAll}>
              Clear All
            </Button>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {notifications.length === 0 ? (
          <div className="text-center py-8">
            <Bell size={48} className="mx-auto mb-4 text-gray-400" />
            <p className="text-muted-foreground">No notifications</p>
            <p className="text-sm text-muted-foreground mt-2">
              You're all caught up! New notifications will appear here.
            </p>
          </div>
        ) : (
          <ScrollArea className="h-[500px]">
            <div className="space-y-3">
              {notifications.map((notification) => (
                <div
                  key={notification.id}
                  className={`border-l-4 p-4 rounded-r-lg ${
                    getTypeColor(notification.type)
                  } ${!notification.read ? 'border-r-2 border-r-blue-500' : ''}`}
                  onClick={() => !notification.read && markAsRead(notification.id)}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      {getTypeIcon(notification.type)}
                      <h3 className={`font-medium ${
                        !notification.read ? 'text-gray-900' : 'text-gray-700'
                      }`}>
                        {notification.title}
                      </h3>
                      {!notification.read && (
                        <div className="w-2 h-2 bg-blue-500 rounded-full" />
                      )}
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="text-xs text-muted-foreground">
                        {formatTimeAgo(notification.timestamp)}
                      </span>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={(e) => {
                          e.stopPropagation();
                          removeNotification(notification.id);
                        }}
                        className="h-6 w-6 p-0"
                      >
                        <X size={12} />
                      </Button>
                    </div>
                  </div>
                  
                  <p className="text-sm text-gray-600 mb-3">
                    {notification.message}
                  </p>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-muted-foreground">
                      {notification.timestamp.toLocaleString()}
                    </span>
                    {notification.actionable && notification.action && (
                      <Button
                        size="sm"
                        onClick={() => handleAction(notification)}
                      >
                        {notification.action}
                      </Button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        )}
        
        {/* Notification Settings */}
        <div className="mt-6 pt-4 border-t">
          <h3 className="font-medium mb-3 flex items-center gap-2">
            <Settings size={16} />
            Notification Settings
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="flex items-center justify-between">
              <span className="text-sm">Desktop Notifications</span>
              <Button
                variant={notificationSettings.showDesktop ? 'default' : 'outline'}
                size="sm"
                onClick={() => set('notifications.desktop.enabled', !notificationSettings.showDesktop)}
              >
                {notificationSettings.showDesktop ? 'ON' : 'OFF'}
              </Button>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">Sound Alerts</span>
              <Button
                variant={notificationSettings.playSound ? 'default' : 'outline'}
                size="sm"
                onClick={() => set('notifications.sound.enabled', !notificationSettings.playSound)}
              >
                {notificationSettings.playSound ? 'ON' : 'OFF'}
              </Button>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">Group Similar</span>
              <Button
                variant={notificationSettings.groupSimilar ? 'default' : 'outline'}
                size="sm"
                onClick={() => set('notifications.groupSimilar', !notificationSettings.groupSimilar)}
              >
                {notificationSettings.groupSimilar ? 'ON' : 'OFF'}
              </Button>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">Retention</span>
              <Badge variant="outline">{notificationSettings.retentionDays}d</Badge>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default NotificationCenter;