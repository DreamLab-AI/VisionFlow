/**
 * Tab Navigation Component
 *
 * Renders the tab bar with support for advanced mode filtering and power user badges.
 * Dynamically adjusts grid layout based on visible tab count.
 */

import React, { useMemo } from 'react';
import { TabsList, TabsTrigger } from '../../../design-system/components/Tabs';
import { Lock, Star } from 'lucide-react';
import type { TabConfig } from './types';

interface TabNavigationProps {
  tabs: TabConfig[];
  pressedButtons: string[];
  spacePilotConnected: boolean;
  rowClassName?: string;
  advancedMode?: boolean;
  isPowerUser?: boolean;
}

export const TabNavigation: React.FC<TabNavigationProps> = ({
  tabs,
  pressedButtons,
  spacePilotConnected,
  rowClassName = '',
  advancedMode = false,
  isPowerUser = false,
}) => {
  // Filter tabs based on advanced mode and power user status
  const visibleTabs = useMemo(() => {
    return tabs.filter(tab => {
      // Hide advanced tabs in basic mode
      if (tab.isAdvanced && !advancedMode) return false;
      // Hide power user tabs from non-power users
      if (tab.isPowerUserOnly && !isPowerUser) return false;
      return true;
    });
  }, [tabs, advancedMode, isPowerUser]);

  // Calculate grid columns based on tab count
  const gridColumns = useMemo(() => {
    const count = visibleTabs.length;
    if (count <= 4) return count;
    if (count <= 6) return 3;
    if (count <= 9) return Math.ceil(count / 2);
    return 4;
  }, [visibleTabs.length]);

  return (
    <TabsList style={{
      width: '100%',
      background: 'rgba(255,255,255,0.08)',
      border: '1px solid rgba(255,255,255,0.15)',
      borderRadius: '4px',
      padding: '2px',
      marginBottom: '6px',
      display: 'grid',
      gridTemplateColumns: `repeat(${gridColumns}, 1fr)`,
      gap: '2px',
      height: 'auto',
      minHeight: 'auto',
      position: 'relative',
      zIndex: 1
    }}>
      {visibleTabs.map((tab) => {
        const IconComponent = tab.icon;
        const isPressed = spacePilotConnected && tab.buttonKey && pressedButtons.includes(`[${tab.buttonKey}]`);
        const isAdvancedTab = tab.isAdvanced;
        const isPowerUserTab = tab.isPowerUserOnly;

        return (
          <TabsTrigger
            key={tab.id}
            value={tab.id}
            title={tab.description}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '2px',
              padding: '4px 2px',
              fontSize: '8px',
              fontWeight: '500',
              color: isAdvancedTab ? 'rgba(168,85,247,0.9)' : 'rgba(255,255,255,0.7)',
              border: isPressed
                ? '1px solid #22c55e'
                : isPowerUserTab
                  ? '1px solid rgba(251,191,36,0.3)'
                  : '0',
              borderRadius: '3px',
              background: isPressed
                ? 'rgba(34,197,94,0.1)'
                : isPowerUserTab
                  ? 'rgba(251,191,36,0.05)'
                  : 'transparent',
              cursor: 'pointer',
              height: '100%',
              transition: 'all 0.2s',
              position: 'relative'
            }}
          >
            {/* Power user indicator */}
            {isPowerUserTab && (
              <div style={{
                position: 'absolute',
                top: '2px',
                right: '2px'
              }}>
                <Star size={6} style={{ color: '#fbbf24', fill: '#fbbf24' }} />
              </div>
            )}

            {/* Advanced mode indicator */}
            {isAdvancedTab && !isPowerUserTab && (
              <div style={{
                position: 'absolute',
                top: '2px',
                right: '2px'
              }}>
                <Lock size={6} style={{ color: '#a855f7' }} />
              </div>
            )}

            {IconComponent && <IconComponent size={12} />}
            <div style={{ textAlign: 'center', lineHeight: '1.1' }}>
              {tab.buttonKey && (
                <div style={{ opacity: 0.6, fontSize: '7px' }}>{tab.buttonKey}</div>
              )}
              <div style={{ fontSize: '8px' }}>{tab.label}</div>
            </div>
          </TabsTrigger>
        );
      })}
    </TabsList>
  );
};
