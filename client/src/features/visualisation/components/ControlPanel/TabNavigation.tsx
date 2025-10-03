/**
 * Tab Navigation Component
 */

import React from 'react';
import { TabsList, TabsTrigger } from '../../../design-system/components/Tabs';
import type { TabConfig } from './types';

interface TabNavigationProps {
  tabs: TabConfig[];
  pressedButtons: string[];
  spacePilotConnected: boolean;
  rowClassName?: string;
}

export const TabNavigation: React.FC<TabNavigationProps> = ({
  tabs,
  pressedButtons,
  spacePilotConnected,
  rowClassName = '',
}) => {
  return (
    <TabsList style={{
      width: '100%',
      background: 'rgba(255,255,255,0.08)',
      border: '1px solid rgba(255,255,255,0.15)',
      borderRadius: '4px',
      padding: '2px',
      marginBottom: '6px',
      display: 'grid',
      gridTemplateColumns: 'repeat(4, 1fr)',
      gap: '2px'
    }}>
      {tabs.map((tab) => {
        const IconComponent = tab.icon;
        const isPressed = spacePilotConnected && tab.buttonKey && pressedButtons.includes(`[${tab.buttonKey}]`);

        return (
          <TabsTrigger
            key={tab.id}
            value={tab.id}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '2px',
              padding: '4px 2px',
              fontSize: '8px',
              fontWeight: '500',
              color: 'rgba(255,255,255,0.7)',
              border: isPressed ? '1px solid #22c55e' : '0',
              borderRadius: '3px',
              background: isPressed ? 'rgba(34,197,94,0.1)' : 'transparent',
              cursor: 'pointer',
              height: '100%',
              transition: 'all 0.2s'
            }}
          >
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
