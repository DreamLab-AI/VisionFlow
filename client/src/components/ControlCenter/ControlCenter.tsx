// frontend/src/components/ControlCenter/ControlCenter.tsx
// Main control center integrating all panels - REAL implementation

import React, { useState } from 'react';
import { SettingsPanel } from './SettingsPanel';
import { ConstraintPanel } from './ConstraintPanel';
import { ProfileManager } from './ProfileManager';
import { QualityGatePanel } from './QualityGatePanel';
import './ControlCenter.css';

type TabType = 'settings' | 'constraints' | 'profiles' | 'quality-gates';

export const ControlCenter: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('settings');
  const [notification, setNotification] = useState<{
    type: 'success' | 'error';
    message: string;
  } | null>(null);

  const showNotification = (type: 'success' | 'error', message: string) => {
    setNotification({ type, message });
    setTimeout(() => setNotification(null), 5000);
  };

  const handleError = (error: string) => {
    showNotification('error', error);
  };

  const handleSuccess = (message: string) => {
    showNotification('success', message);
  };

  return (
    <div className="control-center">
      <div className="control-center-header">
        <h1>VisionFlow Control Center</h1>
        <p>Real-time settings and constraint management</p>
      </div>

      {notification && (
        <div className={`notification ${notification.type}`}>
          {notification.message}
        </div>
      )}

      <div className="control-center-tabs">
        <button
          className={activeTab === 'settings' ? 'active' : ''}
          onClick={() => setActiveTab('settings')}
        >
          Settings
        </button>
        <button
          className={activeTab === 'constraints' ? 'active' : ''}
          onClick={() => setActiveTab('constraints')}
        >
          Constraints
        </button>
        <button
          className={activeTab === 'profiles' ? 'active' : ''}
          onClick={() => setActiveTab('profiles')}
        >
          Profiles
        </button>
        <button
          className={activeTab === 'quality-gates' ? 'active' : ''}
          onClick={() => setActiveTab('quality-gates')}
        >
          Quality Gates
        </button>
      </div>

      <div className="control-center-content">
        {activeTab === 'settings' && (
          <SettingsPanel onError={handleError} onSuccess={handleSuccess} />
        )}
        {activeTab === 'constraints' && (
          <ConstraintPanel onError={handleError} onSuccess={handleSuccess} />
        )}
        {activeTab === 'profiles' && (
          <ProfileManager onError={handleError} onSuccess={handleSuccess} />
        )}
        {activeTab === 'quality-gates' && (
          <QualityGatePanel onError={handleError} onSuccess={handleSuccess} />
        )}
      </div>
    </div>
  );
};
