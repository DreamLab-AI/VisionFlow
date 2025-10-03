/**
 * Settings Tab Content Component
 * Renders settings fields based on configuration
 */

import React, { useCallback, useState } from 'react';
import { useSettingsStore } from '../../../../store/settingsStore';
import { SETTINGS_CONFIG } from './settingsConfig';
import type { SettingField } from './types';

interface SettingsTabContentProps {
  sectionId: string;
}

export const SettingsTabContent: React.FC<SettingsTabContentProps> = ({ sectionId }) => {
  const settings = useSettingsStore(state => state.settings);
  const updateSettings = useSettingsStore(state => state.updateSettings);
  const [nostrConnected, setNostrConnected] = useState(false);
  const [nostrPublicKey, setNostrPublicKey] = useState('');

  // Get value from settings path
  const getValueFromPath = useCallback((path: string): any => {
    const keys = path.split('.');
    let value = settings;
    for (const key of keys) {
      value = value?.[key];
    }
    return value;
  }, [settings]);

  // Update setting by path
  const updateSettingByPath = useCallback(async (path: string, value: any) => {
    const keys = path.split('.');
    const updates: any = {};

    // Build nested update object
    let current = updates;
    for (let i = 0; i < keys.length - 1; i++) {
      current[keys[i]] = {};
      current = current[keys[i]];
    }
    current[keys[keys.length - 1]] = value;

    await updateSettings(updates);
  }, [updateSettings]);

  // Handle Nostr login
  const handleNostrLogin = async () => {
    if (typeof window.nostr === 'undefined') {
      alert('No Nostr extension found! Please install a Nostr extension like nos2x or Alby.');
      return;
    }

    try {
      const publicKey = await window.nostr.getPublicKey();
      setNostrConnected(true);
      setNostrPublicKey(publicKey);

      await updateSettingByPath('auth.nostr.connected', true);
      await updateSettingByPath('auth.nostr.publicKey', publicKey);
    } catch (error) {
      alert('Failed to connect to Nostr extension. Please check your Nostr extension is installed and enabled.');
    }
  };

  // Handle Nostr logout
  const handleNostrLogout = async () => {
    setNostrConnected(false);
    setNostrPublicKey('');

    await updateSettingByPath('auth.nostr.connected', false);
    await updateSettingByPath('auth.nostr.publicKey', '');
  };

  // Render field based on type
  const renderField = (field: SettingField) => {
    const value = getValueFromPath(field.path);

    switch (field.type) {
      case 'toggle':
        return (
          <div key={field.key} style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '4px 0'
          }}>
            <label htmlFor={field.key} style={{
              fontSize: '10px',
              cursor: 'pointer',
              color: 'white'
            }}>
              {field.label}
            </label>
            <button
              id={field.key}
              onClick={() => updateSettingByPath(field.path, !value)}
              style={{
                width: '36px',
                height: '18px',
                borderRadius: '9px',
                border: 'none',
                background: value ? '#10b981' : '#4b5563',
                position: 'relative',
                cursor: 'pointer',
                transition: 'background 0.2s',
                flexShrink: 0
              }}
            >
              <div style={{
                width: '14px',
                height: '14px',
                borderRadius: '50%',
                background: 'white',
                position: 'absolute',
                top: '2px',
                left: value ? '20px' : '2px',
                transition: 'left 0.2s'
              }} />
            </button>
          </div>
        );

      case 'slider':
        return (
          <div key={field.key} style={{ padding: '6px 0' }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: '4px'
            }}>
              <label htmlFor={field.key} style={{ fontSize: '10px', color: 'white' }}>
                {field.label}
              </label>
              <span style={{ fontSize: '9px', color: 'rgba(255,255,255,0.7)' }}>
                {typeof value === 'number' ? value.toFixed(field.step && field.step < 0.01 ? 5 : 2) : '0.00'}
              </span>
            </div>
            <input
              type="range"
              id={field.key}
              value={Number(value) || 0}
              onChange={(e) => updateSettingByPath(field.path, Number(e.target.value))}
              min={field.min || 0}
              max={field.max || 100}
              step={field.step || 0.1}
              style={{
                width: '100%',
                height: '3px',
                borderRadius: '2px',
                background: 'rgba(255,255,255,0.2)',
                outline: 'none',
                cursor: 'pointer'
              }}
            />
          </div>
        );

      case 'color':
        return (
          <div key={field.key} style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '4px 0'
          }}>
            <label htmlFor={field.key} style={{ fontSize: '10px', color: 'white' }}>
              {field.label}
            </label>
            <input
              id={field.key}
              type="color"
              value={value || '#ffffff'}
              onChange={(e) => updateSettingByPath(field.path, e.target.value)}
              style={{
                width: '36px',
                height: '20px',
                borderRadius: '3px',
                border: '1px solid rgba(255,255,255,0.2)',
                cursor: 'pointer'
              }}
            />
          </div>
        );

      case 'select':
        return (
          <div key={field.key} style={{ padding: '4px 0' }}>
            <label htmlFor={field.key} style={{
              fontSize: '10px',
              display: 'block',
              marginBottom: '4px',
              color: 'white'
            }}>
              {field.label}
            </label>
            <select
              id={field.key}
              value={value || field.options?.[0] || ''}
              onChange={(e) => updateSettingByPath(field.path, e.target.value)}
              style={{
                width: '100%',
                background: 'rgba(255,255,255,0.05)',
                border: '1px solid rgba(255,255,255,0.15)',
                borderRadius: '3px',
                fontSize: '10px',
                color: 'white',
                padding: '4px 6px',
                cursor: 'pointer'
              }}
            >
              {field.options?.map((option) => (
                <option key={option} value={option} style={{ background: '#1f2937', color: 'white' }}>
                  {option}
                </option>
              ))}
            </select>
          </div>
        );

      case 'text':
        return (
          <div key={field.key} style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '4px 0',
            gap: '8px'
          }}>
            <label htmlFor={field.key} style={{ fontSize: '10px', color: 'white', flexShrink: 0 }}>
              {field.label}
            </label>
            <input
              id={field.key}
              type="text"
              value={value || ''}
              onChange={(e) => updateSettingByPath(field.path, e.target.value)}
              style={{
                padding: '3px 6px',
                fontSize: '10px',
                background: 'rgba(255,255,255,0.05)',
                border: '1px solid rgba(255,255,255,0.15)',
                borderRadius: '3px',
                width: '80px',
                maxWidth: '80px',
                color: 'white',
                flexShrink: 0
              }}
            />
          </div>
        );

      case 'nostr-button':
        return (
          <div key={field.key} style={{ padding: '6px 0' }}>
            <label style={{ fontSize: '10px', display: 'block', marginBottom: '6px', color: 'white' }}>
              {field.label}
            </label>
            {nostrConnected || getValueFromPath('auth.nostr.connected') ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <div style={{
                  fontSize: '9px',
                  color: '#4ade80',
                  wordBreak: 'break-all',
                  padding: '6px',
                  background: 'rgba(34,197,94,0.1)',
                  borderRadius: '3px',
                  border: '1px solid rgba(34,197,94,0.3)'
                }}>
                  {nostrPublicKey || getValueFromPath('auth.nostr.publicKey')}
                </div>
                <button
                  onClick={handleNostrLogout}
                  style={{
                    width: '100%',
                    background: 'linear-gradient(to right, #ef4444, #dc2626)',
                    color: 'white',
                    padding: '4px 10px',
                    borderRadius: '3px',
                    fontSize: '10px',
                    fontWeight: '600',
                    border: 'none',
                    cursor: 'pointer',
                    transition: 'all 0.2s'
                  }}
                >
                  Logout
                </button>
              </div>
            ) : (
              <button
                onClick={handleNostrLogin}
                style={{
                  width: '100%',
                  background: 'linear-gradient(to right, #a855f7, #9333ea)',
                  color: 'white',
                  padding: '4px 10px',
                  borderRadius: '3px',
                  fontSize: '10px',
                  fontWeight: '600',
                  border: 'none',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
              >
                Connect Nostr
              </button>
            )}
          </div>
        );

      default:
        return null;
    }
  };

  const sectionConfig = SETTINGS_CONFIG[sectionId];

  if (!sectionConfig) {
    return (
      <div style={{
        textAlign: 'center',
        color: 'rgba(255,255,255,0.6)',
        padding: '32px 0'
      }}>
        <p style={{ fontSize: '11px' }}>No settings available for this section</p>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
      <h3 style={{
        fontSize: '11px',
        fontWeight: '600',
        marginBottom: '6px',
        color: '#fbbf24',
        position: 'sticky',
        top: 0,
        background: 'rgba(0,0,0,0.5)',
        backdropFilter: 'blur(4px)',
        padding: '4px 0',
        margin: '0 -8px',
        paddingLeft: '8px',
        paddingRight: '8px',
        zIndex: 10
      }}>
        {sectionConfig.title}
      </h3>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1px' }}>
        {sectionConfig.fields.map(renderField)}
      </div>
    </div>
  );
};
