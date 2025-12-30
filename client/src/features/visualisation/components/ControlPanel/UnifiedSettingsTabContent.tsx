/**
 * Unified Settings Tab Content
 *
 * Renders settings for a given tab with proper advanced mode and power user gating.
 * Integrates with the settingsStore for both local and server-side persistence.
 */

import React, { useCallback, useState, useMemo } from 'react';
import { useSettingsStore } from '../../../../store/settingsStore';
import { useControlPanelContext } from '../../../settings/components/control-panel-context';
import { UNIFIED_SETTINGS_CONFIG, filterSettingsFields } from './unifiedSettingsConfig';
import { nostrAuth } from '../../../../services/nostrAuthService';
import { webSocketService } from '../../../../services/WebSocketService';
import type { SettingField } from './types';
import { Lock, Info, RefreshCw } from 'lucide-react';

interface UnifiedSettingsTabContentProps {
  sectionId: string;
  onError?: (error: string) => void;
  onSuccess?: (message: string) => void;
}

export const UnifiedSettingsTabContent: React.FC<UnifiedSettingsTabContentProps> = ({
  sectionId,
  onError,
  onSuccess
}) => {
  const settings = useSettingsStore(state => state.settings);
  const updateSettings = useSettingsStore(state => state.updateSettings);
  const isPowerUser = useSettingsStore(state => state.isPowerUser);
  const user = useSettingsStore(state => state.user);
  const { advancedMode } = useControlPanelContext();

  const [nostrConnected, setNostrConnected] = useState(false);
  const [nostrPublicKey, setNostrPublicKey] = useState('');
  const [savingField, setSavingField] = useState<string | null>(null);

  // Get the section config
  const sectionConfig = UNIFIED_SETTINGS_CONFIG[sectionId];

  // Filter fields based on advanced mode and power user status
  const visibleFields = useMemo(() => {
    if (!sectionConfig) return [];
    return filterSettingsFields(sectionConfig.fields, advancedMode, isPowerUser);
  }, [sectionConfig, advancedMode, isPowerUser]);

  // Get value from nested path
  const getValueFromPath = useCallback((path: string): any => {
    const keys = path.split('.');
    let value: any = settings;
    for (const key of keys) {
      if (value === undefined || value === null) return undefined;
      value = value[key];
    }
    return value;
  }, [settings]);

  // Check if user can write (power user required for server persistence)
  const canWrite = useCallback((field: SettingField): boolean => {
    // Power user required for power-user-only fields
    if (field.isPowerUserOnly && !isPowerUser) return false;
    // For server persistence, need authenticated user
    return true; // Local settings always writable
  }, [isPowerUser]);

  // Update setting by path with validation and nostr gating
  const updateSettingByPath = useCallback(async (path: string, value: any, field: SettingField) => {
    // Check write permission for power user fields
    if (field.isPowerUserOnly && !isPowerUser) {
      onError?.('Power user authentication required to modify this setting');
      return;
    }

    setSavingField(field.key);

    try {
      const keys = path.split('.');

      await updateSettings((draft) => {
        let current: any = draft;
        for (let i = 0; i < keys.length - 1; i++) {
          if (!current[keys[i]]) {
            current[keys[i]] = {};
          }
          current = current[keys[i]];
        }
        current[keys[keys.length - 1]] = value;
      });

      // For power user only settings, also persist to server
      if (field.isPowerUserOnly && user) {
        // Server persistence would happen here via settingsApi
        // This is handled by updateSettings if persistSettingsOnServer is true
      }

      setSavingField(null);
    } catch (error) {
      setSavingField(null);
      onError?.(`Failed to update ${field.label}`);
    }
  }, [updateSettings, isPowerUser, user, onError]);

  // Nostr login handler
  const handleNostrLogin = async () => {
    try {
      const state = await nostrAuth.login();
      if (state.authenticated && state.user) {
        setNostrConnected(true);
        setNostrPublicKey(state.user.pubkey);
        onSuccess?.('Connected to Nostr');
      }
    } catch (error) {
      onError?.('Failed to connect to Nostr');
    }
  };

  // Nostr logout handler
  const handleNostrLogout = async () => {
    await nostrAuth.logout();
    setNostrConnected(false);
    setNostrPublicKey('');
    onSuccess?.('Disconnected from Nostr');
  };

  // Render a single field
  const renderField = (field: SettingField) => {
    const value = field.path ? getValueFromPath(field.path) : undefined;
    const isWritable = canWrite(field);
    const isSaving = savingField === field.key;

    const fieldStyle = {
      opacity: isWritable ? 1 : 0.5,
      pointerEvents: isWritable ? 'auto' : 'none'
    } as React.CSSProperties;

    switch (field.type) {
      case 'toggle':
        return (
          <div key={field.key} style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '4px 0',
            ...fieldStyle
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <label htmlFor={field.key} style={{
                fontSize: '10px',
                cursor: isWritable ? 'pointer' : 'not-allowed',
                color: 'white'
              }}>
                {field.label}
              </label>
              {field.description && (
                <div title={field.description} style={{ cursor: 'help' }}>
                  <Info size={10} style={{ color: 'rgba(255,255,255,0.4)' }} />
                </div>
              )}
              {field.isPowerUserOnly && (
                <Lock size={10} style={{ color: '#fbbf24' }} />
              )}
            </div>
            <button
              id={field.key}
              onClick={() => isWritable && field.path && updateSettingByPath(field.path, !value, field)}
              disabled={!isWritable || isSaving}
              style={{
                width: '36px',
                height: '18px',
                borderRadius: '9px',
                border: 'none',
                background: value ? '#10b981' : '#4b5563',
                position: 'relative',
                cursor: isWritable ? 'pointer' : 'not-allowed',
                transition: 'background 0.2s',
                flexShrink: 0,
                opacity: isSaving ? 0.6 : 1
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
          <div key={field.key} style={{ padding: '6px 0', ...fieldStyle }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: '4px'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <label htmlFor={field.key} style={{ fontSize: '10px', color: 'white' }}>
                  {field.label}
                </label>
                {field.description && (
                  <div title={field.description} style={{ cursor: 'help' }}>
                    <Info size={10} style={{ color: 'rgba(255,255,255,0.4)' }} />
                  </div>
                )}
                {field.isPowerUserOnly && (
                  <Lock size={10} style={{ color: '#fbbf24' }} />
                )}
              </div>
              <span style={{ fontSize: '9px', color: 'rgba(255,255,255,0.7)' }}>
                {typeof value === 'number'
                  ? value.toFixed(field.step && field.step < 0.01 ? 5 : field.step && field.step < 1 ? 2 : 0)
                  : '0'}
              </span>
            </div>
            <input
              type="range"
              id={field.key}
              value={Number(value) || 0}
              onChange={(e) => isWritable && field.path && updateSettingByPath(field.path, Number(e.target.value), field)}
              disabled={!isWritable || isSaving}
              min={field.min || 0}
              max={field.max || 100}
              step={field.step || 0.1}
              style={{
                width: '100%',
                height: '3px',
                borderRadius: '2px',
                background: 'rgba(255,255,255,0.2)',
                outline: 'none',
                cursor: isWritable ? 'pointer' : 'not-allowed'
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
            padding: '4px 0',
            ...fieldStyle
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <label htmlFor={field.key} style={{ fontSize: '10px', color: 'white' }}>
                {field.label}
              </label>
              {field.isPowerUserOnly && (
                <Lock size={10} style={{ color: '#fbbf24' }} />
              )}
            </div>
            <input
              id={field.key}
              type="color"
              value={value || '#ffffff'}
              onChange={(e) => isWritable && field.path && updateSettingByPath(field.path, e.target.value, field)}
              disabled={!isWritable || isSaving}
              style={{
                width: '36px',
                height: '20px',
                borderRadius: '3px',
                border: '1px solid rgba(255,255,255,0.2)',
                cursor: isWritable ? 'pointer' : 'not-allowed'
              }}
            />
          </div>
        );

      case 'select':
        return (
          <div key={field.key} style={{ padding: '4px 0', ...fieldStyle }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
              <label htmlFor={field.key} style={{ fontSize: '10px', color: 'white' }}>
                {field.label}
              </label>
              {field.isPowerUserOnly && (
                <Lock size={10} style={{ color: '#fbbf24' }} />
              )}
            </div>
            <select
              id={field.key}
              value={value || field.options?.[0] || ''}
              onChange={(e) => isWritable && field.path && updateSettingByPath(field.path, e.target.value, field)}
              disabled={!isWritable || isSaving}
              style={{
                width: '100%',
                background: 'rgba(255,255,255,0.05)',
                border: '1px solid rgba(255,255,255,0.15)',
                borderRadius: '3px',
                fontSize: '10px',
                color: 'white',
                padding: '4px 6px',
                cursor: isWritable ? 'pointer' : 'not-allowed'
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
            gap: '8px',
            ...fieldStyle
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <label htmlFor={field.key} style={{ fontSize: '10px', color: 'white', flexShrink: 0 }}>
                {field.label}
              </label>
              {field.isPowerUserOnly && (
                <Lock size={10} style={{ color: '#fbbf24' }} />
              )}
            </div>
            <input
              id={field.key}
              type="text"
              value={value || ''}
              onChange={(e) => isWritable && field.path && updateSettingByPath(field.path, e.target.value, field)}
              disabled={!isWritable || isSaving}
              style={{
                padding: '3px 6px',
                fontSize: '10px',
                background: 'rgba(255,255,255,0.05)',
                border: '1px solid rgba(255,255,255,0.15)',
                borderRadius: '3px',
                width: '120px',
                maxWidth: '120px',
                color: 'white',
                flexShrink: 0
              }}
            />
          </div>
        );

      case 'nostr-button':
        const isConnected = nostrConnected || nostrAuth.isAuthenticated();
        const pubKey = nostrPublicKey || nostrAuth.getCurrentUser()?.pubkey || '';

        return (
          <div key={field.key} style={{ padding: '6px 0' }}>
            <label style={{ fontSize: '10px', display: 'block', marginBottom: '6px', color: 'white' }}>
              {field.label}
            </label>
            {isConnected ? (
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
                  {pubKey.slice(0, 16)}...{pubKey.slice(-8)}
                </div>
                {isPowerUser && (
                  <div style={{
                    fontSize: '8px',
                    color: '#fbbf24',
                    padding: '3px 6px',
                    background: 'rgba(251,191,36,0.1)',
                    borderRadius: '3px',
                    textAlign: 'center'
                  }}>
                    Power User - Full access
                  </div>
                )}
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
                  Disconnect
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

      case 'action-button':
        const handleAction = () => {
          if (field.action === 'refresh_graph') {
            // Force refresh graph with current filter settings
            webSocketService.forceRefreshFilter();
            onSuccess?.('Graph refresh triggered - applying current filter settings');
          }
        };

        return (
          <div key={field.key} style={{ padding: '8px 0' }}>
            <button
              onClick={handleAction}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                background: 'linear-gradient(to right, #3b82f6, #2563eb)',
                color: 'white',
                padding: '8px 16px',
                borderRadius: '4px',
                fontSize: '11px',
                fontWeight: '600',
                border: 'none',
                cursor: 'pointer',
                transition: 'all 0.2s',
                boxShadow: '0 2px 4px rgba(59, 130, 246, 0.3)'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-1px)';
                e.currentTarget.style.boxShadow = '0 4px 8px rgba(59, 130, 246, 0.4)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 2px 4px rgba(59, 130, 246, 0.3)';
              }}
            >
              <RefreshCw size={14} />
              {field.label}
            </button>
            {field.description && (
              <p style={{
                fontSize: '9px',
                color: 'rgba(255,255,255,0.5)',
                marginTop: '4px',
                textAlign: 'center'
              }}>
                {field.description}
              </p>
            )}
          </div>
        );

      default:
        return null;
    }
  };

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

  // Check if entire section is hidden
  if (sectionConfig.isAdvanced && !advancedMode) {
    return null;
  }

  if (sectionConfig.isPowerUserOnly && !isPowerUser) {
    return (
      <div style={{
        textAlign: 'center',
        color: 'rgba(255,255,255,0.6)',
        padding: '32px 0'
      }}>
        <Lock size={24} style={{ marginBottom: '8px', color: '#fbbf24' }} />
        <p style={{ fontSize: '11px' }}>Power user authentication required</p>
        <button
          onClick={handleNostrLogin}
          style={{
            marginTop: '12px',
            background: 'linear-gradient(to right, #a855f7, #9333ea)',
            color: 'white',
            padding: '6px 16px',
            borderRadius: '4px',
            fontSize: '10px',
            fontWeight: '600',
            border: 'none',
            cursor: 'pointer'
          }}
        >
          Connect Nostr
        </button>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', padding: '4px' }}>
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
        margin: '0 -4px',
        paddingLeft: '4px',
        paddingRight: '4px',
        zIndex: 10,
        display: 'flex',
        alignItems: 'center',
        gap: '6px'
      }}>
        {sectionConfig.title}
        {sectionConfig.isPowerUserOnly && (
          <Lock size={10} style={{ color: '#fbbf24' }} />
        )}
        {!advancedMode && visibleFields.length < sectionConfig.fields.length && (
          <span style={{
            fontSize: '8px',
            color: 'rgba(255,255,255,0.4)',
            marginLeft: 'auto'
          }}>
            +{sectionConfig.fields.length - visibleFields.length} in advanced
          </span>
        )}
      </h3>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1px' }}>
        {visibleFields.map(renderField)}
      </div>
    </div>
  );
};
