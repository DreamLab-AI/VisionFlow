// frontend/src/components/ControlCenter/ProfileManager.tsx
// Profile management with REAL database operations

import React, { useState, useEffect } from 'react';
import {
  settingsApi,
  SettingsProfile,
  SaveProfileRequest,
} from '../../api/settingsApi';
import './ProfileManager.css';

interface ProfileManagerProps {
  onError?: (error: string) => void;
  onSuccess?: (message: string) => void;
}

export const ProfileManager: React.FC<ProfileManagerProps> = ({
  onError,
  onSuccess,
}) => {
  const [profiles, setProfiles] = useState<SettingsProfile[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [newProfileName, setNewProfileName] = useState('');
  const [showSaveDialog, setShowSaveDialog] = useState(false);

  useEffect(() => {
    loadProfiles();
  }, []);

  const loadProfiles = async () => {
    setLoading(true);
    try {
      const response = await settingsApi.listProfiles();
      setProfiles(response.data);
      if (onSuccess) {
        onSuccess(`Loaded ${response.data.length} profiles`);
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to load profiles';
      console.error('Failed to load profiles:', err);
      if (onError) {
        onError(message);
      }
    } finally {
      setLoading(false);
    }
  };

  const saveCurrentProfile = async () => {
    if (!newProfileName.trim()) {
      if (onError) {
        onError('Please enter a profile name');
      }
      return;
    }

    setSaving(true);
    try {
      const request: SaveProfileRequest = { name: newProfileName.trim() };
      const response = await settingsApi.saveProfile(request);
      if (onSuccess) {
        onSuccess(`Profile "${newProfileName}" saved with ID ${response.data.id}`);
      }
      setNewProfileName('');
      setShowSaveDialog(false);
      await loadProfiles();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to save profile';
      console.error('Failed to save profile:', err);
      if (onError) {
        onError(message);
      }
    } finally {
      setSaving(false);
    }
  };

  const loadProfile = async (id: number) => {
    setSaving(true);
    try {
      const response = await settingsApi.loadProfile(id);
      if (onSuccess) {
        onSuccess(`Profile loaded successfully`);
      }
      // Note: Settings would be applied through the settings API
      // This would trigger a refresh in the SettingsPanel
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to load profile';
      console.error('Failed to load profile:', err);
      if (onError) {
        onError(message);
      }
    } finally {
      setSaving(false);
    }
  };

  const deleteProfile = async (id: number, name: string) => {
    if (
      !window.confirm(
        `Are you sure you want to delete profile "${name}"? This cannot be undone.`
      )
    ) {
      return;
    }

    setSaving(true);
    try {
      await settingsApi.deleteProfile(id);
      if (onSuccess) {
        onSuccess(`Profile "${name}" deleted`);
      }
      await loadProfiles();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to delete profile';
      console.error('Failed to delete profile:', err);
      if (onError) {
        onError(message);
      }
    } finally {
      setSaving(false);
    }
  };

  const formatDate = (dateStr: string): string => {
    try {
      const date = new Date(dateStr);
      return date.toLocaleString();
    } catch {
      return dateStr;
    }
  };

  if (loading) {
    return <div className="profile-manager loading">Loading profiles...</div>;
  }

  return (
    <div className="profile-manager">
      <div className="profile-header">
        <h2>Settings Profiles</h2>
        <div className="header-actions">
          <button onClick={loadProfiles} disabled={loading || saving}>
            Refresh
          </button>
          <button
            onClick={() => setShowSaveDialog(true)}
            disabled={saving}
            className="primary"
          >
            Save Current Settings as Profile
          </button>
        </div>
      </div>

      {showSaveDialog && (
        <div className="save-dialog">
          <h3>Save Current Settings</h3>
          <div className="form-group">
            <label>
              Profile Name:
              <input
                type="text"
                value={newProfileName}
                onChange={(e) => setNewProfileName(e.target.value)}
                placeholder="e.g., High Performance, Minimal Constraints"
                autoFocus
                disabled={saving}
              />
            </label>
          </div>
          <div className="dialog-actions">
            <button onClick={saveCurrentProfile} disabled={saving}>
              {saving ? 'Saving...' : 'Save Profile'}
            </button>
            <button
              onClick={() => {
                setShowSaveDialog(false);
                setNewProfileName('');
              }}
              disabled={saving}
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      <div className="profile-list">
        <h3>Saved Profiles ({profiles.length})</h3>
        {profiles.length === 0 ? (
          <div className="empty-state">
            No profiles saved yet. Click "Save Current Settings as Profile" to
            create your first profile.
          </div>
        ) : (
          <div className="profiles-grid">
            {profiles.map((profile) => (
              <div key={profile.id} className="profile-card">
                <div className="profile-name">{profile.name}</div>
                <div className="profile-meta">
                  <small>ID: {profile.id}</small>
                  <small>Created: {formatDate(profile.createdAt)}</small>
                  <small>Updated: {formatDate(profile.updatedAt)}</small>
                </div>
                <div className="profile-actions">
                  <button
                    onClick={() => loadProfile(profile.id)}
                    disabled={saving}
                  >
                    Load
                  </button>
                  <button
                    onClick={() => deleteProfile(profile.id, profile.name)}
                    disabled={saving}
                    className="danger"
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="profile-help">
        <h4>About Profiles:</h4>
        <p>
          Profiles save your current physics, constraint, and rendering settings
          as named configurations. You can switch between profiles to quickly
          apply different visualization styles.
        </p>
        <ul>
          <li>Click "Save Current Settings" to create a new profile</li>
          <li>Click "Load" to apply a saved profile's settings</li>
          <li>Click "Delete" to remove a profile permanently</li>
        </ul>
      </div>
    </div>
  );
};
