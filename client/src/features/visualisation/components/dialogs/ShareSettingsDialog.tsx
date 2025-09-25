/**
 * Share Settings Dialog Component
 * Configure sharing options for graphs
 */

import React, { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/features/design-system/components/Dialog';
import { Button } from '@/features/design-system/components/Button';
import { Label } from '@/features/design-system/components/Label';
import { Input } from '@/features/design-system/components/Input';
import { Textarea } from '@/features/design-system/components/Textarea';
import { Switch } from '@/features/design-system/components/Switch';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/features/design-system/components/Select';
import { Card, CardContent, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { Badge } from '@/features/design-system/components/Badge';
import { Separator } from '@/features/design-system/components/Separator';
import {
  Globe,
  Lock,
  Clock,
  Shield,
  Download,
  MessageCircle,
  Edit3,
  AlertTriangle
} from 'lucide-react';
import { ShareOptions } from '@/api/exportApi';

interface ShareSettingsDialogProps {
  open: boolean;
  onClose: () => void;
  onShare: (options: ShareOptions) => void;
  isLoading?: boolean;
}

export const ShareSettingsDialog: React.FC<ShareSettingsDialogProps> = ({
  open,
  onClose,
  onShare,
  isLoading = false
}) => {
  const [description, setDescription] = useState('');
  const [expiry, setExpiry] = useState('7days');
  const [password, setPassword] = useState('');
  const [usePassword, setUsePassword] = useState(false);
  const [allowDownload, setAllowDownload] = useState(true);
  const [allowComment, setAllowComment] = useState(false);
  const [allowEdit, setAllowEdit] = useState(false);

  const handleSubmit = () => {
    const options: ShareOptions = {
      description: description.trim() || undefined,
      expiry,
      password: usePassword ? password : undefined,
      permissions: {
        allowDownload,
        allowComment,
        allowEdit
      }
    };

    onShare(options);
  };

  const resetForm = () => {
    setDescription('');
    setExpiry('7days');
    setPassword('');
    setUsePassword(false);
    setAllowDownload(true);
    setAllowComment(false);
    setAllowEdit(false);
  };

  const handleClose = () => {
    resetForm();
    onClose();
  };

  const expiryOptions = [
    { value: '1hour', label: '1 Hour', icon: Clock },
    { value: '1day', label: '1 Day', icon: Clock },
    { value: '7days', label: '7 Days', icon: Clock },
    { value: '30days', label: '30 Days', icon: Clock },
    { value: 'never', label: 'Never', icon: Globe }
  ];

  const isSecure = usePassword || expiry !== 'never';

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Globe className="h-5 w-5" />
            Share Graph
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          {/* Security Indicator */}
          <Card className={`border ${isSecure ? 'border-green-200 bg-green-50' : 'border-yellow-200 bg-yellow-50'}`}>
            <CardContent className="pt-3 pb-2">
              <div className="flex items-center gap-2">
                {isSecure ? (
                  <Shield className="h-4 w-4 text-green-600" />
                ) : (
                  <AlertTriangle className="h-4 w-4 text-yellow-600" />
                )}
                <span className="text-sm font-medium">
                  {isSecure ? 'Secure Share' : 'Public Share'}
                </span>
                <Badge variant={isSecure ? 'default' : 'secondary'} className="ml-auto">
                  {isSecure ? 'Protected' : 'Open'}
                </Badge>
              </div>
            </CardContent>
          </Card>

          {/* Description */}
          <div className="space-y-1">
            <Label className="text-sm">Description (Optional)</Label>
            <Textarea
              placeholder="Describe what this graph shows..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={3}
              className="text-sm"
              maxLength={500}
            />
            <div className="text-xs text-muted-foreground text-right">
              {description.length}/500
            </div>
          </div>

          {/* Expiry Settings */}
          <div className="space-y-1">
            <Label className="text-sm">Link Expires</Label>
            <Select value={expiry} onValueChange={setExpiry}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {expiryOptions.map(option => (
                  <SelectItem key={option.value} value={option.value}>
                    <div className="flex items-center gap-2">
                      <option.icon className="h-3 w-3" />
                      {option.label}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Password Protection */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-sm">Password Protection</Label>
              <Switch
                checked={usePassword}
                onCheckedChange={setUsePassword}
              />
            </div>

            {usePassword && (
              <div className="space-y-1">
                <Input
                  type="password"
                  placeholder="Enter password for access"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="text-sm"
                />
                <div className="text-xs text-muted-foreground">
                  Viewers will need this password to access the graph
                </div>
              </div>
            )}
          </div>

          <Separator />

          {/* Permissions */}
          <div className="space-y-3">
            <Label className="text-sm font-medium">Viewer Permissions</Label>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Download className="h-4 w-4 text-muted-foreground" />
                  <Label className="text-sm">Allow Download</Label>
                </div>
                <Switch
                  checked={allowDownload}
                  onCheckedChange={setAllowDownload}
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <MessageCircle className="h-4 w-4 text-muted-foreground" />
                  <Label className="text-sm">Allow Comments</Label>
                </div>
                <Switch
                  checked={allowComment}
                  onCheckedChange={setAllowComment}
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Edit3 className="h-4 w-4 text-muted-foreground" />
                  <Label className="text-sm">Allow Editing</Label>
                </div>
                <Switch
                  checked={allowEdit}
                  onCheckedChange={setAllowEdit}
                />
              </div>
            </div>
          </div>

          {/* Preview */}
          <Card className="bg-muted/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-xs">Share Preview</CardTitle>
            </CardHeader>
            <CardContent className="text-xs text-muted-foreground">
              <div className="space-y-1">
                <div>• Expires: {expiryOptions.find(opt => opt.value === expiry)?.label}</div>
                <div>• Password: {usePassword ? 'Required' : 'None'}</div>
                <div>• Download: {allowDownload ? 'Allowed' : 'Restricted'}</div>
                <div>• Comments: {allowComment ? 'Enabled' : 'Disabled'}</div>
                <div>• Editing: {allowEdit ? 'Enabled' : 'Disabled'}</div>
              </div>
            </CardContent>
          </Card>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={handleClose}>
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={isLoading || (usePassword && !password)}
          >
            {isLoading ? 'Creating...' : 'Create Share Link'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default ShareSettingsDialog;