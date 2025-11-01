

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
import { Badge } from '@/features/design-system/components/Badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/features/design-system/components/Select';
import { Card, CardContent, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { Separator } from '@/features/design-system/components/Separator';
import {
  Upload,
  Globe,
  Lock,
  Tag,
  FileText,
  Shield,
  AlertTriangle,
  CheckCircle,
  X
} from 'lucide-react';
import { PublishMetadata } from '@/api/exportApi';

interface PublishGraphDialogProps {
  open: boolean;
  onClose: () => void;
  onPublish: (metadata: PublishMetadata) => void;
  isLoading?: boolean;
}

const categories = [
  'Network Analysis',
  'Social Networks',
  'Biological Networks',
  'Knowledge Graphs',
  'Transportation',
  'Financial Networks',
  'Research Data',
  'Educational',
  'Other'
];

const licenses = [
  { value: 'cc-by', label: 'CC BY - Attribution', description: 'Free use with attribution' },
  { value: 'cc-by-sa', label: 'CC BY-SA - Attribution-ShareAlike', description: 'Free use with attribution, derivatives must share-alike' },
  { value: 'cc-by-nc', label: 'CC BY-NC - Attribution-NonCommercial', description: 'Non-commercial use only' },
  { value: 'mit', label: 'MIT License', description: 'Very permissive open source' },
  { value: 'apache', label: 'Apache 2.0', description: 'Permissive with patent protection' },
  { value: 'private', label: 'All Rights Reserved', description: 'Traditional copyright' }
];

export const PublishGraphDialog: React.FC<PublishGraphDialogProps> = ({
  open,
  onClose,
  onPublish,
  isLoading = false
}) => {
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [category, setCategory] = useState('');
  const [tags, setTags] = useState<string[]>([]);
  const [tagInput, setTagInput] = useState('');
  const [isPublic, setIsPublic] = useState(true);
  const [license, setLicense] = useState('cc-by');

  const handleAddTag = () => {
    const tag = tagInput.trim().toLowerCase();
    if (tag && !tags.includes(tag) && tags.length < 10) {
      setTags(prev => [...prev, tag]);
      setTagInput('');
    }
  };

  const handleRemoveTag = (tagToRemove: string) => {
    setTags(prev => prev.filter(tag => tag !== tagToRemove));
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      handleAddTag();
    }
  };

  const handleSubmit = () => {
    if (!title.trim() || !description.trim() || !category) return;

    const metadata: PublishMetadata = {
      title: title.trim(),
      description: description.trim(),
      tags,
      category,
      isPublic,
      license: license || undefined
    };

    onPublish(metadata);
  };

  const resetForm = () => {
    setTitle('');
    setDescription('');
    setCategory('');
    setTags([]);
    setTagInput('');
    setIsPublic(true);
    setLicense('cc-by');
  };

  const handleClose = () => {
    resetForm();
    onClose();
  };

  const selectedLicense = licenses.find(l => l.value === license);
  const canSubmit = title.trim() && description.trim() && category;

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Upload className="h-5 w-5" />
            Publish to Graph Repository
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          {}
          <Card className={`border ${isPublic ? 'border-blue-200 bg-blue-50' : 'border-gray-200 bg-gray-50'}`}>
            <CardContent className="pt-3 pb-2">
              <div className="flex items-center gap-2">
                {isPublic ? (
                  <Globe className="h-4 w-4 text-blue-600" />
                ) : (
                  <Lock className="h-4 w-4 text-gray-600" />
                )}
                <span className="text-sm font-medium">
                  {isPublic ? 'Public Repository' : 'Private Repository'}
                </span>
                <Badge variant={isPublic ? 'default' : 'secondary'} className="ml-auto">
                  {isPublic ? 'Discoverable' : 'Unlisted'}
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                {isPublic
                  ? 'Anyone can discover, view, and cite this graph'
                  : 'Only accessible via direct link, not searchable'
                }
              </p>
            </CardContent>
          </Card>

          {}
          <div className="space-y-3">
            <div className="space-y-1">
              <Label className="text-sm">Title *</Label>
              <Input
                placeholder="Give your graph a descriptive title"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                maxLength={100}
              />
              <div className="text-xs text-muted-foreground text-right">
                {title.length}/100
              </div>
            </div>

            <div className="space-y-1">
              <Label className="text-sm">Description *</Label>
              <Textarea
                placeholder="Describe what this graph represents, its data source, and key insights..."
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={4}
                maxLength={1000}
              />
              <div className="text-xs text-muted-foreground text-right">
                {description.length}/1000
              </div>
            </div>

            <div className="space-y-1">
              <Label className="text-sm">Category *</Label>
              <Select value={category} onValueChange={setCategory}>
                <SelectTrigger>
                  <SelectValue placeholder="Select a category" />
                </SelectTrigger>
                <SelectContent>
                  {categories.map(cat => (
                    <SelectItem key={cat} value={cat}>
                      {cat}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <Separator />

          {}
          <div className="space-y-2">
            <Label className="text-sm">Tags (Optional)</Label>
            <div className="flex flex-wrap gap-1 mb-2">
              {tags.map(tag => (
                <Badge
                  key={tag}
                  variant="secondary"
                  className="text-xs cursor-pointer hover:bg-red-100"
                  onClick={() => handleRemoveTag(tag)}
                >
                  <Tag className="h-3 w-3 mr-1" />
                  {tag}
                  <X className="h-3 w-3 ml-1" />
                </Badge>
              ))}
            </div>
            <div className="flex gap-2">
              <Input
                placeholder="Add tags (press Enter or comma to add)"
                value={tagInput}
                onChange={(e) => setTagInput(e.target.value)}
                onKeyDown={handleKeyPress}
                disabled={tags.length >= 10}
                className="flex-1"
              />
              <Button
                type="button"
                variant="outline"
                onClick={handleAddTag}
                disabled={!tagInput.trim() || tags.length >= 10}
              >
                Add
              </Button>
            </div>
            <div className="text-xs text-muted-foreground">
              Add up to 10 tags to help others discover your graph
            </div>
          </div>

          <Separator />

          {}
          <div className="space-y-3">
            <Label className="text-sm font-medium">Publishing Options</Label>

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Globe className="h-4 w-4 text-muted-foreground" />
                <div>
                  <Label className="text-sm">Make Public</Label>
                  <p className="text-xs text-muted-foreground">
                    Allow others to discover this graph
                  </p>
                </div>
              </div>
              <Switch
                checked={isPublic}
                onCheckedChange={setIsPublic}
              />
            </div>

            <div className="space-y-1">
              <Label className="text-sm">License</Label>
              <Select value={license} onValueChange={setLicense}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {licenses.map(lic => (
                    <SelectItem key={lic.value} value={lic.value}>
                      <div className="flex flex-col">
                        <span>{lic.label}</span>
                        <span className="text-xs text-muted-foreground">
                          {lic.description}
                        </span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <Separator />

          {}
          <Card className="bg-muted/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-xs flex items-center gap-2">
                <FileText className="h-3 w-3" />
                Publication Preview
              </CardTitle>
            </CardHeader>
            <CardContent className="text-xs">
              <div className="space-y-2">
                <div>
                  <span className="font-medium">{title || 'Untitled Graph'}</span>
                  <div className="flex items-center gap-1 mt-1">
                    <Badge variant="outline" className="text-xs">
                      {category || 'Uncategorized'}
                    </Badge>
                    <Badge variant={isPublic ? 'default' : 'secondary'} className="text-xs">
                      {isPublic ? 'Public' : 'Private'}
                    </Badge>
                    {selectedLicense && (
                      <Badge variant="outline" className="text-xs">
                        {selectedLicense.label}
                      </Badge>
                    )}
                  </div>
                </div>

                <p className="text-muted-foreground">
                  {description || 'No description provided'}
                </p>

                {tags.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {tags.map(tag => (
                      <Badge key={tag} variant="outline" className="text-xs">
                        #{tag}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {}
          <Card className="border-yellow-200 bg-yellow-50">
            <CardContent className="pt-3 pb-2">
              <div className="flex items-start gap-2">
                <AlertTriangle className="h-4 w-4 text-yellow-600 mt-0.5" />
                <div className="text-xs">
                  <p className="font-medium text-yellow-800">Publishing Agreement</p>
                  <p className="text-yellow-700 mt-1">
                    By publishing, you confirm that you have rights to share this data and agree to our
                    terms of service. Published graphs cannot be completely removed once indexed.
                  </p>
                </div>
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
            disabled={!canSubmit || isLoading}
          >
            {isLoading ? (
              <>
                <Upload className="h-3 w-3 mr-2 animate-spin" />
                Publishing...
              </>
            ) : (
              <>
                <Upload className="h-3 w-3 mr-2" />
                Publish Graph
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default PublishGraphDialog;