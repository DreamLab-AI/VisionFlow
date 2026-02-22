/**
 * ResourceEditor Component
 * JSON-LD resource editor with validation
 */

import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { Save, X, RefreshCw, AlertTriangle, Check, Clock, FileJson, Code, Eye } from 'lucide-react';
import { Button } from '@/features/design-system/components/Button';
import { Card, CardContent, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { Badge } from '@/features/design-system/components/Badge';
import { Separator } from '@/features/design-system/components/Separator';
import { Textarea } from '@/features/design-system/components/Textarea';
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from '@/features/design-system/components/Tabs';
import {
  TooltipProvider,
  TooltipRoot,
  TooltipTrigger,
  TooltipContent,
} from '@/features/design-system/components/Tooltip';
import { cn } from '@/utils/classNameUtils';
import { createLogger } from '../../../utils/loggerConfig';
import { useSolidResource, ResourceMetadata } from '../hooks/useSolidResource';

const logger = createLogger('ResourceEditor');
import { JsonLdDocument } from '@/services/SolidPodService';

interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

const validateJsonLd = (content: string): ValidationResult => {
  const result: ValidationResult = {
    valid: true,
    errors: [],
    warnings: [],
  };

  try {
    const parsed = JSON.parse(content);

    // Check for @context
    if (!parsed['@context']) {
      result.warnings.push('Missing @context - document may not be valid JSON-LD');
    }

    // Check for @type
    if (!parsed['@type']) {
      result.warnings.push('Missing @type - consider adding a type for better interoperability');
    }

    // Check for @id
    if (!parsed['@id']) {
      result.warnings.push('Missing @id - consider adding an identifier');
    }

    // Validate @context format
    if (parsed['@context']) {
      const ctx = parsed['@context'];
      if (typeof ctx !== 'string' && typeof ctx !== 'object') {
        result.errors.push('@context must be a string URL or an object');
        result.valid = false;
      }
    }
  } catch (e) {
    result.valid = false;
    result.errors.push(`Invalid JSON: ${e instanceof Error ? e.message : 'Parse error'}`);
  }

  return result;
};

interface MetadataDisplayProps {
  metadata: ResourceMetadata;
}

const MetadataDisplay: React.FC<MetadataDisplayProps> = ({ metadata }) => {
  return (
    <div className="flex flex-wrap items-center gap-4 text-xs text-muted-foreground">
      {metadata.lastModified && (
        <div className="flex items-center gap-1">
          <Clock className="h-3 w-3" />
          <span>Modified: {new Date(metadata.lastModified).toLocaleString()}</span>
        </div>
      )}
      {metadata.etag && (
        <div className="flex items-center gap-1">
          <Code className="h-3 w-3" />
          <span>ETag: {metadata.etag}</span>
        </div>
      )}
      {metadata.contentType && (
        <div className="flex items-center gap-1">
          <FileJson className="h-3 w-3" />
          <span>{metadata.contentType}</span>
        </div>
      )}
    </div>
  );
};

interface JsonPreviewProps {
  content: string;
}

const JsonPreview: React.FC<JsonPreviewProps> = ({ content }) => {
  const formatted = useMemo(() => {
    try {
      return JSON.stringify(JSON.parse(content), null, 2);
    } catch {
      return content;
    }
  }, [content]);

  return (
    <pre className="p-4 bg-muted rounded-md overflow-auto text-sm font-mono whitespace-pre">
      {formatted}
    </pre>
  );
};

export interface ResourceEditorProps {
  resourceUrl: string;
  onClose?: () => void;
  className?: string;
}

export const ResourceEditor: React.FC<ResourceEditorProps> = ({
  resourceUrl,
  onClose,
  className,
}) => {
  const {
    resource,
    metadata,
    isLoading,
    error,
    fetch: fetchResource,
    save: saveResource,
  } = useSolidResource(resourceUrl);

  const [content, setContent] = useState('');
  const [isDirty, setIsDirty] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [validation, setValidation] = useState<ValidationResult>({
    valid: true,
    errors: [],
    warnings: [],
  });
  const [activeTab, setActiveTab] = useState('edit');

  // Initialize content from resource
  useEffect(() => {
    if (resource) {
      const jsonContent = JSON.stringify(resource, null, 2);
      setContent(jsonContent);
      setIsDirty(false);
      setValidation(validateJsonLd(jsonContent));
    }
  }, [resource]);

  // Fetch resource on mount
  useEffect(() => {
    fetchResource();
  }, [fetchResource]);

  const handleContentChange = useCallback((value: string) => {
    setContent(value);
    setIsDirty(true);
    setValidation(validateJsonLd(value));
  }, []);

  const handleSave = useCallback(async () => {
    if (!validation.valid) {
      return;
    }

    setIsSaving(true);
    try {
      const parsed = JSON.parse(content) as JsonLdDocument;
      const success = await saveResource(parsed);
      if (success) {
        setIsDirty(false);
      }
    } catch (e) {
      // Validation should catch JSON errors
      logger.error('Save failed:', e);
    } finally {
      setIsSaving(false);
    }
  }, [content, validation.valid, saveResource]);

  const handleRefresh = useCallback(() => {
    if (isDirty) {
      if (!window.confirm('You have unsaved changes. Discard and refresh?')) {
        return;
      }
    }
    fetchResource();
  }, [isDirty, fetchResource]);

  const handleCancel = useCallback(() => {
    if (isDirty) {
      if (!window.confirm('You have unsaved changes. Discard?')) {
        return;
      }
    }
    onClose?.();
  }, [isDirty, onClose]);

  const resourceName = useMemo(() => {
    return resourceUrl.split('/').filter(Boolean).pop() || 'Resource';
  }, [resourceUrl]);

  if (isLoading && !resource) {
    return (
      <Card className={cn('h-full flex flex-col', className)}>
        <CardContent className="flex-1 flex items-center justify-center">
          <RefreshCw className="h-6 w-6 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={cn('h-full flex flex-col', className)}>
      <CardHeader className="py-3 px-4 border-b flex-shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 min-w-0">
            <FileJson className="h-4 w-4 flex-shrink-0 text-blue-500" />
            <CardTitle className="text-sm font-medium truncate">
              {resourceName}
            </CardTitle>
            {isDirty && (
              <Badge variant="warning" className="text-xs">
                Unsaved
              </Badge>
            )}
          </div>

          <div className="flex items-center gap-1">
            <TooltipProvider>
              <TooltipRoot>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    onClick={handleRefresh}
                    disabled={isLoading}
                  >
                    <RefreshCw className={cn('h-4 w-4', isLoading && 'animate-spin')} />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Refresh</TooltipContent>
              </TooltipRoot>
            </TooltipProvider>

            {onClose && (
              <TooltipProvider>
                <TooltipRoot>
                  <TooltipTrigger asChild>
                    <Button variant="ghost" size="icon-sm" onClick={handleCancel}>
                      <X className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Close</TooltipContent>
                </TooltipRoot>
              </TooltipProvider>
            )}
          </div>
        </div>

        <MetadataDisplay metadata={metadata} />
      </CardHeader>

      <div className="flex-1 flex flex-col min-h-0">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col">
          <TabsList className="mx-4 mt-2">
            <TabsTrigger value="edit" className="text-xs">
              <Code className="h-3 w-3 mr-1" />
              Edit
            </TabsTrigger>
            <TabsTrigger value="preview" className="text-xs">
              <Eye className="h-3 w-3 mr-1" />
              Preview
            </TabsTrigger>
          </TabsList>

          <TabsContent value="edit" className="flex-1 p-4 pt-2 flex flex-col">
            <Textarea
              value={content}
              onChange={(e) => handleContentChange(e.target.value)}
              className={cn(
                'flex-1 font-mono text-sm resize-none',
                !validation.valid && 'border-destructive'
              )}
              placeholder="Enter JSON-LD content..."
            />
          </TabsContent>

          <TabsContent value="preview" className="flex-1 p-4 pt-2 overflow-auto">
            <JsonPreview content={content} />
          </TabsContent>
        </Tabs>

        {/* Validation Messages */}
        {(validation.errors.length > 0 || validation.warnings.length > 0) && (
          <div className="px-4 pb-2 space-y-2">
            {validation.errors.map((err, i) => (
              <div
                key={`error-${i}`}
                className="flex items-center gap-2 text-xs text-destructive"
              >
                <AlertTriangle className="h-3 w-3 flex-shrink-0" />
                {err}
              </div>
            ))}
            {validation.warnings.map((warn, i) => (
              <div
                key={`warning-${i}`}
                className="flex items-center gap-2 text-xs text-yellow-600"
              >
                <AlertTriangle className="h-3 w-3 flex-shrink-0" />
                {warn}
              </div>
            ))}
          </div>
        )}

        {error && (
          <div className="mx-4 mb-2 p-2 rounded-md bg-destructive/10 text-destructive text-xs flex items-center gap-2">
            <AlertTriangle className="h-3 w-3 flex-shrink-0" />
            {error}
          </div>
        )}

        <Separator />

        {/* Actions */}
        <div className="p-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            {validation.valid ? (
              <Badge variant="success" className="text-xs">
                <Check className="h-3 w-3 mr-1" />
                Valid JSON-LD
              </Badge>
            ) : (
              <Badge variant="destructive" className="text-xs">
                <AlertTriangle className="h-3 w-3 mr-1" />
                Invalid
              </Badge>
            )}
          </div>

          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={handleCancel}>
              Cancel
            </Button>
            <Button
              size="sm"
              onClick={handleSave}
              disabled={!isDirty || !validation.valid || isSaving}
              loading={isSaving}
            >
              <Save className="h-4 w-4 mr-2" />
              Save
            </Button>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default ResourceEditor;
