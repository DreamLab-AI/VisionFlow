/**
 * Graph Export Tab Component
 * Data export and sharing functionality with UK English localisation
 */

import React, { useState, useCallback } from 'react';
import { 
  Download, 
  Share,
  FileImage,
  FileText,
  Database,
  Code,
  Globe,
  Mail,
  Copy,
  ExternalLink,
  AlertCircle,
  CheckCircle,
  Loader2
} from 'lucide-react';
import { Button } from '@/features/design-system/components/Button';
import { Switch } from '@/features/design-system/components/Switch';
import { Label } from '@/features/design-system/components/Label';
import { Badge } from '@/features/design-system/components/Badge';
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/features/design-system/components/Select';
import { Card, CardContent, CardHeader, CardTitle } from '@/features/design-system/components/Card';
import { Input } from '@/features/design-system/components/Input';
import { Textarea } from '@/features/design-system/components/Textarea';
import { Separator } from '@/features/design-system/components/Separator';
import { toast } from '@/features/design-system/components/Toast';

interface GraphExportTabProps {
  graphId?: string;
  graphData?: any;
  onExport?: (format: string, options: any) => void;
}

export const GraphExportTab: React.FC<GraphExportTabProps> = ({ 
  graphId = 'default',
  graphData,
  onExport
}) => {
  // Export states
  const [exportFormat, setExportFormat] = useState('json');
  const [imageFormat, setImageFormat] = useState('png');
  const [imageResolution, setImageResolution] = useState('1920x1080');
  const [includeMetadata, setIncludeMetadata] = useState(true);
  const [compressionEnabled, setCompressionEnabled] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  
  // Sharing states
  const [shareUrl, setShareUrl] = useState('');
  const [shareExpiry, setShareExpiry] = useState('7days');
  const [sharePassword, setSharePassword] = useState('');
  const [shareDescription, setShareDescription] = useState('');
  const [isSharing, setIsSharing] = useState(false);
  
  // Publication states
  const [embedCode, setEmbedCode] = useState('');
  const [apiEndpoint, setApiEndpoint] = useState('');

  const exportFormats = [
    { value: 'json', label: 'JSON Data', icon: Database },
    { value: 'csv', label: 'CSV Spreadsheet', icon: FileText },
    { value: 'graphml', label: 'GraphML', icon: Code },
    { value: 'gexf', label: 'GEXF Format', icon: Code },
    { value: 'svg', label: 'SVG Vector', icon: FileImage },
    { value: 'png', label: 'PNG Image', icon: FileImage },
    { value: 'pdf', label: 'PDF Report', icon: FileText }
  ];

  const handleDataExport = useCallback(async (format: string) => {
    setIsExporting(true);
    
    const exportOptions = {
      format,
      includeMetadata,
      compression: compressionEnabled,
      resolution: format === 'png' ? imageResolution : undefined
    };
    
    try {
      toast({
        title: "Starting Export",
        description: `Preparing ${format.toUpperCase()} export...`
      });
      
      // Simulate export process
      setTimeout(() => {
        setIsExporting(false);
        onExport?.(format, exportOptions);
        
        toast({
          title: "Export Complete",
          description: `Graph data exported as ${format.toUpperCase()} successfully`,
          action: {
            label: "View",
            onClick: () => console.log('View export')
          }
        });
      }, 2000);
      
    } catch (error) {
      setIsExporting(false);
      toast({
        title: "Export Failed",
        description: "An error occurred during export. Please try again.",
        variant: "destructive"
      });
    }
  }, [includeMetadata, compressionEnabled, imageResolution, onExport]);

  const handleCreateShareLink = useCallback(async () => {
    setIsSharing(true);
    
    try {
      toast({
        title: "Creating Share Link",
        description: "Generating secure sharing URL..."
      });
      
      // Simulate share link creation
      setTimeout(() => {
        const mockShareUrl = `${window.location.origin}/shared/${graphId}-${Date.now()}`;
        setShareUrl(mockShareUrl);
        setIsSharing(false);
        
        toast({
          title: "Share Link Created",
          description: "Link copied to clipboard automatically",
          action: {
            label: "Copy Again",
            onClick: () => {
              navigator.clipboard.writeText(mockShareUrl);
              toast({ title: "Link copied to clipboard" });
            }
          }
        });
        
        // Auto-copy to clipboard
        navigator.clipboard.writeText(mockShareUrl);
      }, 1500);
      
    } catch (error) {
      setIsSharing(false);
      toast({
        title: "Share Link Failed",
        description: "Could not create share link. Please try again.",
        variant: "destructive"
      });
    }
  }, [graphId]);

  const generateEmbedCode = useCallback(() => {
    const embedHtml = `<iframe 
  src="${window.location.origin}/embed/${graphId}" 
  width="800" 
  height="600" 
  frameborder="0"
  title="Interactive Graph Visualisation">
</iframe>`;
    
    setEmbedCode(embedHtml);
    navigator.clipboard.writeText(embedHtml);
    
    toast({
      title: "Embed Code Generated",
      description: "HTML embed code copied to clipboard"
    });
  }, [graphId]);

  const generateApiEndpoint = useCallback(() => {
    const endpoint = `${window.location.origin}/api/graphs/${graphId}/data`;
    setApiEndpoint(endpoint);
    navigator.clipboard.writeText(endpoint);
    
    toast({
      title: "API Endpoint Generated",
      description: "REST API endpoint copied to clipboard"
    });
  }, [graphId]);

  return (
    <div className="space-y-4">
      {/* Data Export */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <Download className="h-4 w-4" />
            Data Export
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-1">
            <Label className="text-xs">Export Format</Label>
            <Select value={exportFormat} onValueChange={setExportFormat}>
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select format" />
              </SelectTrigger>
              <SelectContent>
                {exportFormats.map(format => (
                  <SelectItem key={format.value} value={format.value}>
                    <div className="flex items-center gap-2">
                      <format.icon className="h-3 w-3" />
                      {format.label}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          
          {['png', 'svg'].includes(exportFormat) && (
            <div className="space-y-1">
              <Label className="text-xs">Image Resolution</Label>
              <Select value={imageResolution} onValueChange={setImageResolution}>
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select resolution" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1920x1080">1920×1080 (Full HD)</SelectItem>
                  <SelectItem value="2560x1440">2560×1440 (2K)</SelectItem>
                  <SelectItem value="3840x2160">3840×2160 (4K)</SelectItem>
                  <SelectItem value="7680x4320">7680×4320 (8K)</SelectItem>
                </SelectContent>
              </Select>
            </div>
          )}
          
          <div className="flex items-center justify-between">
            <Label className="text-xs">Include Metadata</Label>
            <Switch
              checked={includeMetadata}
              onCheckedChange={setIncludeMetadata}
            />
          </div>
          
          <div className="flex items-center justify-between">
            <Label className="text-xs">Enable Compression</Label>
            <Switch
              checked={compressionEnabled}
              onCheckedChange={setCompressionEnabled}
            />
          </div>
          
          <Button 
            onClick={() => handleDataExport(exportFormat)}
            disabled={isExporting}
            className="w-full"
          >
            {isExporting ? (
              <>
                <Loader2 className="h-3 w-3 mr-2 animate-spin" />
                Exporting...
              </>
            ) : (
              <>
                <Download className="h-3 w-3 mr-2" />
                Export {exportFormat.toUpperCase()}
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Sharing & Collaboration */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <Share className="h-4 w-4" />
            Sharing & Collaboration
            <Badge variant="secondary" className="text-xs">
              <AlertCircle className="h-3 w-3 mr-1" />
              Partial
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-2">
            <Label className="text-xs">Share Description</Label>
            <Textarea
              placeholder="Describe what this graph shows..."
              value={shareDescription}
              onChange={(e) => setShareDescription(e.target.value)}
              className="text-xs"
              rows={2}
            />
          </div>
          
          <div className="grid grid-cols-2 gap-2">
            <div className="space-y-1">
              <Label className="text-xs">Expires In</Label>
              <Select value={shareExpiry} onValueChange={setShareExpiry}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1hour">1 Hour</SelectItem>
                  <SelectItem value="1day">1 Day</SelectItem>
                  <SelectItem value="7days">7 Days</SelectItem>
                  <SelectItem value="30days">30 Days</SelectItem>
                  <SelectItem value="never">Never</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="space-y-1">
              <Label className="text-xs">Password (Optional)</Label>
              <Input
                type="password"
                placeholder="Security password"
                value={sharePassword}
                onChange={(e) => setSharePassword(e.target.value)}
                className="text-xs"
              />
            </div>
          </div>
          
          <Button 
            onClick={handleCreateShareLink}
            disabled={isSharing}
            className="w-full"
            variant="outline"
          >
            {isSharing ? (
              <>
                <Loader2 className="h-3 w-3 mr-2 animate-spin" />
                Creating Link...
              </>
            ) : (
              <>
                <Globe className="h-3 w-3 mr-2" />
                Create Share Link
              </>
            )}
          </Button>
          
          {shareUrl && (
            <div className="p-2 bg-muted rounded space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs text-green-600 flex items-center gap-1">
                  <CheckCircle className="h-3 w-3" />
                  Share Link Created
                </Label>
                <Button 
                  size="sm" 
                  variant="ghost"
                  onClick={() => {
                    navigator.clipboard.writeText(shareUrl);
                    toast({ title: "Link copied to clipboard" });
                  }}
                >
                  <Copy className="h-3 w-3" />
                </Button>
              </div>
              <div className="text-xs font-mono p-1 bg-background rounded border break-all">
                {shareUrl}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Web Integration */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <Code className="h-4 w-4" />
            Web Integration
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-2 gap-2">
            <Button 
              variant="outline" 
              size="sm"
              onClick={generateEmbedCode}
              className="w-full"
            >
              <FileText className="h-3 w-3 mr-1" />
              Embed Code
            </Button>
            <Button 
              variant="outline" 
              size="sm"
              onClick={generateApiEndpoint}
              className="w-full"
            >
              <Database className="h-3 w-3 mr-1" />
              API Endpoint
            </Button>
          </div>
          
          {embedCode && (
            <div className="space-y-1">
              <Label className="text-xs">HTML Embed Code</Label>
              <Textarea
                value={embedCode}
                readOnly
                className="text-xs font-mono"
                rows={4}
                onClick={(e) => (e.target as HTMLTextAreaElement).select()}
              />
            </div>
          )}
          
          {apiEndpoint && (
            <div className="space-y-1">
              <Label className="text-xs">REST API Endpoint</Label>
              <div className="flex items-center gap-2">
                <Input
                  value={apiEndpoint}
                  readOnly
                  className="text-xs font-mono flex-1"
                  onClick={(e) => (e.target as HTMLInputElement).select()}
                />
                <Button 
                  size="sm" 
                  variant="ghost"
                  onClick={() => window.open(apiEndpoint, '_blank')}
                >
                  <ExternalLink className="h-3 w-3" />
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <Mail className="h-4 w-4" />
            Quick Actions
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <div className="grid grid-cols-2 gap-2">
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => handleDataExport('png')}
              className="w-full"
            >
              <FileImage className="h-3 w-3 mr-1" />
              Save Image
            </Button>
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => handleDataExport('json')}
              className="w-full"
            >
              <Database className="h-3 w-3 mr-1" />
              Export Data
            </Button>
          </div>
          
          <div className="text-xs text-muted-foreground p-2 bg-muted/50 rounded">
            <strong>Note:</strong> Advanced sharing and collaboration features are under development. 
            Current implementation provides basic export and link generation.
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default GraphExportTab;