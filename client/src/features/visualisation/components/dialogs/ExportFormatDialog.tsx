/**
 * Export Format Dialog Component
 * Select format and configure export options
 */

import React, { useState, useMemo } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/features/design-system/components/Dialog';
import { Button } from '@/features/design-system/components/Button';
import { Label } from '@/features/design-system/components/Label';
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
  FileImage,
  FileText,
  Database,
  Code,
  Download,
  Zap,
  HardDrive,
  Clock,
  Info
} from 'lucide-react';
import { ExportOptions, estimateExportSize } from '@/api/exportApi';

interface ExportFormatDialogProps {
  open: boolean;
  onClose: () => void;
  onExport: (options: ExportOptions) => void;
  graphData?: any;
  isLoading?: boolean;
}

const exportFormats = [
  {
    value: 'json',
    label: 'JSON Data',
    icon: Database,
    description: 'JavaScript Object Notation - standard web format',
    category: 'Data',
    features: ['Metadata', 'Compression', 'Fast'],
    size: 'Medium'
  },
  {
    value: 'csv',
    label: 'CSV Spreadsheet',
    icon: FileText,
    description: 'Comma-separated values for Excel/Sheets',
    category: 'Data',
    features: ['Excel Compatible', 'Lightweight'],
    size: 'Small'
  },
  {
    value: 'graphml',
    label: 'GraphML',
    icon: Code,
    description: 'XML-based graph markup language',
    category: 'Graph',
    features: ['Standard', 'Metadata', 'Interoperable'],
    size: 'Large'
  },
  {
    value: 'gexf',
    label: 'GEXF',
    icon: Code,
    description: 'Graph Exchange XML Format (Gephi compatible)',
    category: 'Graph',
    features: ['Gephi Compatible', 'Dynamic Graphs'],
    size: 'Large'
  },
  {
    value: 'svg',
    label: 'SVG Vector',
    icon: FileImage,
    description: 'Scalable vector graphics',
    category: 'Visual',
    features: ['Vector', 'Scalable', 'Web Compatible'],
    size: 'Large'
  },
  {
    value: 'png',
    label: 'PNG Image',
    icon: FileImage,
    description: 'Portable Network Graphics bitmap',
    category: 'Visual',
    features: ['High Quality', 'Lossless', 'Resolution Options'],
    size: 'Medium'
  },
  {
    value: 'pdf',
    label: 'PDF Document',
    icon: FileText,
    description: 'Portable Document Format with metadata',
    category: 'Document',
    features: ['Print Ready', 'Metadata', 'Universal'],
    size: 'Medium'
  },
  {
    value: 'xlsx',
    label: 'Excel Workbook',
    icon: FileText,
    description: 'Microsoft Excel format with multiple sheets',
    category: 'Data',
    features: ['Excel Compatible', 'Multiple Sheets'],
    size: 'Medium'
  }
];

export const ExportFormatDialog: React.FC<ExportFormatDialogProps> = ({
  open,
  onClose,
  onExport,
  graphData,
  isLoading = false
}) => {
  const [selectedFormat, setSelectedFormat] = useState('json');
  const [includeMetadata, setIncludeMetadata] = useState(true);
  const [compressionEnabled, setCompressionEnabled] = useState(false);
  const [resolution, setResolution] = useState('1920x1080');
  const [quality, setQuality] = useState(90);

  const selectedFormatData = exportFormats.find(f => f.value === selectedFormat);
  const isImageFormat = ['png', 'svg', 'pdf'].includes(selectedFormat);
  const isDataFormat = ['json', 'csv', 'graphml', 'gexf', 'xlsx'].includes(selectedFormat);

  const estimatedSize = useMemo(() => {
    if (!graphData) return 0;
    return estimateExportSize(graphData, selectedFormat);
  }, [graphData, selectedFormat]);

  const formatSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  };

  const handleExport = () => {
    const options: ExportOptions = {
      format: selectedFormat,
      includeMetadata: isDataFormat ? includeMetadata : false,
      compression: compressionEnabled,
      resolution: isImageFormat ? resolution : undefined,
      quality: selectedFormat === 'png' ? quality : undefined
    };

    onExport(options);
  };

  const handleClose = () => {
    setSelectedFormat('json');
    setIncludeMetadata(true);
    setCompressionEnabled(false);
    setResolution('1920x1080');
    setQuality(90);
    onClose();
  };

  const categories = Array.from(new Set(exportFormats.map(f => f.category)));

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Download className="h-5 w-5" />
            Export Graph Data
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          {/* Format Selection */}
          <div className="space-y-3">
            <Label className="text-sm font-medium">Export Format</Label>

            {categories.map(category => (
              <div key={category} className="space-y-2">
                <Label className="text-xs font-medium text-muted-foreground">
                  {category} Formats
                </Label>
                <div className="grid grid-cols-2 gap-2">
                  {exportFormats
                    .filter(format => format.category === category)
                    .map(format => (
                      <Card
                        key={format.value}
                        className={`cursor-pointer transition-all ${
                          selectedFormat === format.value
                            ? 'ring-2 ring-primary bg-primary/5'
                            : 'hover:bg-muted/50'
                        }`}
                        onClick={() => setSelectedFormat(format.value)}
                      >
                        <CardContent className="p-3">
                          <div className="flex items-start gap-2">
                            <format.icon className="h-4 w-4 mt-0.5 flex-shrink-0" />
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-1">
                                <span className="text-sm font-medium">
                                  {format.label}
                                </span>
                                <Badge variant="outline" className="text-xs">
                                  {format.size}
                                </Badge>
                              </div>
                              <p className="text-xs text-muted-foreground mt-1">
                                {format.description}
                              </p>
                              <div className="flex flex-wrap gap-1 mt-1">
                                {format.features.map(feature => (
                                  <Badge
                                    key={feature}
                                    variant="secondary"
                                    className="text-xs"
                                  >
                                    {feature}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                </div>
              </div>
            ))}
          </div>

          <Separator />

          {/* Format Options */}
          {selectedFormatData && (
            <div className="space-y-3">
              <Label className="text-sm font-medium">Export Options</Label>

              {/* Image Resolution */}
              {isImageFormat && (
                <div className="space-y-1">
                  <Label className="text-xs">Resolution</Label>
                  <Select value={resolution} onValueChange={setResolution}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1280x720">1280×720 (HD)</SelectItem>
                      <SelectItem value="1920x1080">1920×1080 (Full HD)</SelectItem>
                      <SelectItem value="2560x1440">2560×1440 (2K)</SelectItem>
                      <SelectItem value="3840x2160">3840×2160 (4K)</SelectItem>
                      <SelectItem value="7680x4320">7680×4320 (8K)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              )}

              {/* Image Quality */}
              {selectedFormat === 'png' && (
                <div className="space-y-1">
                  <Label className="text-xs">Quality</Label>
                  <Select value={quality.toString()} onValueChange={(value) => setQuality(Number(value))}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="60">60% (Small file)</SelectItem>
                      <SelectItem value="80">80% (Good quality)</SelectItem>
                      <SelectItem value="90">90% (High quality)</SelectItem>
                      <SelectItem value="100">100% (Maximum quality)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              )}

              {/* Metadata */}
              {isDataFormat && (
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Info className="h-4 w-4 text-muted-foreground" />
                    <Label className="text-xs">Include Metadata</Label>
                  </div>
                  <Switch
                    checked={includeMetadata}
                    onCheckedChange={setIncludeMetadata}
                  />
                </div>
              )}

              {/* Compression */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Zap className="h-4 w-4 text-muted-foreground" />
                  <Label className="text-xs">Enable Compression</Label>
                </div>
                <Switch
                  checked={compressionEnabled}
                  onCheckedChange={setCompressionEnabled}
                />
              </div>
            </div>
          )}

          <Separator />

          {/* Export Preview */}
          <Card className="bg-muted/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-xs flex items-center gap-2">
                <HardDrive className="h-3 w-3" />
                Export Summary
              </CardTitle>
            </CardHeader>
            <CardContent className="text-xs">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <div className="text-muted-foreground">Format:</div>
                  <div className="font-medium">
                    {selectedFormatData?.label}
                  </div>
                </div>
                <div className="space-y-1">
                  <div className="text-muted-foreground">Estimated Size:</div>
                  <div className="font-medium flex items-center gap-1">
                    {formatSize(estimatedSize)}
                    {compressionEnabled && (
                      <Badge variant="secondary" className="text-xs">
                        Compressed
                      </Badge>
                    )}
                  </div>
                </div>
                {isImageFormat && (
                  <div className="space-y-1">
                    <div className="text-muted-foreground">Resolution:</div>
                    <div className="font-medium">{resolution}</div>
                  </div>
                )}
                {isDataFormat && (
                  <div className="space-y-1">
                    <div className="text-muted-foreground">Metadata:</div>
                    <div className="font-medium">
                      {includeMetadata ? 'Included' : 'Excluded'}
                    </div>
                  </div>
                )}
              </div>

              <div className="mt-3 pt-2 border-t">
                <div className="text-muted-foreground mb-1">Features:</div>
                <div className="flex flex-wrap gap-1">
                  {selectedFormatData?.features.map(feature => (
                    <Badge key={feature} variant="outline" className="text-xs">
                      {feature}
                    </Badge>
                  ))}
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
            onClick={handleExport}
            disabled={isLoading}
          >
            {isLoading ? (
              <>
                <Clock className="h-3 w-3 mr-2 animate-spin" />
                Exporting...
              </>
            ) : (
              <>
                <Download className="h-3 w-3 mr-2" />
                Export {selectedFormatData?.label}
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default ExportFormatDialog;