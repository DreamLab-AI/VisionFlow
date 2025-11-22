/**
 * File drop zone component for loading ontology files
 * Migrated to shadcn/ui with Tailwind CSS
 */

import { useCallback, useState } from 'react';
import { Upload, AlertCircle, Loader2 } from 'lucide-react';
import { useGraphStore } from '../../stores/useGraphStore';
import { useUIStore } from '../../stores/useUIStore';
import { Card, CardContent } from '../ui/card';
import { Button } from '../ui/button';
import { cn } from '@/lib/utils';
import type { OntologyData } from '../../types/ontology';

interface FileDropZoneProps {
  onFileLoaded?: (filename: string) => void;
}

export function FileDropZone({ onFileLoaded }: FileDropZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadOntology = useGraphStore((state) => state.loadOntology);
  const addNotification = useUIStore((state) => state.addNotification);

  /**
   * Parse and validate ontology JSON data
   */
  const parseOntology = useCallback((data: any): OntologyData | null => {
    try {
      if (!data.class || !Array.isArray(data.class)) {
        throw new Error('Invalid ontology format: missing class array');
      }

      return {
        header: data.header,
        namespace: data.namespace,
        class: data.class,
        property: data.property || [],
        datatype: data.datatype,
        classAttribute: data.classAttribute,
        propertyAttribute: data.propertyAttribute
      };
    } catch (err) {
      console.error('Ontology parsing error:', err);
      return null;
    }
  }, []);

  /**
   * Load ontology from file
   */
  const loadFile = useCallback(async (file: File) => {
    setIsLoading(true);
    setError(null);

    try {
      if (!file.name.endsWith('.json')) {
        throw new Error('Only JSON files are currently supported');
      }

      const text = await file.text();
      const data = JSON.parse(text);
      const ontology = parseOntology(data);

      if (!ontology) {
        throw new Error('Failed to parse ontology data');
      }

      loadOntology(ontology);

      addNotification({
        type: 'success',
        message: `Loaded ${ontology.class.length} classes and ${ontology.property?.length || 0} properties`,
        duration: 3000
      });

      onFileLoaded?.(file.name);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setError(message);

      addNotification({
        type: 'error',
        message: `Failed to load file: ${message}`,
        duration: 5000
      });
    } finally {
      setIsLoading(false);
    }
  }, [loadOntology, parseOntology, addNotification, onFileLoaded]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      loadFile(files[0]);
    }
  }, [loadFile]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      loadFile(files[0]);
    }
  }, [loadFile]);

  const loadSample = useCallback(async (filename: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`/ontologies/${filename}`);
      if (!response.ok) {
        throw new Error('Failed to load sample ontology');
      }

      const data = await response.json();
      const ontology = parseOntology(data);

      if (!ontology) {
        throw new Error('Failed to parse sample ontology');
      }

      loadOntology(ontology);

      addNotification({
        type: 'success',
        message: `Loaded sample: ${filename}`,
        duration: 3000
      });

      onFileLoaded?.(filename);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setError(message);

      addNotification({
        type: 'error',
        message: `Failed to load sample: ${message}`,
        duration: 5000
      });
    } finally {
      setIsLoading(false);
    }
  }, [loadOntology, parseOntology, addNotification, onFileLoaded]);

  return (
    <div className="flex flex-col gap-6 w-full max-w-2xl mx-auto p-8">
      <Card
        className={cn(
          "border-2 border-dashed transition-all duration-300 cursor-pointer",
          isDragging && "border-primary bg-primary/5 scale-[1.02]",
          isLoading && "pointer-events-none opacity-70"
        )}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <CardContent className="flex flex-col items-center justify-center p-12 text-center">
          {isLoading ? (
            <>
              <Loader2 className="h-16 w-16 animate-spin text-primary mb-4" />
              <p className="text-lg font-medium">Loading ontology...</p>
            </>
          ) : (
            <>
              <Upload
                className={cn(
                  "h-16 w-16 mb-4 text-muted-foreground transition-colors",
                  isDragging && "text-primary animate-bounce"
                )}
              />
              <h3 className="text-xl font-semibold mb-2">Drop ontology file here</h3>
              <p className="text-muted-foreground mb-4">or</p>

              <label className="cursor-pointer">
                <input
                  type="file"
                  accept=".json"
                  onChange={handleFileInput}
                  className="hidden"
                />
                <Button type="button" size="lg">
                  Choose File
                </Button>
              </label>

              <p className="text-sm text-muted-foreground mt-4">
                Supports: JSON (WebVOWL format)
              </p>
            </>
          )}
        </CardContent>
      </Card>

      {error && (
        <Card className="border-destructive bg-destructive/10">
          <CardContent className="flex items-center gap-3 p-4">
            <AlertCircle className="h-5 w-5 text-destructive flex-shrink-0" />
            <span className="text-sm text-destructive">{error}</span>
          </CardContent>
        </Card>
      )}

      <div className="border-t pt-6">
        <h4 className="text-lg font-semibold mb-4">Sample Ontologies</h4>
        <div className="flex flex-wrap gap-3">
          <Button
            variant="outline"
            onClick={() => loadSample('foaf.json')}
            disabled={isLoading}
          >
            FOAF
          </Button>
          <Button
            variant="outline"
            onClick={() => loadSample('sioc.json')}
            disabled={isLoading}
          >
            SIOC
          </Button>
          <Button
            variant="outline"
            onClick={() => loadSample('minimal.json')}
            disabled={isLoading}
          >
            Minimal Example
          </Button>
        </div>
      </div>
    </div>
  );
}
