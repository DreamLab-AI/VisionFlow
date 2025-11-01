


export const downloadFile = (
  data: Blob | string | ArrayBuffer,
  filename: string,
  mimeType: string = 'application/octet-stream'
): void => {
  let blob: Blob;

  if (data instanceof Blob) {
    blob = data;
  } else if (data instanceof ArrayBuffer) {
    blob = new Blob([data], { type: mimeType });
  } else {
    blob = new Blob([data], { type: mimeType });
  }

  const url = URL.createObjectURL(blob);

  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.style.display = 'none';

  document.body.appendChild(link);
  link.click();

  
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};


export const downloadJSON = (
  data: any,
  filename: string = 'data.json',
  pretty: boolean = true
): void => {
  const jsonString = pretty
    ? JSON.stringify(data, null, 2)
    : JSON.stringify(data);

  downloadFile(jsonString, filename, 'application/json');
};


export const downloadCSV = (
  data: string[][] | Record<string, any>[],
  filename: string = 'data.csv'
): void => {
  let csvContent: string;

  if (Array.isArray(data) && data.length > 0) {
    if (Array.isArray(data[0])) {
      
      csvContent = (data as string[][])
        .map(row => row.map(cell => `"${String(cell).replace(/"/g, '""')}"`).join(','))
        .join('\n');
    } else {
      
      const objects = data as Record<string, any>[];
      const headers = Object.keys(objects[0] || {});

      const headerRow = headers.map(h => `"${h}"`).join(',');
      const dataRows = objects.map(obj =>
        headers.map(h => `"${String(obj[h] || '').replace(/"/g, '""')}"`).join(',')
      );

      csvContent = [headerRow, ...dataRows].join('\n');
    }
  } else {
    csvContent = '';
  }

  downloadFile(csvContent, filename, 'text/csv');
};


export const downloadText = (
  content: string,
  filename: string = 'file.txt',
  mimeType: string = 'text/plain'
): void => {
  downloadFile(content, filename, mimeType);
};


export const downloadXML = (
  content: string,
  filename: string = 'data.xml'
): void => {
  downloadFile(content, filename, 'application/xml');
};


export const downloadImageFromCanvas = (
  canvas: HTMLCanvasElement,
  filename: string = 'image.png',
  quality: number = 0.92
): void => {
  canvas.toBlob(
    (blob) => {
      if (blob) {
        downloadFile(blob, filename, 'image/png');
      }
    },
    'image/png',
    quality
  );
};


export const downloadImageFromUrl = async (
  url: string,
  filename?: string
): Promise<void> => {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const blob = await response.blob();
    const actualFilename = filename || url.split('/').pop() || 'image.png';

    downloadFile(blob, actualFilename, blob.type);
  } catch (error) {
    console.error('Failed to download image:', error);
    throw error;
  }
};


export const copyToClipboard = async (text: string): Promise<boolean> => {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    } else {
      
      return fallbackCopyToClipboard(text);
    }
  } catch (error) {
    console.error('Failed to copy to clipboard:', error);
    return fallbackCopyToClipboard(text);
  }
};


const fallbackCopyToClipboard = (text: string): boolean => {
  try {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-999999px';
    textArea.style.top = '-999999px';
    textArea.style.opacity = '0';

    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();

    const result = document.execCommand('copy');
    document.body.removeChild(textArea);

    return result;
  } catch (error) {
    console.error('Fallback clipboard copy failed:', error);
    return false;
  }
};


export const readFileAsText = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });
};


export const readFileAsDataURL = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
};


export const readFileAsArrayBuffer = (file: File): Promise<ArrayBuffer> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as ArrayBuffer);
    reader.onerror = () => reject(reader.error);
    reader.readAsArrayBuffer(file);
  });
};


export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};


export const getFileExtension = (filename: string): string => {
  const lastDot = filename.lastIndexOf('.');
  return lastDot !== -1 ? filename.substring(lastDot + 1).toLowerCase() : '';
};


export const generateSafeFilename = (
  name: string,
  extension: string = '',
  maxLength: number = 255
): string => {
  
  const safeName = name
    .replace(/[<>:"/\\|?*\x00-\x1f]/g, '_')
    .replace(/\s+/g, '_')
    .replace(/_+/g, '_')
    .trim();

  
  const fullName = extension ? `${safeName}.${extension}` : safeName;

  
  return fullName.length > maxLength
    ? fullName.substring(0, maxLength - (extension.length + 1)) + (extension ? `.${extension}` : '')
    : fullName;
};


export const isPreviewableFileType = (mimeType: string): boolean => {
  const previewableTypes = [
    'text/',
    'application/json',
    'application/xml',
    'image/',
    'audio/',
    'video/'
  ];

  return previewableTypes.some(type => mimeType.startsWith(type));
};


export const createDownloadProgress = (
  onProgress: (loaded: number, total: number) => void
) => {
  return async (url: string, filename?: string): Promise<void> => {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const contentLength = response.headers.get('content-length');
    const total = contentLength ? parseInt(contentLength, 10) : 0;
    let loaded = 0;

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Response body is not readable');
    }

    const chunks: Uint8Array[] = [];

    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      chunks.push(value);
      loaded += value.length;

      if (total > 0) {
        onProgress(loaded, total);
      }
    }

    
    const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
    const result = new Uint8Array(totalLength);
    let offset = 0;

    for (const chunk of chunks) {
      result.set(chunk, offset);
      offset += chunk.length;
    }

    const blob = new Blob([result]);
    const actualFilename = filename || url.split('/').pop() || 'download';

    downloadFile(blob, actualFilename);
  };
};