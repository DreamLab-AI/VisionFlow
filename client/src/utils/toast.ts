// Simple toast utility for UI notifications
// This provides a consistent interface for toast notifications across the app

export interface ToastMessage {
  id?: string;
  message: string;
  type: 'success' | 'error' | 'warning' | 'info';
  duration?: number;
}

// Global toast queue - in a real app this would use a toast library
let toastQueue: ToastMessage[] = [];
let toastListeners: ((toasts: ToastMessage[]) => void)[] = [];

const notifyListeners = () => {
  toastListeners.forEach(listener => listener([...toastQueue]));
};

const addToast = (message: string, type: ToastMessage['type'], duration = 4000) => {
  const id = Math.random().toString(36).substr(2, 9);
  const toast: ToastMessage = { id, message, type, duration };
  
  toastQueue.push(toast);
  notifyListeners();
  
  // Auto-remove toast after duration
  setTimeout(() => {
    removeToast(id);
  }, duration);
  
  // Also log to console for development
  console[type === 'error' ? 'error' : 'info'](`Toast ${type}: ${message}`);
};

const removeToast = (id: string) => {
  toastQueue = toastQueue.filter(toast => toast.id !== id);
  notifyListeners();
};

// Toast API
export const toast = {
  success: (message: string, duration?: number) => addToast(message, 'success', duration),
  error: (message: string, duration?: number) => addToast(message, 'error', duration),
  warning: (message: string, duration?: number) => addToast(message, 'warning', duration),
  info: (message: string, duration?: number) => addToast(message, 'info', duration),
  
  // For subscribing to toast updates (for React components)
  subscribe: (listener: (toasts: ToastMessage[]) => void) => {
    toastListeners.push(listener);
    return () => {
      toastListeners = toastListeners.filter(l => l !== listener);
    };
  },
  
  // Remove specific toast
  dismiss: removeToast,
  
  // Clear all toasts
  clear: () => {
    toastQueue = [];
    notifyListeners();
  }
};

// Default export for convenience
export default toast;