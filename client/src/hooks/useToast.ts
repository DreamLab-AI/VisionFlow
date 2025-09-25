// Simple toast hook for user notifications
import { useState, useCallback } from 'react';

export interface Toast {
  id: string;
  title: string;
  description?: string;
  variant?: 'default' | 'destructive' | 'success';
  duration?: number;
}

export const useToast = () => {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const toast = useCallback(({
    title,
    description,
    variant = 'default',
    duration = 5000,
  }: Omit<Toast, 'id'>) => {
    const id = Math.random().toString(36).substring(2);
    const newToast: Toast = {
      id,
      title,
      description,
      variant,
      duration,
    };

    setToasts(prev => [...prev, newToast]);

    // Auto-remove after duration
    if (duration > 0) {
      setTimeout(() => {
        setToasts(prev => prev.filter(t => t.id !== id));
      }, duration);
    }

    return id;
  }, []);

  const dismiss = useCallback((toastId: string) => {
    setToasts(prev => prev.filter(t => t.id !== toastId));
  }, []);

  const success = useCallback((title: string, description?: string) => {
    return toast({ title, description, variant: 'success' });
  }, [toast]);

  const error = useCallback((title: string, description?: string) => {
    return toast({ title, description, variant: 'destructive' });
  }, [toast]);

  return {
    toasts,
    toast,
    success,
    error,
    dismiss,
  };
};