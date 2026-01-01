import { create } from 'zustand';

export interface WorkerErrorState {
  hasWorkerError: boolean;
  errorMessage: string | null;
  errorDetails: string | null;
  setWorkerError: (message: string, details?: string) => void;
  clearWorkerError: () => void;
  retryWorkerInit: (() => Promise<void>) | null;
  setRetryHandler: (handler: () => Promise<void>) => void;
}

export const useWorkerErrorStore = create<WorkerErrorState>((set) => ({
  hasWorkerError: false,
  errorMessage: null,
  errorDetails: null,
  retryWorkerInit: null,

  setWorkerError: (message: string, details?: string) => set({
    hasWorkerError: true,
    errorMessage: message,
    errorDetails: details || null
  }),

  clearWorkerError: () => set({
    hasWorkerError: false,
    errorMessage: null,
    errorDetails: null
  }),

  setRetryHandler: (handler: () => Promise<void>) => set({
    retryWorkerInit: handler
  })
}));
