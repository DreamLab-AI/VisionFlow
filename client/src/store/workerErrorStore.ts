import { create } from 'zustand';

export interface WorkerErrorState {
  hasWorkerError: boolean;
  errorMessage: string | null;
  errorDetails: string | null;
  /** Consecutive transient error count — auto-clears below threshold */
  transientErrorCount: number;
  setWorkerError: (message: string, details?: string) => void;
  /** Record a transient error (e.g. a single dropped frame).
   *  Only escalates to visible error after TRANSIENT_THRESHOLD consecutive hits. */
  recordTransientError: (context: string) => void;
  /** Reset transient counter (call on every successful frame). */
  resetTransientErrors: () => void;
  clearWorkerError: () => void;
  retryWorkerInit: (() => Promise<void>) | null;
  setRetryHandler: (handler: () => Promise<void>) => void;
}

/** Number of consecutive transient errors before showing the error modal */
const TRANSIENT_THRESHOLD = 30;

export const useWorkerErrorStore = create<WorkerErrorState>((set, get) => ({
  hasWorkerError: false,
  errorMessage: null,
  errorDetails: null,
  transientErrorCount: 0,
  retryWorkerInit: null,

  setWorkerError: (message: string, details?: string) => set({
    hasWorkerError: true,
    errorMessage: message,
    errorDetails: details || null
  }),

  recordTransientError: (context: string) => {
    const count = get().transientErrorCount + 1;
    if (count >= TRANSIENT_THRESHOLD && !get().hasWorkerError) {
      set({
        transientErrorCount: count,
        hasWorkerError: true,
        errorMessage: 'Data flow interrupted — the graph worker is not responding.',
        errorDetails: `${count} consecutive errors in ${context}. Click retry or the system will auto-recover when data resumes.`,
      });
    } else {
      set({ transientErrorCount: count });
    }
  },

  resetTransientErrors: () => {
    const state = get();
    if (state.transientErrorCount > 0) {
      // Auto-clear the error modal if it was caused by transient errors
      if (state.hasWorkerError && state.transientErrorCount >= TRANSIENT_THRESHOLD) {
        set({
          transientErrorCount: 0,
          hasWorkerError: false,
          errorMessage: null,
          errorDetails: null,
        });
      } else {
        set({ transientErrorCount: 0 });
      }
    }
  },

  clearWorkerError: () => set({
    hasWorkerError: false,
    errorMessage: null,
    errorDetails: null,
    transientErrorCount: 0,
  }),

  setRetryHandler: (handler: () => Promise<void>) => set({
    retryWorkerInit: handler
  })
}));
