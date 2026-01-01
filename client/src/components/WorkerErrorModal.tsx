import React, { useState } from 'react';
import { Modal, ModalBody, ModalFooter } from '../features/design-system/components/Modal';
import { useWorkerErrorStore } from '../store/workerErrorStore';
import { AlertTriangle, RefreshCw, ExternalLink } from 'lucide-react';

export const WorkerErrorModal: React.FC = () => {
  const { hasWorkerError, errorMessage, errorDetails, clearWorkerError, retryWorkerInit } = useWorkerErrorStore();
  const [isRetrying, setIsRetrying] = useState(false);

  const handleRetry = async () => {
    if (!retryWorkerInit) {
      window.location.reload();
      return;
    }

    setIsRetrying(true);
    try {
      await retryWorkerInit();
      clearWorkerError();
    } catch {
      // Error will be shown again via the store
    } finally {
      setIsRetrying(false);
    }
  };

  const handleDismiss = () => {
    clearWorkerError();
  };

  return (
    <Modal
      isOpen={hasWorkerError}
      onClose={handleDismiss}
      title="Graph Engine Initialization Failed"
      size="medium"
      closeOnOverlayClick={false}
    >
      <ModalBody>
        <div className="flex items-start gap-4">
          <div className="flex-shrink-0">
            <AlertTriangle className="h-8 w-8 text-amber-500" />
          </div>
          <div className="flex-1">
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              {errorMessage || 'The graph visualization engine failed to initialize.'}
            </p>

            {errorDetails && (
              <div className="bg-gray-100 dark:bg-gray-700 rounded-md p-3 mb-4 font-mono text-sm text-gray-600 dark:text-gray-400">
                {errorDetails}
              </div>
            )}

            <div className="bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-800 rounded-md p-4">
              <h4 className="font-medium text-blue-800 dark:text-blue-200 mb-2">
                Browser Compatibility Requirements
              </h4>
              <ul className="list-disc list-inside text-sm text-blue-700 dark:text-blue-300 space-y-1">
                <li>
                  <strong>SharedArrayBuffer</strong> must be enabled (requires secure context)
                </li>
                <li>
                  <strong>Web Workers</strong> must be supported and not blocked
                </li>
                <li>
                  Page must be served over <strong>HTTPS</strong> or from <strong>localhost</strong>
                </li>
                <li>
                  Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy headers must be set
                </li>
              </ul>
            </div>

            <div className="mt-4 text-sm text-gray-500 dark:text-gray-400">
              <p className="mb-2">
                The graph will display with limited functionality. Some features like smooth physics animations may not work.
              </p>
              <a
                href="https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SharedArrayBuffer#security_requirements"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-blue-600 dark:text-blue-400 hover:underline"
              >
                Learn more about SharedArrayBuffer requirements
                <ExternalLink className="h-3 w-3" />
              </a>
            </div>
          </div>
        </div>
      </ModalBody>
      <ModalFooter>
        <button
          type="button"
          className="rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-700 hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600"
          onClick={handleDismiss}
        >
          Continue Anyway
        </button>
        <button
          type="button"
          className="rounded-lg bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center gap-2"
          onClick={handleRetry}
          disabled={isRetrying}
        >
          {isRetrying ? (
            <>
              <RefreshCw className="h-4 w-4 animate-spin" />
              Retrying...
            </>
          ) : (
            <>
              <RefreshCw className="h-4 w-4" />
              Retry Initialization
            </>
          )}
        </button>
      </ModalFooter>
    </Modal>
  );
};
