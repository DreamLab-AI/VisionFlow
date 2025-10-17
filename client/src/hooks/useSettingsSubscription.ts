import { useEffect, useState, useRef } from 'react';
import { settingsApi } from '../api/settingsApi';
import { createLogger } from '../utils/loggerConfig';

const logger = createLogger('useSettingsSubscription');

export function useSettingsSubscription<T>(
  path: string,
  defaultValue?: T
): T | undefined {
  const [value, setValue] = useState<T | undefined>(defaultValue);
  const isMountedRef = useRef(true);

  useEffect(() => {
    isMountedRef.current = true;

    // Initial fetch
    settingsApi
      .getSettingByPath(path)
      .then((fetchedValue) => {
        if (isMountedRef.current) {
          setValue(fetchedValue);
        }
      })
      .catch((error) => {
        logger.error(`Failed to fetch initial value for ${path}:`, error);
      });

    // Subscribe to changes
    const unsubscribe = settingsApi.subscribeToSettingChanges(
      path,
      (changedPath, newValue) => {
        if (isMountedRef.current) {
          logger.debug(`Setting ${changedPath} changed to:`, newValue);
          setValue(newValue);
        }
      }
    );

    return () => {
      isMountedRef.current = false;
      unsubscribe();
    };
  }, [path]);

  return value;
}

export function useSettingsBatch<T extends Record<string, any>>(
  paths: string[]
): T | null {
  const [values, setValues] = useState<T | null>(null);
  const isMountedRef = useRef(true);
  const pathsKey = paths.join(',');

  useEffect(() => {
    isMountedRef.current = true;

    // Initial fetch
    settingsApi
      .getSettingsByPaths(paths)
      .then((fetchedValues) => {
        if (isMountedRef.current) {
          setValues(fetchedValues as T);
        }
      })
      .catch((error) => {
        logger.error(`Failed to fetch batch settings:`, error);
      });

    // Subscribe to each path
    const unsubscribers = paths.map((path) =>
      settingsApi.subscribeToSettingChanges(path, (changedPath, newValue) => {
        if (isMountedRef.current) {
          setValues((prev) => ({
            ...prev,
            [changedPath]: newValue,
          } as T));
        }
      })
    );

    return () => {
      isMountedRef.current = false;
      unsubscribers.forEach((unsub) => unsub());
    };
  }, [pathsKey]);

  return values;
}

export function useCacheMetrics() {
  const [metrics, setMetrics] = useState(settingsApi.getCacheMetrics());

  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(settingsApi.getCacheMetrics());
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  return metrics;
}
