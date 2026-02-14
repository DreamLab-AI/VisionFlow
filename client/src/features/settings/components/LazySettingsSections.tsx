import React, { lazy } from 'react';

// Placeholder component for panels not yet implemented
const PlaceholderPanel: React.FC<{ name: string }> = ({ name }) => (
  <div className="p-4 text-center text-gray-400">
    <p>{name} — coming soon</p>
  </div>
);

// Lazy load heavy setting sections — panels that exist get lazy-loaded,
// missing panels use inline placeholders to avoid import errors.
export const LazyAdvancedSettings = lazy(() =>
  Promise.resolve({ default: () => <PlaceholderPanel name="Advanced Settings" /> })
);

export const LazyXRSettings = lazy(() =>
  Promise.resolve({ default: () => <PlaceholderPanel name="XR Settings" /> })
);

export const LazyVisualizationSettings = lazy(() =>
  Promise.resolve({ default: () => <PlaceholderPanel name="Visualisation Settings" /> })
);

export const LazyAISettings = lazy(() =>
  Promise.resolve({ default: () => <PlaceholderPanel name="AI Settings" /> })
);

export const LazySystemSettings = lazy(() =>
  Promise.resolve({ default: () => <PlaceholderPanel name="System Settings" /> })
);

// Simple inline loading spinner
const LoadingSpinner = () => <div className="animate-spin h-6 w-6 border-2 border-primary border-t-transparent rounded-full" />;

interface LazySettingSectionProps {
  component: React.LazyExoticComponent<React.ComponentType<any>>;
  props?: any;
}

export const LazySettingSection: React.FC<LazySettingSectionProps> = ({ component: Component, props = {} }) => {
  return (
    <React.Suspense
      fallback={
        <div className="flex items-center justify-center h-32">
          <LoadingSpinner />
        </div>
      }
    >
      <Component {...props} />
    </React.Suspense>
  );
};
