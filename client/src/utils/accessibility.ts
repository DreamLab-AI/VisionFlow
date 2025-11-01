

import { useEffect, useRef, useCallback } from 'react';


export function useFocusTrap(isActive: boolean) {
  const containerRef = useRef<HTMLElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!isActive || !containerRef.current) return;

    
    previousFocusRef.current = document.activeElement as HTMLElement;

    
    const getFocusableElements = () => {
      if (!containerRef.current) return [];
      
      const focusableSelectors = [
        'a[href]:not([disabled])',
        'button:not([disabled])',
        'textarea:not([disabled])',
        'input:not([disabled])',
        'select:not([disabled])',
        '[tabindex]:not([tabindex="-1"])',
      ].join(',');

      return Array.from(
        containerRef.current.querySelectorAll<HTMLElement>(focusableSelectors)
      ).filter(el => !el.hasAttribute('aria-hidden'));
    };

    
    const focusableElements = getFocusableElements();
    if (focusableElements.length > 0) {
      focusableElements[0].focus();
    }

    
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Tab' || !containerRef.current) return;

      const focusableElements = getFocusableElements();
      if (focusableElements.length === 0) return;

      const currentIndex = focusableElements.findIndex(
        el => el === document.activeElement
      );

      if (e.shiftKey) {
        
        if (currentIndex <= 0) {
          e.preventDefault();
          focusableElements[focusableElements.length - 1].focus();
        }
      } else {
        
        if (currentIndex === focusableElements.length - 1) {
          e.preventDefault();
          focusableElements[0].focus();
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      
      
      if (previousFocusRef.current && previousFocusRef.current.focus) {
        previousFocusRef.current.focus();
      }
    };
  }, [isActive]);

  return containerRef;
}


class ScreenReaderAnnouncer {
  private container: HTMLElement | null = null;
  private timeout: NodeJS.Timeout | null = null;

  constructor() {
    this.createContainer();
  }

  private createContainer() {
    if (typeof document === 'undefined') return;

    this.container = document.createElement('div');
    this.container.setAttribute('role', 'status');
    this.container.setAttribute('aria-live', 'polite');
    this.container.setAttribute('aria-atomic', 'true');
    this.container.className = 'sr-only';
    this.container.style.position = 'absolute';
    this.container.style.width = '1px';
    this.container.style.height = '1px';
    this.container.style.padding = '0';
    this.container.style.margin = '-1px';
    this.container.style.overflow = 'hidden';
    this.container.style.clip = 'rect(0, 0, 0, 0)';
    this.container.style.whiteSpace = 'nowrap';
    this.container.style.border = '0';
    
    document.body.appendChild(this.container);
  }

  announce(message: string, priority: 'polite' | 'assertive' = 'polite') {
    if (!this.container) return;

    
    if (this.timeout) {
      clearTimeout(this.timeout);
    }

    
    this.container.setAttribute('aria-live', priority);

    
    this.container.textContent = '';
    
    
    this.timeout = setTimeout(() => {
      if (this.container) {
        this.container.textContent = message;
      }
    }, 100);
  }

  destroy() {
    if (this.timeout) {
      clearTimeout(this.timeout);
    }
    if (this.container && this.container.parentNode) {
      this.container.parentNode.removeChild(this.container);
    }
  }
}

// Singleton instance
let announcerInstance: ScreenReaderAnnouncer | null = null;


export function getAnnouncer(): ScreenReaderAnnouncer {
  if (!announcerInstance && typeof window !== 'undefined') {
    announcerInstance = new ScreenReaderAnnouncer();
  }
  return announcerInstance!;
}


export function useAnnounce() {
  const announce = useCallback((message: string, priority?: 'polite' | 'assertive') => {
    const announcer = getAnnouncer();
    if (announcer) {
      announcer.announce(message, priority);
    }
  }, []);

  return announce;
}


export function useRovingTabIndex<T extends HTMLElement>(
  items: T[],
  activeIndex: number,
  orientation: 'horizontal' | 'vertical' = 'horizontal'
) {
  useEffect(() => {
    items.forEach((item, index) => {
      if (item) {
        item.setAttribute('tabindex', index === activeIndex ? '0' : '-1');
      }
    });
  }, [items, activeIndex]);

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent<T>) => {
      const { key } = event;
      let newIndex = activeIndex;

      if (orientation === 'horizontal') {
        if (key === 'ArrowRight') {
          newIndex = (activeIndex + 1) % items.length;
        } else if (key === 'ArrowLeft') {
          newIndex = (activeIndex - 1 + items.length) % items.length;
        }
      } else {
        if (key === 'ArrowDown') {
          newIndex = (activeIndex + 1) % items.length;
        } else if (key === 'ArrowUp') {
          newIndex = (activeIndex - 1 + items.length) % items.length;
        }
      }

      if (key === 'Home') {
        newIndex = 0;
      } else if (key === 'End') {
        newIndex = items.length - 1;
      }

      if (newIndex !== activeIndex) {
        event.preventDefault();
        items[newIndex]?.focus();
        return newIndex;
      }

      return activeIndex;
    },
    [activeIndex, items, orientation]
  );

  return { handleKeyDown };
}


export function useId(prefix = 'id') {
  const ref = useRef<string>();
  
  if (!ref.current) {
    ref.current = `${prefix}-${Math.random().toString(36).substr(2, 9)}`;
  }
  
  return ref.current;
}


export const ariaLabels = {
  
  mainNavigation: 'Main navigation',
  breadcrumb: 'Breadcrumb navigation',
  pagination: 'Pagination navigation',
  
  
  close: 'Close',
  menu: 'Open menu',
  settings: 'Open settings',
  search: 'Search',
  submit: 'Submit',
  cancel: 'Cancel',
  save: 'Save changes',
  delete: 'Delete',
  edit: 'Edit',
  
  
  required: 'Required field',
  error: 'Error',
  success: 'Success',
  
  
  loading: 'Loading...',
  loadingMore: 'Loading more items...',
  
  
  expand: 'Expand',
  collapse: 'Collapse',
  toggleOn: 'Toggle on',
  toggleOff: 'Toggle off',
  
  
  sortAscending: 'Sort ascending',
  sortDescending: 'Sort descending',
  
  
  online: 'Online',
  offline: 'Offline',
  connected: 'Connected',
  disconnected: 'Disconnected',
};


export const focusRingClasses = 'focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background';


export const srOnlyClasses = 'absolute w-px h-px p-0 -m-px overflow-hidden whitespace-nowrap border-0';