// Simple toast notification system
export interface ToastOptions {
  duration?: number;
  type?: 'success' | 'error' | 'warning' | 'info';
}

class ToastManager {
  private container: HTMLDivElement | null = null;

  private ensureContainer() {
    if (!this.container && typeof document !== 'undefined') {
      this.container = document.getElementById('toast-container') as HTMLDivElement;
      
      if (!this.container) {
        this.container = document.createElement('div');
        this.container.id = 'toast-container';
        this.container.style.cssText = `
          position: fixed;
          top: 20px;
          right: 20px;
          z-index: 9999;
          display: flex;
          flex-direction: column;
          gap: 10px;
          pointer-events: none;
        `;
        document.body.appendChild(this.container);
      }
    }
    return this.container;
  }

  show(message: string, options: ToastOptions = {}) {
    const container = this.ensureContainer();
    if (!container) return;

    const { duration = 3000, type = 'info' } = options;

    const toast = document.createElement('div');
    toast.style.cssText = `
      padding: 12px 20px;
      border-radius: 6px;
      background: ${this.getBackgroundColor(type)};
      color: white;
      font-size: 14px;
      font-family: system-ui, -apple-system, sans-serif;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
      pointer-events: auto;
      cursor: pointer;
      animation: slideIn 0.3s ease;
      max-width: 350px;
      word-wrap: break-word;
    `;
    
    toast.textContent = message;
    toast.onclick = () => this.remove(toast);
    
    container.appendChild(toast);

    // Auto remove after duration
    setTimeout(() => this.remove(toast), duration);
  }

  private remove(toast: HTMLDivElement) {
    toast.style.animation = 'slideOut 0.3s ease';
    setTimeout(() => {
      toast.remove();
    }, 300);
  }

  private getBackgroundColor(type: string): string {
    switch (type) {
      case 'success':
        return '#10b981'; // green
      case 'error':
        return '#ef4444'; // red
      case 'warning':
        return '#f59e0b'; // yellow
      case 'info':
      default:
        return '#3b82f6'; // blue
    }
  }

  success(message: string, options?: Omit<ToastOptions, 'type'>) {
    this.show(message, { ...options, type: 'success' });
  }

  error(message: string, options?: Omit<ToastOptions, 'type'>) {
    this.show(message, { ...options, type: 'error' });
  }

  warning(message: string, options?: Omit<ToastOptions, 'type'>) {
    this.show(message, { ...options, type: 'warning' });
  }

  info(message: string, options?: Omit<ToastOptions, 'type'>) {
    this.show(message, { ...options, type: 'info' });
  }
}

// Create singleton instance
export const toast = new ToastManager();

// Add animations
if (typeof document !== 'undefined' && !document.getElementById('toast-animations')) {
  const style = document.createElement('style');
  style.id = 'toast-animations';
  style.textContent = `
    @keyframes slideIn {
      from {
        transform: translateX(100%);
        opacity: 0;
      }
      to {
        transform: translateX(0);
        opacity: 1;
      }
    }
    
    @keyframes slideOut {
      from {
        transform: translateX(0);
        opacity: 1;
      }
      to {
        transform: translateX(100%);
        opacity: 0;
      }
    }
  `;
  document.head.appendChild(style);
}