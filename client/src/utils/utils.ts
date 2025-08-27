/**
 * Check if a value is defined (not undefined and not null)
 */
export function isDefined<T>(value: T | undefined | null): value is T {
  return value !== undefined && value !== null;
}

/**
 * Debounce a function call
 */
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: ReturnType<typeof setTimeout> | null = null;
  
  return function(...args: Parameters<T>) {
    if (timeout) {
      clearTimeout(timeout);
    }
    
    timeout = setTimeout(() => {
      func(...args);
    }, wait);
  };
}

/**
 * Truncate a string to the specified length
 */
export function truncate(str: string, length: number): string {
  if (str.length <= length) {
    return str;
  }
  
  return str.slice(0, length) + '...';
}