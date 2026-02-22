import { describe, it, expect } from 'vitest';
import { cn } from '../classNameUtils';

describe('cn (classNameUtils)', () => {
  it('should merge simple class names', () => {
    const result = cn('px-2', 'py-1');
    expect(result).toContain('px-2');
    expect(result).toContain('py-1');
  });

  it('should handle conditional classes', () => {
    const isActive = true;
    const result = cn('base', isActive && 'active');
    expect(result).toContain('base');
    expect(result).toContain('active');
  });

  it('should filter out falsy values', () => {
    const result = cn('base', false, null, undefined, 0, 'end');
    expect(result).toContain('base');
    expect(result).toContain('end');
    expect(result).not.toContain('false');
    expect(result).not.toContain('null');
  });

  it('should merge conflicting tailwind classes (last wins)', () => {
    const result = cn('px-2', 'px-4');
    expect(result).toBe('px-4');
  });

  it('should handle empty input', () => {
    const result = cn();
    expect(result).toBe('');
  });

  it('should handle array input', () => {
    const result = cn(['px-2', 'py-1']);
    expect(result).toContain('px-2');
    expect(result).toContain('py-1');
  });
});
