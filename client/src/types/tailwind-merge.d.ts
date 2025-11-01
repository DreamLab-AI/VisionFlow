declare module 'tailwind-merge' {
  
  export function twMerge(...classLists: (string | undefined | null | false)[]): string;

  
  export function twMergeConfig(config: Record<string, unknown>): typeof twMerge;
}