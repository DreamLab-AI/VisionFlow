
export const formatSettingName = (name: string): string => {
  
  const spacedName = name.replace(/([A-Z])/g, ' $1')
    
    .replace(/_/g, ' ')
    
    .trim();
  
  
  return spacedName
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};