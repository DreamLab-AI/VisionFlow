import React from 'react';

interface ScrollAreaProps {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}

export const ScrollArea: React.FC<ScrollAreaProps> = ({ 
  children, 
  className = '', 
  style = {} 
}) => {
  return (
    <div 
      className={`overflow-auto ${className}`}
      style={{
        ...style,
        scrollbarWidth: 'thin',
        scrollbarColor: 'rgba(255, 255, 255, 0.2) rgba(255, 255, 255, 0.05)',
      }}
    >
      {children}
    </div>
  );
};

// Add custom scrollbar styles
const scrollbarStyles = `
  .scroll-area::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }
  
  .scroll-area::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
  }
  
  .scroll-area::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 4px;
  }
  
  .scroll-area::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.3);
  }
`;

// Inject styles if not already present
if (typeof document !== 'undefined' && !document.getElementById('scroll-area-styles')) {
  const styleElement = document.createElement('style');
  styleElement.id = 'scroll-area-styles';
  styleElement.innerHTML = scrollbarStyles;
  document.head.appendChild(styleElement);
}

export default ScrollArea;