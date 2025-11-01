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
    <div className={`overflow-auto scroll-area ${className}`} style={{ ...style }}>
      {children}
    </div>
  );
};

// Add custom scrollbar styles
const scrollbarStyles = `
  .scroll-area::-webkit-scrollbar {
    display: none;
  }

  .scroll-area {
    -ms-overflow-style: none;  
    scrollbar-width: none;  
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