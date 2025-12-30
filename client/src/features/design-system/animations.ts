export const animations = {
  transitions: {
    spring: {
      snappy: { type: 'spring' as const, stiffness: 300, damping: 30 },
      smooth: { type: 'spring' as const, stiffness: 100, damping: 20 },
      bounce: { type: 'spring' as const, stiffness: 150, damping: 15 },
    },
    easing: {
      easeIn: { ease: 'easeIn' as const, duration: 0.2 },
      easeOut: { ease: 'easeOut' as const, duration: 0.2 },
      easeInOut: { ease: 'easeInOut' as const, duration: 0.2 },
    },
  },
  variants: {
    scale: {
      initial: { scale: 0.95, opacity: 0 },
      animate: { scale: 1, opacity: 1 },
      exit: { scale: 0.95, opacity: 0 },
    },
    slideDown: {
      initial: { y: -20, opacity: 0 },
      animate: { y: 0, opacity: 1 },
      exit: { y: -20, opacity: 0 },
    },
    slideUp: {
      initial: { y: 20, opacity: 0 },
      animate: { y: 0, opacity: 1 },
      exit: { y: 20, opacity: 0 },
    },
    slideLeft: {
      initial: { x: 20, opacity: 0 },
      animate: { x: 0, opacity: 1 },
      exit: { x: 20, opacity: 0 },
    },
    slideRight: {
      initial: { x: -20, opacity: 0 },
      animate: { x: 0, opacity: 1 },
      exit: { x: -20, opacity: 0 },
    },
  },
};