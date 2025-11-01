export interface OnboardingStep {
  id: string;
  title: string;
  description: string;
  target?: string; 
  position?: 'top' | 'bottom' | 'left' | 'right' | 'center';
  action?: () => void | Promise<void>;
  skipable?: boolean;
  nextButtonText?: string;
  prevButtonText?: string;
}

export interface OnboardingFlow {
  id: string;
  name: string;
  description: string;
  steps: OnboardingStep[];
  completionKey?: string; 
}

export interface OnboardingState {
  isActive: boolean;
  currentFlow: OnboardingFlow | null;
  currentStepIndex: number;
  completedFlows: string[];
}