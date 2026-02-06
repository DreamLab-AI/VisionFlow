import { useState, useCallback, useEffect } from 'react';
import { OnboardingState, OnboardingFlow, OnboardingStep } from '../types';

const COMPLETED_FLOWS_KEY = 'onboarding.completedFlows';

export function useOnboarding() {
  const [state, setState] = useState<OnboardingState>({
    isActive: false,
    currentFlow: null,
    currentStepIndex: 0,
    completedFlows: []
  });

  
  useEffect(() => {
    try {
      const stored = localStorage.getItem(COMPLETED_FLOWS_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed) && parsed.every((v: unknown) => typeof v === 'string')) {
          setState(prev => ({
            ...prev,
            completedFlows: parsed as string[]
          }));
        }
      }
    } catch (error) {
      console.error('Failed to load onboarding state:', error);
    }
  }, []);

  
  const startFlow = useCallback((flow: OnboardingFlow, forceRestart = false) => {
    if (!forceRestart && state.completedFlows.includes(flow.id)) {
      return false;
    }

    setState(prev => ({
      ...prev,
      isActive: true,
      currentFlow: flow,
      currentStepIndex: 0
    }));

    return true;
  }, [state.completedFlows]);

  
  const nextStep = useCallback(async () => {
    if (!state.currentFlow) return;

    const currentStep = state.currentFlow.steps[state.currentStepIndex];
    
    
    if (currentStep.action) {
      try {
        await currentStep.action();
      } catch (error) {
        console.error('Failed to execute step action:', error);
      }
    }

    if (state.currentStepIndex < state.currentFlow.steps.length - 1) {
      setState(prev => ({
        ...prev,
        currentStepIndex: prev.currentStepIndex + 1
      }));
    } else {
      
      completeFlow();
    }
  }, [state.currentFlow, state.currentStepIndex]);

  
  const prevStep = useCallback(() => {
    if (state.currentStepIndex > 0) {
      setState(prev => ({
        ...prev,
        currentStepIndex: prev.currentStepIndex - 1
      }));
    }
  }, [state.currentStepIndex]);

  
  const skipFlow = useCallback(() => {
    completeFlow();
  }, []);

  
  const completeFlow = useCallback(() => {
    if (!state.currentFlow) return;

    const flowId = state.currentFlow.id;
    const newCompletedFlows = [...state.completedFlows, flowId];

    
    try {
      localStorage.setItem(COMPLETED_FLOWS_KEY, JSON.stringify(newCompletedFlows));
    } catch (error) {
      console.error('Failed to save onboarding state:', error);
    }

    setState(prev => ({
      ...prev,
      isActive: false,
      currentFlow: null,
      currentStepIndex: 0,
      completedFlows: newCompletedFlows
    }));

    
    window.dispatchEvent(new CustomEvent('onboarding-completed', { 
      detail: { flowId } 
    }));
  }, [state.currentFlow, state.completedFlows]);

  
  const resetOnboarding = useCallback(() => {
    localStorage.removeItem(COMPLETED_FLOWS_KEY);
    setState({
      isActive: false,
      currentFlow: null,
      currentStepIndex: 0,
      completedFlows: []
    });
  }, []);

  
  const currentStep = state.currentFlow 
    ? state.currentFlow.steps[state.currentStepIndex]
    : null;

  const hasNextStep = state.currentFlow 
    ? state.currentStepIndex < state.currentFlow.steps.length - 1
    : false;

  const hasPrevStep = state.currentStepIndex > 0;

  return {
    ...state,
    currentStep,
    hasNextStep,
    hasPrevStep,
    startFlow,
    nextStep,
    prevStep,
    skipFlow,
    completeFlow,
    resetOnboarding
  };
}