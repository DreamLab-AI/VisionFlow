import React, { useState, useMemo } from 'react';
import { useSelectiveSetting, useSelectiveSettings, useSettingSetter } from '@/hooks/useSelectiveSettingsStore';
import { Card, CardHeader, CardTitle, CardContent } from '@/features/design-system/components/Card';
import { Button } from '@/features/design-system/components/Button';
import { Input } from '@/features/design-system/components/Input';
import { Badge } from '@/features/design-system/components/Badge';
import { ScrollArea } from '@/features/design-system/components/ScrollArea';
import { Bot, Send, Mic, Settings, Zap, Brain } from 'lucide-react';
import { createLogger } from '@/utils/logger';

const logger = createLogger('AIAssistant');

interface AIAssistantProps {
  className?: string;
}

interface ChatMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  suggestions?: string[];
}

export const AIAssistant: React.FC<AIAssistantProps> = ({ className }) => {
  const { set } = useSettingSetter();
  const [message, setMessage] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Subscribe only to AI-related settings
  const aiSettings = useSelectiveSettings({
    enabled: 'ai.enabled',
    model: 'ai.model.selected',
    maxTokens: 'ai.model.maxTokens',
    temperature: 'ai.model.temperature',
    voiceEnabled: 'ai.voice.enabled',
    voiceLanguage: 'ai.voice.language',
    suggestions: 'ai.suggestions.enabled',
    contextAware: 'ai.context.enabled',
    privacy: 'ai.privacy.mode'
  });
  
  // Mock chat messages
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([
    {
      id: '1',
      type: 'assistant',
      content: 'Hello! I\'m your AI assistant. I can help you analyze data, create visualizations, manage settings, and answer questions about your workflow.',
      timestamp: new Date(Date.now() - 5 * 60 * 1000),
      suggestions: ['Show me my data', 'Create a visualization', 'Export reports']
    }
  ]);
  
  const quickActions = useMemo(() => [
    { label: 'Analyze Data', icon: <Brain size={16} />, prompt: 'Analyze my current dataset and show insights' },
    { label: 'Create Chart', icon: <Zap size={16} />, prompt: 'Create a visualization from my data' },
    { label: 'Export Data', icon: <Settings size={16} />, prompt: 'Help me export my data in the best format' },
    { label: 'Optimize Performance', icon: <Zap size={16} />, prompt: 'Check system performance and suggest optimizations' }
  ], []);
  
  const handleSendMessage = async () => {
    if (!message.trim() || isProcessing) return;
    
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: message,
      timestamp: new Date()
    };
    
    setChatHistory(prev => [...prev, userMessage]);
    setMessage('');
    setIsProcessing(true);
    
    // Simulate AI processing
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    const aiResponse: ChatMessage = {
      id: (Date.now() + 1).toString(),
      type: 'assistant',
      content: `I understand you want to "${message}". Let me help you with that. Based on your current settings and data, here are some recommendations...`,
      timestamp: new Date(),
      suggestions: ['Tell me more', 'Show examples', 'Apply changes']
    };
    
    setChatHistory(prev => [...prev, aiResponse]);
    setIsProcessing(false);
    
    logger.info('AI message processed', { userMessage: message });
  };
  
  const handleQuickAction = (prompt: string) => {
    setMessage(prompt);
    handleSendMessage();
  };
  
  const handleSuggestion = (suggestion: string) => {
    setMessage(suggestion);
  };
  
  if (!aiSettings.enabled) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Bot size={20} />
            AI Assistant
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <Bot size={48} className="mx-auto mb-4 text-gray-400" />
            <p className="text-muted-foreground">AI Assistant is disabled</p>
            <p className="text-sm text-muted-foreground mt-2">
              Enable AI assistance in settings to get intelligent help and insights
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Bot size={20} />
            AI Assistant
            <Badge className="bg-green-100 text-green-800">Online</Badge>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline">{aiSettings.model}</Badge>
            {aiSettings.voiceEnabled && (
              <Badge className="bg-blue-100 text-blue-800">
                Voice
              </Badge>
            )}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Quick Actions */}
        <div>
          <h3 className="text-sm font-medium mb-2">Quick Actions</h3>
          <div className="grid grid-cols-2 gap-2">
            {quickActions.map((action, index) => (
              <Button
                key={index}
                variant="outline"
                size="sm"
                onClick={() => handleQuickAction(action.prompt)}
                className="h-auto p-3 flex flex-col items-start gap-1"
              >
                <div className="flex items-center gap-2">
                  {action.icon}
                  <span className="text-xs font-medium">{action.label}</span>
                </div>
              </Button>
            ))}
          </div>
        </div>
        
        {/* Chat History */}
        <div>
          <ScrollArea className="h-[300px] border rounded p-3">
            <div className="space-y-4">
              {chatHistory.map((msg) => (
                <div key={msg.id} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[80%] rounded-lg p-3 ${
                    msg.type === 'user' 
                      ? 'bg-blue-500 text-white' 
                      : 'bg-gray-100 text-gray-900'
                  }`}>
                    <p className="text-sm">{msg.content}</p>
                    <p className="text-xs opacity-70 mt-1">
                      {msg.timestamp.toLocaleTimeString()}
                    </p>
                    
                    {msg.suggestions && msg.suggestions.length > 0 && (
                      <div className="mt-2 space-y-1">
                        {msg.suggestions.map((suggestion, index) => (
                          <Button
                            key={index}
                            size="sm"
                            variant="ghost"
                            onClick={() => handleSuggestion(suggestion)}
                            className="h-6 px-2 text-xs bg-white/20 hover:bg-white/30"
                          >
                            {suggestion}
                          </Button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              
              {isProcessing && (
                <div className="flex justify-start">
                  <div className="bg-gray-100 rounded-lg p-3">
                    <div className="flex items-center gap-2">
                      <div className="animate-spin w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full" />
                      <span className="text-sm text-muted-foreground">AI is thinking...</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>
        </div>
        
        {/* Message Input */}
        <div className="flex gap-2">
          <div className="flex-1 relative">
            <Input
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Ask me anything about your data or workflow..."
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              disabled={isProcessing}
            />
          </div>
          {aiSettings.voiceEnabled && (
            <Button size="sm" variant="outline" disabled={isProcessing}>
              <Mic size={16} />
            </Button>
          )}
          <Button 
            size="sm" 
            onClick={handleSendMessage}
            disabled={!message.trim() || isProcessing}
          >
            <Send size={16} />
          </Button>
        </div>
        
        {/* AI Status */}
        <div className="flex items-center justify-between text-xs text-muted-foreground pt-2 border-t">
          <div className="flex items-center gap-4">
            <span>Model: {aiSettings.model}</span>
            <span>Temperature: {aiSettings.temperature}</span>
            {aiSettings.contextAware && (
              <Badge variant="secondary" className="text-xs">
                Context-aware
              </Badge>
            )}
          </div>
          <span>Privacy: {aiSettings.privacy}</span>
        </div>
      </CardContent>
    </Card>
  );
};

export default AIAssistant;