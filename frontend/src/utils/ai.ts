import { AIResponse } from '../types/chat';

export async function processAIResponse(userInput: string): Promise<AIResponse> {
  // Simulate AI processing with security context awareness
  const securityKeywords = ['alert', 'threat', 'attack', 'vulnerability', 'incident'];
  const hasSecurityContext = securityKeywords.some(keyword => 
    userInput.toLowerCase().includes(keyword)
  );

  if (hasSecurityContext) {
    return {
      message: 'I detect a security-related query. Let me analyze that for you...',
      suggestions: [
        'View related alerts',
        'Check threat intel',
        'Run security scan'
      ],
      confidence: 0.95,
      context: {
        securityContext: 'security_incident',
        relatedAlerts: ['Alert-123', 'Alert-124']
      }
    };
  }

  return {
    message: 'How can I assist you with security analysis?',
    suggestions: [
      'Show recent alerts',
      'Security overview',
      'System status'
    ],
    confidence: 0.85
  };
}