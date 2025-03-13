import React, { createContext, useContext, useState, useCallback } from 'react';
import { Message, AIResponse } from '../types/chat';
import { processAIResponse } from '../utils/ai';

interface ChatContextType {
  messages: Message[];
  addMessage: (content: string) => void;
  clearChat: () => void;
  isOpen: boolean;
  toggleChat: () => void;
  isMinimized: boolean;
  toggleMinimize: () => void;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

export function ChatProvider({ children }: { children: React.ReactNode }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);

  const addMessage = useCallback(async (content: string) => {
    const userMessage: Message = {
      id: crypto.randomUUID(),
      content,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);

    // Process AI response
    const aiResponse = await processAIResponse(content);
    const aiMessage: Message = {
      id: crypto.randomUUID(),
      content: aiResponse.message,
      sender: 'ai',
      timestamp: new Date(),
      suggestions: aiResponse.suggestions
    };

    setMessages(prev => [...prev, aiMessage]);
  }, []);

  const clearChat = useCallback(() => {
    setMessages([]);
  }, []);

  const toggleChat = useCallback(() => {
    setIsOpen(prev => !prev);
  }, []);

  const toggleMinimize = useCallback(() => {
    setIsMinimized(prev => !prev);
  }, []);

  return (
    <ChatContext.Provider value={{
      messages,
      addMessage,
      clearChat,
      isOpen,
      toggleChat,
      isMinimized,
      toggleMinimize
    }}>
      {children}
    </ChatContext.Provider>
  );
}

export const useChat = () => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
};