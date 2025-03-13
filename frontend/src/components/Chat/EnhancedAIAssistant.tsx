import React from 'react';
import { MessageSquare, X, Maximize2, Minimize2 } from 'lucide-react';
import { useChat } from '../../contexts/ChatContext';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';

export function EnhancedAIAssistant() {
  const { 
    messages, 
    addMessage, 
    isOpen, 
    toggleChat, 
    isMinimized, 
    toggleMinimize 
  } = useChat();

  if (!isOpen) {
    return (
      <button
        onClick={toggleChat}
        className="fixed bottom-4 right-4 p-4 bg-blue-500 text-white rounded-full shadow-lg hover:bg-blue-600 z-50"
      >
        <MessageSquare className="h-6 w-6" />
      </button>
    );
  }

  return (
    <div className={`fixed right-4 bottom-4 w-96 bg-white dark:bg-gray-800 rounded-lg shadow-xl transition-all duration-200 z-50 ${
      isMinimized ? 'h-14' : 'h-[500px]'
    }`}>
      <div className="flex items-center justify-between p-4 border-b dark:border-gray-700">
        <h3 className="font-medium text-gray-900 dark:text-white">Security Assistant</h3>
        <div className="flex items-center space-x-2">
          {isMinimized ? (
            <Maximize2
              className="h-4 w-4 cursor-pointer text-gray-500 hover:text-gray-700"
              onClick={toggleMinimize}
            />
          ) : (
            <Minimize2
              className="h-4 w-4 cursor-pointer text-gray-500 hover:text-gray-700"
              onClick={toggleMinimize}
            />
          )}
          <X
            className="h-4 w-4 cursor-pointer text-gray-500 hover:text-gray-700"
            onClick={toggleChat}
          />
        </div>
      </div>

      {!isMinimized && (
        <>
          <div className="h-[380px] overflow-y-auto p-4 space-y-4">
            {messages.map(message => (
              <ChatMessage key={message.id} message={message} />
            ))}
          </div>
          <ChatInput onSend={addMessage} />
        </>
      )}
    </div>
  );
}