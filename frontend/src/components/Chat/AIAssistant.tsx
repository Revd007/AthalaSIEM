import React, { useState } from 'react';
import { MessageSquare, X, Maximize2, Minimize2 } from 'lucide-react';

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'ai';
  timestamp: Date;
}

export function AIAssistant() {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');

  const handleSend = () => {
    if (!input.trim()) return;

    const newMessage: Message = {
      id: crypto.randomUUID(),
      content: input,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, newMessage]);
    setInput('');

    // Simulate AI response
    setTimeout(() => {
      const aiResponse: Message = {
        id: crypto.randomUUID(),
        content: "I'm analyzing your request. How can I assist you with security analysis?",
        sender: 'ai',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, aiResponse]);
    }, 1000);
  };

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-4 right-4 p-4 bg-blue-500 text-white rounded-full shadow-lg hover:bg-blue-600"
      >
        <MessageSquare className="h-6 w-6" />
      </button>
    );
  }

  return (
    <div className={`fixed right-4 bottom-4 w-96 bg-white rounded-lg shadow-xl transition-all duration-200 ${
      isMinimized ? 'h-14' : 'h-[500px]'
    }`}>
      <div className="flex items-center justify-between p-4 border-b">
        <h3 className="font-medium">Security Assistant</h3>
        <div className="flex items-center space-x-2">
          {isMinimized ? (
            <Maximize2
              className="h-4 w-4 cursor-pointer text-gray-500 hover:text-gray-700"
              onClick={() => setIsMinimized(false)}
            />
          ) : (
            <Minimize2
              className="h-4 w-4 cursor-pointer text-gray-500 hover:text-gray-700"
              onClick={() => setIsMinimized(true)}
            />
          )}
          <X
            className="h-4 w-4 cursor-pointer text-gray-500 hover:text-gray-700"
            onClick={() => setIsOpen(false)}
          />
        </div>
      </div>

      {!isMinimized && (
        <>
          <div className="h-[380px] overflow-y-auto p-4 space-y-4">
            {messages.map(message => (
              <div
                key={message.id}
                className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`max-w-[80%] rounded-lg p-3 ${
                  message.sender === 'user'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 text-gray-900'
                }`}>
                  {message.content}
                </div>
              </div>
            ))}
          </div>

          <div className="p-4 border-t">
            <div className="flex space-x-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                placeholder="Ask about security analysis..."
                className="flex-1 p-2 border rounded"
              />
              <button
                onClick={handleSend}
                className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
              >
                Send
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}