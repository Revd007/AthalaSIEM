import React, { useState } from 'react';
import { Bell, X } from 'lucide-react';

interface Alert {
  id: string;
  title: string;
  description: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  timestamp: string;
  isRead: boolean;
}

const mockAlerts: Alert[] = [
  {
    id: '1',
    title: 'Critical Security Alert',
    description: 'Multiple failed login attempts detected',
    severity: 'critical',
    timestamp: new Date().toISOString(),
    isRead: false,
  },
  {
    id: '2',
    title: 'Suspicious Activity',
    description: 'Unusual network traffic detected',
    severity: 'high',
    timestamp: new Date().toISOString(),
    isRead: false,
  },
];

export function AlertNotifications() {
  const [isOpen, setIsOpen] = useState(false);
  const [alerts, setAlerts] = useState(mockAlerts);

  const unreadCount = alerts.filter(alert => !alert.isRead).length;

  const markAsRead = (alertId: string) => {
    setAlerts(alerts.map(alert => 
      alert.id === alertId ? { ...alert, isRead: true } : alert
    ));
  };

  return (
    <div className="relative">
      <button 
        className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 relative"
        onClick={() => setIsOpen(!isOpen)}
      >
        <Bell className="h-5 w-5 text-gray-500 dark:text-gray-400" />
        {unreadCount > 0 && (
          <span className="absolute top-0 right-0 h-4 w-4 bg-red-500 rounded-full text-xs text-white flex items-center justify-center">
            {unreadCount}
          </span>
        )}
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-96 bg-white dark:bg-gray-800 rounded-lg shadow-lg ring-1 ring-black ring-opacity-5">
          <div className="p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900 dark:text-white">Notifications</h3>
              <button 
                onClick={() => setIsOpen(false)}
                className="text-gray-400 hover:text-gray-500"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            
            <div className="space-y-4 max-h-96 overflow-y-auto">
              {alerts.map(alert => (
                <div 
                  key={alert.id}
                  className={`p-3 rounded-lg ${
                    !alert.isRead ? 'bg-blue-50 dark:bg-blue-900/20' : 'bg-gray-50 dark:bg-gray-700'
                  }`}
                  onClick={() => markAsRead(alert.id)}
                >
                  <div className="flex items-center justify-between">
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      alert.severity === 'critical' ? 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200' :
                      alert.severity === 'high' ? 'bg-orange-100 text-orange-800 dark:bg-orange-900/50 dark:text-orange-200' :
                      'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-200'
                    }`}>
                      {alert.severity}
                    </span>
                    <span className="text-xs text-gray-500">
                      {new Date(alert.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  <h4 className="font-medium text-gray-900 dark:text-white mt-2">
                    {alert.title}
                  </h4>
                  <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">
                    {alert.description}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
} 