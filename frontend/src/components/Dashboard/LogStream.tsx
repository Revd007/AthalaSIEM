import React from 'react';
import { Clock, Server, Shield } from 'lucide-react';

interface LogEntry {
  id: number;
  timestamp: string;
  source: string;
  message: string;
  severity: 'low' | 'medium' | 'high';
}

const logs: LogEntry[] = [
  {
    id: 1,
    timestamp: '2024-03-15 10:30:45',
    source: 'Firewall',
    message: 'Blocked incoming connection from 203.0.113.1',
    severity: 'medium' as const,
  },
  {
    id: 2,
    timestamp: '2024-03-15 10:30:42',
    source: 'IDS',
    message: 'Signature match: SQL injection attempt',
    severity: 'high' as const,
  },
  {
    id: 3,
    timestamp: '2024-03-15 10:30:40',
    source: 'Authentication',
    message: 'Failed login attempt for user admin',
    severity: 'low' as const,
  },
];

const getSeverityClasses = (severity: LogEntry['severity']) => {
  const classes = {
    low: 'bg-green-100 text-green-800',
    medium: 'bg-yellow-100 text-yellow-800',
    high: 'bg-red-100 text-red-800'
  }[severity];
  
  return classes;
};

export function LogStream() {
  return (
    <div className="bg-white rounded-lg p-6 shadow-sm">
      <h2 className="text-lg font-semibold mb-4">Live Log Stream</h2>
      <div className="space-y-3">
        {logs.map((log) => (
          <div key={log.id} className="flex items-start space-x-3 p-3 bg-gray-50 rounded-lg">
            <Server className="w-5 h-5 text-gray-400 mt-1" />
            <div className="flex-1 min-w-0">
              <div className="flex items-center space-x-2">
                <span className="text-sm font-medium text-gray-900">{log.source}</span>
                <span className={`text-xs px-2 py-0.5 rounded-full ${getSeverityClasses(log.severity)}`}>
                  {log.severity}
                </span>
              </div>
              <p className="text-sm text-gray-600 mt-1">{log.message}</p>
              <div className="flex items-center space-x-2 mt-1">
                <Clock className="w-4 h-4 text-gray-400" />
                <span className="text-xs text-gray-500">{log.timestamp}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}