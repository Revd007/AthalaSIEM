import React from 'react';
import { Clock, Server, Shield, AlertTriangle } from 'lucide-react';

interface LogEntry {
  id: string;
  timestamp: string;
  source: string;
  message: string;
  severity: 'info' | 'warning' | 'critical';
  type: 'system' | 'security' | 'network';
}

const recentLogs: LogEntry[] = [
  {
    id: '1',
    timestamp: new Date().toISOString(),
    source: 'Firewall',
    message: 'Blocked suspicious connection from 192.168.1.100',
    severity: 'warning',
    type: 'security'
  },
  {
    id: '2',
    timestamp: new Date().toISOString(),
    source: 'IDS',
    message: 'Possible SQL injection attempt detected',
    severity: 'critical',
    type: 'security'
  }
];

export function LogStream() {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">Live Security Events</h3>
        <div className="flex items-center space-x-2">
          <div className="flex items-center">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse mr-2"></div>
            <span className="text-sm text-gray-500">Live</span>
          </div>
        </div>
      </div>
      <div className="space-y-2">
        {recentLogs.map(log => (
          <div key={log.id} className="flex items-start space-x-3 p-2 bg-gray-50 dark:bg-gray-700 rounded">
            <div className="flex-shrink-0">
              {log.severity === 'critical' ? (
                <AlertTriangle className="h-5 w-5 text-red-500" />
              ) : (
                <Shield className="h-5 w-5 text-yellow-500" />
              )}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium">{log.message}</p>
              <div className="flex items-center mt-1 text-xs text-gray-500">
                <Clock className="h-3 w-3 mr-1" />
                <span>{new Date(log.timestamp).toLocaleTimeString()}</span>
                <span className="mx-1">•</span>
                <span>{log.source}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}