import React from 'react';
import { FileText, AlertTriangle, Clock, Check } from 'lucide-react';

interface FIMEvent {
  id: string;
  path: string;
  type: 'modified' | 'created' | 'deleted' | 'permission_changed';
  timestamp: string;
  hash: string;
  user: string;
  severity: 'low' | 'medium' | 'high';
}

const mockFIMEvents: FIMEvent[] = [
  {
    id: '1',
    path: '/etc/passwd',
    type: 'modified',
    timestamp: '2024-03-15T10:30:00Z',
    hash: 'sha256:abc123...',
    user: 'root',
    severity: 'high',
  },
  {
    id: '2',
    path: '/var/www/html/index.php',
    type: 'modified',
    timestamp: '2024-03-15T10:25:00Z',
    hash: 'sha256:def456...',
    user: 'www-data',
    severity: 'medium',
  },
];

const severityColors = {
  low: 'blue',
  medium: 'yellow',
  high: 'red',
};

export function FIMMonitoring() {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <FileText className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">File Integrity Monitoring</h2>
        </div>
        <div className="flex space-x-2">
          <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm flex items-center">
            <Check className="h-4 w-4 mr-1" />
            Monitoring Active
          </span>
        </div>
      </div>

      <div className="space-y-4">
        {mockFIMEvents.map((event) => (
          <div key={event.id} className="border dark:border-gray-700 rounded-lg p-4">
            <div className="flex items-start justify-between">
              <div>
                <div className="flex items-center space-x-2">
                  <AlertTriangle className={`h-5 w-5 text-${severityColors[event.severity]}-500`} />
                  <span className="font-medium text-gray-900 dark:text-white">{event.path}</span>
                  <span className={`px-2 py-1 text-xs rounded-full bg-${severityColors[event.severity]}-100 text-${severityColors[event.severity]}-800`}>
                    {event.severity}
                  </span>
                </div>
                <div className="mt-2 text-sm text-gray-600 dark:text-gray-300">
                  <div className="flex items-center space-x-4">
                    <span className="flex items-center">
                      <Clock className="h-4 w-4 mr-1" />
                      {new Date(event.timestamp).toLocaleString()}
                    </span>
                    <span>Type: {event.type}</span>
                    <span>User: {event.user}</span>
                  </div>
                </div>
                <div className="mt-1 text-xs text-gray-500">
                  Hash: {event.hash}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}