import React from 'react';
import { Users, AlertTriangle, Clock, ArrowRight } from 'lucide-react';

interface UserActivityItem {
  id: string;
  user: string;
  action: string;
  timestamp: string;
  severity: 'low' | 'medium' | 'high';
  icon: string;
}

const recentActivities: UserActivityItem[] = [
  {
    id: '1',
    user: 'john.doe',
    action: 'Accessed sensitive files',
    timestamp: '2 minutes ago',
    severity: 'high',
    icon: '🔐',
  },
  {
    id: '2',
    user: 'admin.user',
    action: 'Modified firewall rules',
    timestamp: '5 minutes ago',
    severity: 'medium',
    icon: '🛡️',
  },
  {
    id: '3',
    user: 'jane.smith',
    action: 'Failed login attempt',
    timestamp: '10 minutes ago',
    severity: 'low',
    icon: '🔑',
  },
];

const severityColors = {
  low: 'bg-blue-100 text-blue-800',
  medium: 'bg-yellow-100 text-yellow-800',
  high: 'bg-red-100 text-red-800',
};

export function UserActivityOverview() {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Users className="h-5 w-5 text-blue-500" />
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Recent User Activity</h3>
        </div>
        <button className="text-sm text-blue-600 hover:text-blue-700 flex items-center">
          View All
          <ArrowRight className="h-4 w-4 ml-1" />
        </button>
      </div>

      <div className="space-y-3">
        {recentActivities.map((activity) => (
          <div
            key={activity.id}
            className="flex items-start space-x-3 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
          >
            <div className="flex-shrink-0 text-xl">{activity.icon}</div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between">
                <p className="text-sm font-medium text-gray-900 dark:text-white">
                  {activity.user}
                </p>
                <span className={`px-2 py-1 text-xs rounded-full ${severityColors[activity.severity]}`}>
                  {activity.severity}
                </span>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-300">{activity.action}</p>
              <div className="flex items-center mt-1 text-xs text-gray-500 dark:text-gray-400">
                <Clock className="h-3 w-3 mr-1" />
                {activity.timestamp}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}