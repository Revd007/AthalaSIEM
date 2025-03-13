import React from 'react';
import { Users, AlertTriangle, Clock, Search } from 'lucide-react';

interface UserActivity {
  id: string;
  user: string;
  action: string;
  resource: string;
  timestamp: string;
  status: 'normal' | 'suspicious' | 'blocked';
  details: string;
}

const activities: UserActivity[] = [
  {
    id: '1',
    user: 'john.doe',
    action: 'File Access',
    resource: '/sensitive/customer-data.xlsx',
    timestamp: '2024-03-15 10:30:45',
    status: 'suspicious',
    details: 'Unusual access pattern detected',
  },
  {
    id: '2',
    user: 'admin.user',
    action: 'Configuration Change',
    resource: 'Firewall Rules',
    timestamp: '2024-03-15 10:28:30',
    status: 'normal',
    details: 'Scheduled maintenance',
  },
  {
    id: '3',
    user: 'jane.smith',
    action: 'Database Query',
    resource: 'Customer Database',
    timestamp: '2024-03-15 10:25:15',
    status: 'blocked',
    details: 'Unauthorized access attempt',
  },
];

const statusColors = {
  normal: 'green',
  suspicious: 'yellow',
  blocked: 'red',
};

export function UserActivityMonitor() {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Users className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">User Activity Monitor</h2>
        </div>
        <div className="flex space-x-4">
          <div className="relative">
            <input
              type="text"
              placeholder="Search activities..."
              className="pl-10 pr-4 py-2 border rounded-lg"
            />
            <Search className="absolute left-3 top-2.5 h-5 w-5 text-gray-400" />
          </div>
          <button className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600">
            Export Log
          </button>
        </div>
      </div>

      <div className="space-y-4">
        {activities.map((activity) => (
          <div key={activity.id} className="border rounded-lg p-4">
            <div className="flex items-start justify-between">
              <div>
                <div className="flex items-center space-x-2">
                  <span className="font-medium text-gray-900 dark:text-white">{activity.user}</span>
                  <span className={`px-2 py-1 text-xs rounded-full bg-${statusColors[activity.status]}-100 text-${statusColors[activity.status]}-800`}>
                    {activity.status}
                  </span>
                </div>
                <p className="mt-1 text-sm text-gray-600">{activity.action} - {activity.resource}</p>
                <div className="mt-2 flex items-center text-sm text-gray-500">
                  <Clock className="h-4 w-4 mr-1" />
                  {activity.timestamp}
                </div>
              </div>
              {activity.status !== 'normal' && (
                <AlertTriangle className={`h-5 w-5 text-${statusColors[activity.status]}-500`} />
              )}
            </div>
            {activity.details && (
              <p className="mt-2 text-sm text-gray-500 border-t pt-2">{activity.details}</p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}