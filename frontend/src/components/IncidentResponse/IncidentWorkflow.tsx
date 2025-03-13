import React from 'react';
import { AlertCircle, Clock, CheckCircle, User, MessageSquare } from 'lucide-react';

interface Incident {
  id: number;
  title: string;
  status: 'investigating' | 'containment' | 'resolved';
  priority: string;
  assignee: string;
  timeline: {
    time: string;
    action: string;
    user: string;
  }[];
}

const incidents: Incident[] = [
  {
    id: 1,
    title: 'Ransomware Attack Attempt',
    status: 'investigating' as const,
    priority: 'critical',
    assignee: 'Sarah Chen',
    timeline: [
      { time: '10:30 AM', action: 'Incident detected', user: 'System' },
      { time: '10:32 AM', action: 'Alert triggered', user: 'System' },
      { time: '10:35 AM', action: 'Investigation started', user: 'Sarah Chen' },
    ],
  },
  {
    id: 2,
    title: 'Data Exfiltration Detection',
    status: 'containment',
    priority: 'high',
    assignee: 'Mike Johnson',
    timeline: [
      { time: '09:15 AM', action: 'Unusual data transfer detected', user: 'System' },
      { time: '09:20 AM', action: 'Network segment isolated', user: 'Mike Johnson' },
    ],
  },
];

const statusColors = {
  investigating: 'yellow',
  containment: 'blue',
  resolved: 'green'
} as const;

export function IncidentWorkflow() {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Active Incidents</h2>
        <button className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600">
          Create Incident
        </button>
      </div>

      <div className="space-y-6">
        {incidents.map((incident: Incident) => (
          <div key={incident.id} className="border dark:border-gray-700 rounded-lg p-4">
            <div className="flex items-start justify-between">
              <div>
                <div className="flex items-center space-x-2">
                  <AlertCircle className="h-5 w-5 text-red-500" />
                  <h3 className="font-medium text-gray-900 dark:text-white">{incident.title}</h3>
                  <span className={`px-2 py-1 text-xs rounded-full bg-${statusColors[incident.status]}-100 text-${statusColors[incident.status]}-800`}>
                    {incident.status}
                  </span>
                </div>
                <div className="mt-2 flex items-center space-x-4 text-sm text-gray-500">
                  <div className="flex items-center">
                    <User className="h-4 w-4 mr-1" />
                    {incident.assignee}
                  </div>
                  <div className="flex items-center">
                    <Clock className="h-4 w-4 mr-1" />
                    {incident.timeline[0].time}
                  </div>
                </div>
              </div>
              <button className="flex items-center px-3 py-1 text-sm border rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700">
                <MessageSquare className="h-4 w-4 mr-1" />
                Update
              </button>
            </div>

            <div className="mt-4">
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Timeline</h4>
              <div className="space-y-2">
                {incident.timeline.map((event, index) => (
                  <div key={index} className="flex items-start space-x-2 text-sm">
                    <div className="w-16 text-gray-500">{event.time}</div>
                    <div className="flex-1 text-gray-600 dark:text-gray-400">{event.action}</div>
                    <div className="text-gray-500">{event.user}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}