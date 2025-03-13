import React from 'react';
import { Play, CheckCircle2, AlertTriangle, Clock, Settings } from 'lucide-react';

interface Step {
  action: string;
  status: 'completed' | 'in_progress' | 'pending';
}

interface Playbook {
  id: number;
  name: string;
  status: string;
  steps: Step[];
  triggers: string[];
}

const playbooks: Playbook[] = [
  {
    id: 1,
    name: 'Ransomware Response',
    status: 'active',
    steps: [
      { action: 'Isolate affected systems', status: 'completed' },
      { action: 'Block malicious IPs', status: 'completed' },
      { action: 'Scan for IOCs', status: 'in_progress' },
      { action: 'Restore from backup', status: 'pending' },
    ],
    triggers: ['Ransomware Detection', 'Mass File Encryption'],
  },
  {
    id: 2,
    name: 'Data Exfiltration Prevention',
    status: 'standby',
    steps: [
      { action: 'Block outbound traffic', status: 'pending' },
      { action: 'Analyze data transfers', status: 'pending' },
      { action: 'Revoke compromised credentials', status: 'pending' },
    ],
    triggers: ['Unusual Data Transfer', 'DLP Alert'],
  },
];

const statusIcons = {
  completed: CheckCircle2,
  in_progress: Clock,
  pending: AlertTriangle,
};

const statusColors = {
  completed: 'text-green-500',
  in_progress: 'text-blue-500',
  pending: 'text-gray-400',
};

export function AutomatedPlaybooks() {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Settings className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Automated Playbooks</h2>
        </div>
        <button className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 flex items-center">
          <Play className="h-4 w-4 mr-2" />
          Create Playbook
        </button>
      </div>

      <div className="space-y-6">
        {playbooks.map((playbook) => (
          <div key={playbook.id} className="border dark:border-gray-700 rounded-lg p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-medium text-gray-900 dark:text-white">{playbook.name}</h3>
              <span className={`px-2 py-1 text-xs rounded-full ${
                playbook.status === 'active' ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
              }`}>
                {playbook.status}
              </span>
            </div>

            <div className="space-y-3">
              {playbook.steps.map((step: Step, index) => {
                const Icon = statusIcons[step.status];
                return (
                  <div key={index} className="flex items-center space-x-3">
                    <Icon className={`h-5 w-5 ${statusColors[step.status]}`} />
                    <span className="text-sm text-gray-600 dark:text-gray-300">{step.action}</span>
                  </div>
                );
              })}
            </div>

            <div className="mt-4">
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Triggers:</h4>
              <div className="flex flex-wrap gap-2">
                {playbook.triggers.map((trigger, index) => (
                  <span
                    key={index}
                    className="px-2 py-1 text-xs bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 rounded-full"
                  >
                    {trigger}
                  </span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}