import React from 'react';
import { ClipboardList, CheckCircle, XCircle, Clock } from 'lucide-react';

interface ChangeRequest {
  id: string;
  title: string;
  type: 'emergency' | 'standard' | 'normal';
  status: 'pending' | 'approved' | 'rejected' | 'implemented';
  requester: string;
  dateSubmitted: string;
  implementation: string;
  risk: 'low' | 'medium' | 'high';
  approvers: string[];
  description: string;
}

const mockChanges: ChangeRequest[] = [
  {
    id: 'CR-001',
    title: 'Firewall Rule Update',
    type: 'standard',
    status: 'pending',
    requester: 'John Doe',
    dateSubmitted: '2024-03-15',
    implementation: '2024-03-20',
    risk: 'medium',
    approvers: ['Security Team', 'Network Team'],
    description: 'Update firewall rules to accommodate new application servers',
  },
  {
    id: 'CR-002',
    title: 'Emergency Patch Deployment',
    type: 'emergency',
    status: 'approved',
    requester: 'Jane Smith',
    dateSubmitted: '2024-03-14',
    implementation: '2024-03-14',
    risk: 'high',
    approvers: ['Security Team', 'System Admin'],
    description: 'Deploy critical security patch to address zero-day vulnerability',
  },
];

const statusIcons = {
  pending: Clock,
  approved: CheckCircle,
  rejected: XCircle,
  implemented: CheckCircle,
};

const statusColors = {
  pending: 'yellow',
  approved: 'green',
  rejected: 'red',
  implemented: 'blue',
};

export function ChangeManagement() {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <ClipboardList className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Change Management</h2>
        </div>
        <button className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600">
          New Change Request
        </button>
      </div>

      <div className="space-y-6">
        {mockChanges.map((change) => {
          const StatusIcon = statusIcons[change.status];
          const color = statusColors[change.status];

          return (
            <div key={change.id} className="border dark:border-gray-700 rounded-lg p-4">
              <div className="flex items-start justify-between">
                <div>
                  <div className="flex items-center space-x-2">
                    <span className="font-medium text-gray-900 dark:text-white">{change.id}</span>
                    <span className={`px-2 py-1 text-xs rounded-full bg-${color}-100 text-${color}-800`}>
                      {change.status}
                    </span>
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      change.type === 'emergency' ? 'bg-red-100 text-red-800' : 'bg-blue-100 text-blue-800'
                    }`}>
                      {change.type}
                    </span>
                  </div>
                  <h3 className="mt-1 font-medium text-gray-900 dark:text-white">{change.title}</h3>
                  <p className="mt-1 text-sm text-gray-500">{change.description}</p>
                </div>
                <StatusIcon className={`h-5 w-5 text-${color}-500`} />
              </div>

              <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-gray-500">Requester</p>
                  <p className="font-medium text-gray-900 dark:text-white">{change.requester}</p>
                </div>
                <div>
                  <p className="text-gray-500">Implementation Date</p>
                  <p className="font-medium text-gray-900 dark:text-white">{change.implementation}</p>
                </div>
                <div>
                  <p className="text-gray-500">Risk Level</p>
                  <p className={`font-medium ${
                    change.risk === 'high' ? 'text-red-600' : change.risk === 'medium' ? 'text-yellow-600' : 'text-green-600'
                  }`}>
                    {change.risk.toUpperCase()}
                  </p>
                </div>
                <div>
                  <p className="text-gray-500">Approvers</p>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {change.approvers.map((approver, index) => (
                      <span
                        key={index}
                        className="px-2 py-1 text-xs bg-gray-100 text-gray-700 rounded-full"
                      >
                        {approver}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}