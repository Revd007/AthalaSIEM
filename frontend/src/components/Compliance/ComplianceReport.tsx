import React from 'react';
import { CheckCircle, XCircle, AlertTriangle } from 'lucide-react';

interface ComplianceItem {
  standard: string;
  requirement: string;
  status: 'compliant' | 'non-compliant' | 'warning';
  lastCheck: string;
  details: string;
}

const complianceItems: ComplianceItem[] = [
  {
    standard: 'PCI DSS',
    requirement: 'Requirement 5: Protect systems against malware',
    status: 'compliant',
    lastCheck: '2024-03-15',
    details: 'Anti-virus software installed and updated regularly',
  },
  {
    standard: 'HIPAA',
    requirement: '164.312(a)(1) Access Control',
    status: 'non-compliant',
    lastCheck: '2024-03-14',
    details: 'Missing multi-factor authentication for remote access',
  },
  {
    standard: 'ISO 27001',
    requirement: 'A.12.2 Protection from malware',
    status: 'warning',
    lastCheck: '2024-03-13',
    details: 'Malware definitions need updating',
  },
];

const statusConfig = {
  compliant: { icon: CheckCircle, color: 'text-green-500', bg: 'bg-green-50' },
  'non-compliant': { icon: XCircle, color: 'text-red-500', bg: 'bg-red-50' },
  warning: { icon: AlertTriangle, color: 'text-yellow-500', bg: 'bg-yellow-50' }
} as const

export function ComplianceReport() {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">Compliance Status</h2>
      
      <div className="space-y-4">
        {complianceItems.map((item, index) => {
          const StatusIcon = statusConfig[item.status].icon;
          
          return (
            <div key={index} className="border dark:border-gray-700 rounded-lg p-4">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-2">
                    <span className="font-medium text-gray-900 dark:text-white">{item.standard}</span>
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${statusConfig[item.status].bg} ${statusConfig[item.status].color}`}>
                      <StatusIcon className="w-4 h-4 mr-1" />
                      {item.status}
                    </span>
                  </div>
                  <p className="mt-1 text-sm text-gray-600 dark:text-gray-300">{item.requirement}</p>
                  <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">{item.details}</p>
                  <p className="mt-1 text-xs text-gray-400 dark:text-gray-500">Last checked: {item.lastCheck}</p>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}