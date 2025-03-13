import React from 'react';
import { AlertCircle, Shield, AlertTriangle, Info, AlertOctagon } from 'lucide-react';

interface Alert {
  id: number;
  title: string;
  description: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  time: string;
}

const alerts: Alert[] = [
  {
    id: 1,
    title: 'Brute Force Attack Detected',
    severity: 'high' as const,
    time: '2 minutes ago',
    description: 'Multiple failed login attempts from IP 192.168.1.100',
  },
  {
    id: 2,
    title: 'Malware Detected',
    severity: 'critical',
    time: '15 minutes ago',
    description: 'Trojan detected in file upload: malicious.exe',
  },
  {
    id: 3,
    title: 'Unusual Network Activity',
    severity: 'medium',
    time: '1 hour ago',
    description: 'High volume of outbound traffic to unknown IP',
  },
];

const severityIcon = {
  low: Info,
  medium: AlertCircle,
  high: AlertTriangle,
  critical: AlertOctagon
} as const;

const severityColor = {
  low: 'blue',
  medium: 'yellow',
  high: 'orange',
  critical: 'red'
} as const;

export function RecentAlerts() {
  return (
    <div className="bg-white rounded-lg p-6 shadow-sm">
      <h2 className="text-lg font-semibold mb-4">Recent Alerts</h2>
      <div className="space-y-4">
        {alerts.map((alert: Alert) => {
          const Icon = severityIcon[alert.severity];
          const color = severityColor[alert.severity];
          
          return (
            <div
              key={alert.id}
              className={`flex items-start space-x-4 p-4 bg-${color}-50 rounded-lg`}
            >
              <Icon className={`w-5 h-5 text-${color}-500 mt-1`} />
              <div>
                <div className="flex items-center space-x-2">
                  <h3 className="font-medium">{alert.title}</h3>
                  <span className={`text-xs px-2 py-1 rounded-full bg-${color}-100 text-${color}-800`}>
                    {alert.severity}
                  </span>
                </div>
                <p className="text-sm text-gray-600 mt-1">{alert.description}</p>
                <p className="text-xs text-gray-500 mt-1">{alert.time}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}