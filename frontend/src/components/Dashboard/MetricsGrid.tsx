import React from 'react';
import { AlertCircle, Shield, Activity, Network, Users, Clock } from 'lucide-react';

const metrics = [
  { label: 'Events/sec', value: '1,432', icon: Activity, change: '+12.5%', color: 'blue' },
  { label: 'Critical Alerts', value: '23', icon: AlertCircle, change: '-5.2%', color: 'red' },
  { label: 'Threats Blocked', value: '1,284', icon: Shield, change: '+8.1%', color: 'green' },
  { label: 'Active Users', value: '847', icon: Users, change: '+3.2%', color: 'purple' },
  { label: 'Network Load', value: '76%', icon: Network, change: '+2.4%', color: 'orange' },
  { label: 'Avg Response', value: '1.2s', icon: Clock, change: '-1.8%', color: 'indigo' },
];

export function MetricsGrid() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {metrics.map((metric) => (
        <div key={metric.label} className="bg-white rounded-lg p-6 shadow-sm">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">{metric.label}</p>
              <p className="text-2xl font-semibold mt-1">{metric.value}</p>
            </div>
            <div className={`rounded-full p-3 bg-${metric.color}-50`}>
              <metric.icon className={`w-6 h-6 text-${metric.color}-500`} />
            </div>
          </div>
          <div className="flex items-center mt-4">
            <span className={`text-sm ${metric.change.startsWith('+') ? 'text-green-500' : 'text-red-500'}`}>
              {metric.change}
            </span>
            <span className="text-sm text-gray-500 ml-2">vs last hour</span>
          </div>
        </div>
      ))}
    </div>
  );
}