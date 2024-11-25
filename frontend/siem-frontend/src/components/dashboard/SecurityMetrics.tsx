import React from 'react';
import { Card } from '../ui/card';

interface MetricCardProps {
  title: string;
  value: string | number;
  change?: string;
  trend?: 'up' | 'down' | 'neutral';
}

const MetricCard: React.FC<MetricCardProps> = ({ title, value, change, trend }) => (
  <div className="p-4 rounded-lg bg-white shadow-sm">
    <h3 className="text-sm font-medium text-gray-500">{title}</h3>
    <div className="mt-2 flex items-baseline">
      <p className="text-2xl font-semibold text-gray-900">{value}</p>
      {change && (
        <span className={`ml-2 text-sm ${
          trend === 'up' ? 'text-red-600' : 
          trend === 'down' ? 'text-green-600' : 
          'text-gray-500'
        }`}>
          {change}
        </span>
      )}
    </div>
  </div>
);

export const SecurityMetrics: React.FC = () => {
  // In a real application, these would come from your API
  const metrics = [
    {
      title: 'Total Alerts',
      value: '247',
      change: '+12.5%',
      trend: 'up' as const
    },
    {
      title: 'Critical Threats',
      value: '8',
      change: '-3%',
      trend: 'down' as const
    },
    {
      title: 'Average Response Time',
      value: '15m',
      change: '-2m',
      trend: 'down' as const
    },
    {
      title: 'Security Score',
      value: '85/100',
      change: '+5',
      trend: 'up' as const
    }
  ];

  return (
    <div>
      <h2 className="text-lg font-semibold mb-4">Security Metrics</h2>
      <div className="grid grid-cols-2 gap-4">
        {metrics.map((metric, index) => (
          <MetricCard
            key={index}
            title={metric.title}
            value={metric.value}
            change={metric.change}
            trend={metric.trend}
          />
        ))}
      </div>
    </div>
  );
};

export default SecurityMetrics;