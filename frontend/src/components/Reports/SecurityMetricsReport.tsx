import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { FileText, TrendingUp, Shield } from 'lucide-react';

const securityMetrics = [
  { month: 'Jan', incidents: 45, resolved: 42, mttr: 2.5 },
  { month: 'Feb', incidents: 38, resolved: 35, mttr: 2.8 },
  { month: 'Mar', incidents: 52, resolved: 48, mttr: 2.2 },
  { month: 'Apr', incidents: 41, resolved: 39, mttr: 2.4 },
  { month: 'May', incidents: 35, resolved: 33, mttr: 2.1 },
  { month: 'Jun', incidents: 48, resolved: 45, mttr: 2.3 },
];

const kpis = [
  {
    title: 'Mean Time to Detect',
    value: '1.8 hours',
    change: '-12%',
    trend: 'positive',
  },
  {
    title: 'Mean Time to Respond',
    value: '2.4 hours',
    change: '-8%',
    trend: 'positive',
  },
  {
    title: 'Resolution Rate',
    value: '94.5%',
    change: '+2.5%',
    trend: 'positive',
  },
];

export function SecurityMetricsReport() {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <FileText className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Security Metrics Report</h2>
        </div>
        <button className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600">
          Export Report
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        {kpis.map((kpi, index) => (
          <div key={index} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
            <h3 className="text-sm text-gray-600 dark:text-gray-300">{kpi.title}</h3>
            <div className="mt-2 flex items-center justify-between">
              <span className="text-2xl font-semibold text-gray-900 dark:text-white">
                {kpi.value}
              </span>
              <div className={`flex items-center text-sm ${
                kpi.trend === 'positive' ? 'text-green-500' : 'text-red-500'
              }`}>
                <TrendingUp className="h-4 w-4 mr-1" />
                {kpi.change}
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={securityMetrics}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="incidents" name="Total Incidents" fill="#ef4444" />
            <Bar dataKey="resolved" name="Resolved" fill="#10b981" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}