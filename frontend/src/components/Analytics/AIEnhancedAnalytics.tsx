import React from 'react';
import { Brain, Shield, AlertTriangle, TrendingUp } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const anomalyData = [
  { timestamp: '00:00', baseline: 100, actual: 98, predicted: 102 },
  { timestamp: '04:00', baseline: 95, actual: 140, predicted: 97 },
  { timestamp: '08:00', baseline: 120, actual: 125, predicted: 118 },
  { timestamp: '12:00', baseline: 150, actual: 180, predicted: 148 },
  { timestamp: '16:00', baseline: 130, actual: 135, predicted: 132 },
  { timestamp: '20:00', baseline: 110, actual: 115, predicted: 108 },
];

const threatDistribution = [
  { name: 'Malware', value: 35, color: '#ef4444' },
  { name: 'Phishing', value: 25, color: '#f59e0b' },
  { name: 'DDoS', value: 20, color: '#3b82f6' },
  { name: 'Data Breach', value: 15, color: '#10b981' },
  { name: 'Other', value: 5, color: '#6b7280' },
];

export function AIEnhancedAnalytics() {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Brain className="h-6 w-6 text-purple-500" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">AI Security Analysis</h2>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Anomaly Detection</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={anomalyData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis />
                <Tooltip />
                <Area type="monotone" dataKey="baseline" stackId="1" stroke="#9ca3af" fill="#d1d5db" name="Baseline" />
                <Area type="monotone" dataKey="actual" stackId="2" stroke="#3b82f6" fill="#93c5fd" name="Actual" />
                <Area type="monotone" dataKey="predicted" stackId="3" stroke="#10b981" fill="#6ee7b7" name="Predicted" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Threat Distribution</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={threatDistribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {threatDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="mt-8">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">AI Insights</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-yellow-50 dark:bg-yellow-900/30 p-4 rounded-lg">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="h-5 w-5 text-yellow-500" />
              <h4 className="font-medium text-yellow-800 dark:text-yellow-200">Anomaly Detected</h4>
            </div>
            <p className="mt-2 text-sm text-yellow-700 dark:text-yellow-300">
              Unusual spike in failed login attempts detected at 04:00. 40% above baseline.
            </p>
          </div>
          <div className="bg-blue-50 dark:bg-blue-900/30 p-4 rounded-lg">
            <div className="flex items-center space-x-2">
              <TrendingUp className="h-5 w-5 text-blue-500" />
              <h4 className="font-medium text-blue-800 dark:text-blue-200">Trend Analysis</h4>
            </div>
            <p className="mt-2 text-sm text-blue-700 dark:text-blue-300">
              Predicted 25% increase in phishing attempts based on current patterns.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}