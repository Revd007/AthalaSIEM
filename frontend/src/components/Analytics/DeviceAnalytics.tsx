import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { Monitor, Server, Cloud } from 'lucide-react';

const deviceData = [
  { name: 'Windows Servers', value: 45, type: 'windows' },
  { name: 'Linux Servers', value: 30, type: 'linux' },
  { name: 'Cloud Collectors', value: 25, type: 'cloud' },
];

const severityData = [
  { name: 'Critical', value: 15, color: '#ef4444' },
  { name: 'High', value: 25, color: '#f97316' },
  { name: 'Medium', value: 35, color: '#eab308' },
  { name: 'Low', value: 25, color: '#3b82f6' },
];

const COLORS = ['#3b82f6', '#10b981', '#8b5cf6'];

export function DeviceAnalytics() {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Device Analytics</h2>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Device Distribution</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={deviceData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {deviceData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Alert Severity by Device Type</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={severityData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {severityData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
        {deviceData.map((device, index) => (
          <div key={device.type} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
            <div className="flex items-center space-x-2">
              {device.type === 'windows' && <Monitor className="h-5 w-5 text-blue-500" />}
              {device.type === 'linux' && <Server className="h-5 w-5 text-green-500" />}
              {device.type === 'cloud' && <Cloud className="h-5 w-5 text-purple-500" />}
              <span className="font-medium text-gray-900 dark:text-white">{device.name}</span>
            </div>
            <div className="mt-2 text-sm text-gray-600 dark:text-gray-300">
              {device.value}% of total devices
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}