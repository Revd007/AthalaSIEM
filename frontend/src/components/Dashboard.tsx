import React from 'react';
import { AlertCircle, ArrowUpRight, Shield, Activity } from 'lucide-react';

const stats = [
  { label: 'Total Events', value: '157,893', icon: Activity, change: '+12.5%' },
  { label: 'Critical Alerts', value: '23', icon: AlertCircle, change: '-5.2%' },
  { label: 'Threats Blocked', value: '1,284', icon: Shield, change: '+8.1%' },
];

export function Dashboard() {
  return (
    <div className="flex-1 p-8">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        {stats.map((stat) => (
          <div key={stat.label} className="bg-white rounded-lg p-6 shadow-sm">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">{stat.label}</p>
                <p className="text-2xl font-semibold mt-1">{stat.value}</p>
              </div>
              <div className="rounded-full p-3 bg-blue-50">
                <stat.icon className="w-6 h-6 text-blue-500" />
              </div>
            </div>
            <div className="flex items-center mt-4">
              <ArrowUpRight className="w-4 h-4 text-green-500" />
              <span className="text-sm text-green-500 ml-1">{stat.change}</span>
              <span className="text-sm text-gray-500 ml-2">vs last week</span>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg p-6 shadow-sm">
          <h2 className="text-lg font-semibold mb-4">Recent Events</h2>
          <div className="space-y-4">
            {/* Placeholder for events list */}
            <div className="border-l-4 border-yellow-500 pl-4">
              <p className="text-sm text-gray-600">Failed Login Attempt</p>
              <p className="text-xs text-gray-500">2 minutes ago</p>
            </div>
            <div className="border-l-4 border-red-500 pl-4">
              <p className="text-sm text-gray-600">Suspicious File Access</p>
              <p className="text-xs text-gray-500">15 minutes ago</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg p-6 shadow-sm">
          <h2 className="text-lg font-semibold mb-4">Top Security Alerts</h2>
          <div className="space-y-4">
            {/* Placeholder for alerts */}
            <div className="flex items-center justify-between p-3 bg-red-50 rounded-lg">
              <div>
                <p className="text-sm font-medium">Malware Detected</p>
                <p className="text-xs text-gray-500">3 instances</p>
              </div>
              <AlertCircle className="w-5 h-5 text-red-500" />
            </div>
            <div className="flex items-center justify-between p-3 bg-yellow-50 rounded-lg">
              <div>
                <p className="text-sm font-medium">Unauthorized Access</p>
                <p className="text-xs text-gray-500">5 attempts</p>
              </div>
              <Shield className="w-5 h-5 text-yellow-500" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}