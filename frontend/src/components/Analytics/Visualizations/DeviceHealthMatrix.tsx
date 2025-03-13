import React from 'react';
import { Server, Activity, Memory, HardDrive } from 'lucide-react';
import { motion } from 'framer-motion';

interface DeviceHealth {
  id: string;
  name: string;
  type: 'windows' | 'linux' | 'cloud';
  metrics: {
    cpu: number;
    memory: number;
    disk: number;
    network: number;
  };
  status: 'healthy' | 'warning' | 'critical';
}

const devices: DeviceHealth[] = [
  {
    id: '1',
    name: 'WIN-DC01',
    type: 'windows',
    metrics: { cpu: 45, memory: 60, disk: 75, network: 30 },
    status: 'healthy',
  },
  {
    id: '2',
    name: 'LIN-WEB01',
    type: 'linux',
    metrics: { cpu: 85, memory: 70, disk: 65, network: 90 },
    status: 'warning',
  },
  {
    id: '3',
    name: 'AWS-PROD01',
    type: 'cloud',
    metrics: { cpu: 95, memory: 90, disk: 85, network: 95 },
    status: 'critical',
  },
];

const statusColors = {
  healthy: 'bg-green-100 text-green-800',
  warning: 'bg-yellow-100 text-yellow-800',
  critical: 'bg-red-100 text-red-800',
};

export function DeviceHealthMatrix() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="space-y-4"
    >
      {devices.map((device, index) => (
        <motion.div
          key={device.id}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.1 }}
          className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm"
        >
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-3">
              <Server className="h-6 w-6 text-blue-500" />
              <div>
                <h3 className="font-medium text-gray-900 dark:text-white">{device.name}</h3>
                <p className="text-sm text-gray-500">{device.type}</p>
              </div>
            </div>
            <span className={`px-3 py-1 rounded-full text-sm ${statusColors[device.status]}`}>
              {device.status}
            </span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard
              icon={Activity}
              label="CPU"
              value={device.metrics.cpu}
              color={getMetricColor(device.metrics.cpu)}
            />
            <MetricCard
              icon={Memory}
              label="Memory"
              value={device.metrics.memory}
              color={getMetricColor(device.metrics.memory)}
            />
            <MetricCard
              icon={HardDrive}
              label="Disk"
              value={device.metrics.disk}
              color={getMetricColor(device.metrics.disk)}
            />
            <MetricCard
              icon={Server}
              label="Network"
              value={device.metrics.network}
              color={getMetricColor(device.metrics.network)}
            />
          </div>
        </motion.div>
      ))}
    </motion.div>
  );
}

function getMetricColor(value: number): string {
  if (value >= 90) return 'text-red-500';
  if (value >= 75) return 'text-yellow-500';
  return 'text-green-500';
}

interface MetricCardProps {
  icon: React.ElementType;
  label: string;
  value: number;
  color: string;
}

function MetricCard({ icon: Icon, label, value, color }: MetricCardProps) {
  return (
    <div className="flex items-center space-x-2">
      <Icon className={`h-5 w-5 ${color}`} />
      <div>
        <div className="text-sm font-medium text-gray-600 dark:text-gray-300">{label}</div>
        <div className={`text-lg font-semibold ${color}`}>{value}%</div>
      </div>
    </div>
  );
}