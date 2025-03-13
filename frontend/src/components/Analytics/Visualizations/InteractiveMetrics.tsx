import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MetricsGauge } from './MetricsGauge';
import { useMetricsSocket } from '../hooks/useMetricsSocket';

export function InteractiveMetrics() {
  const [selectedDevice, setSelectedDevice] = useState<string | null>(null);
  const metrics = useMetricsSocket();

  const latestMetrics = metrics[metrics.length - 1];

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm"
    >
      <h2 className="text-xl font-semibold mb-6">Real-time Metrics</h2>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
        <MetricsGauge
          value={latestMetrics?.metrics.cpu ?? 0}
          label="CPU Usage"
          color="#ef4444"
        />
        <MetricsGauge
          value={latestMetrics?.metrics.memory ?? 0}
          label="Memory"
          color="#3b82f6"
        />
        <MetricsGauge
          value={latestMetrics?.metrics.disk ?? 0}
          label="Disk"
          color="#10b981"
        />
        <MetricsGauge
          value={latestMetrics?.metrics.network ?? 0}
          label="Network"
          color="#8b5cf6"
        />
      </div>

      <AnimatePresence>
        {metrics.map((update, index) => (
          <motion.div
            key={update.timestamp}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="mt-4 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
          >
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-500">
                {new Date(update.timestamp).toLocaleTimeString()}
              </span>
              <span className="text-sm font-medium">
                Device: {update.deviceId}
              </span>
            </div>
          </motion.div>
        ))}
      </AnimatePresence>
    </motion.div>
  );
}