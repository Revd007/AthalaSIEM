import React from 'react';
import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useMetricsSocket } from '../hooks/useMetricsSocket';

export function MetricsTimeline() {
  const metrics = useMetricsSocket();

  const timelineData = metrics.map(update => ({
    time: new Date(update.timestamp).toLocaleTimeString(),
    ...update.metrics
  }));

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm"
    >
      <h2 className="text-xl font-semibold mb-6">Metrics Timeline</h2>
      
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={timelineData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis domain={[0, 100]} />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(255, 255, 255, 0.9)',
                borderRadius: '8px',
                boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
              }}
            />
            <Line
              type="monotone"
              dataKey="cpu"
              stroke="#ef4444"
              strokeWidth={2}
              dot={false}
              name="CPU"
            />
            <Line
              type="monotone"
              dataKey="memory"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              name="Memory"
            />
            <Line
              type="monotone"
              dataKey="disk"
              stroke="#10b981"
              strokeWidth={2}
              dot={false}
              name="Disk"
            />
            <Line
              type="monotone"
              dataKey="network"
              stroke="#8b5cf6"
              strokeWidth={2}
              dot={false}
              name="Network"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}