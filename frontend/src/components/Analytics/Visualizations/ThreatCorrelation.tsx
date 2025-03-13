import React from 'react';
import { Network } from 'lucide-react';
import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const correlationData = [
  { time: '00:00', malware: 20, network: 15, auth: 10 },
  { time: '04:00', malware: 35, network: 25, auth: 30 },
  { time: '08:00', malware: 25, network: 20, auth: 15 },
  { time: '12:00', malware: 40, network: 30, auth: 25 },
  { time: '16:00', malware: 30, network: 35, auth: 20 },
  { time: '20:00', malware: 45, network: 40, auth: 35 },
];

export function ThreatCorrelation() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm"
    >
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Network className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Threat Correlation</h2>
        </div>
      </div>

      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={correlationData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'rgba(255, 255, 255, 0.9)',
                borderRadius: '8px',
                border: 'none',
                boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
              }}
            />
            <Line 
              type="monotone" 
              dataKey="malware" 
              stroke="#ef4444" 
              strokeWidth={2}
              dot={{ r: 4 }}
              activeDot={{ r: 6, strokeWidth: 2 }}
              name="Malware Events"
            />
            <Line 
              type="monotone" 
              dataKey="network" 
              stroke="#3b82f6" 
              strokeWidth={2}
              dot={{ r: 4 }}
              activeDot={{ r: 6, strokeWidth: 2 }}
              name="Network Events"
            />
            <Line 
              type="monotone" 
              dataKey="auth" 
              stroke="#10b981" 
              strokeWidth={2}
              dot={{ r: 4 }}
              activeDot={{ r: 6, strokeWidth: 2 }}
              name="Authentication Events"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}