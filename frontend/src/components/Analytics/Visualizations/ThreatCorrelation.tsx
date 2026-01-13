'use client'

import React, { useMemo } from 'react';
import { Network } from 'lucide-react';
import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useQuery } from '@tanstack/react-query';
import { logService } from '@/services/log-service';

export function ThreatCorrelation() {
  // Fetch logs for correlation analysis
  const { data: logsData, isLoading } = useQuery({
    queryKey: ['threat-correlation-logs'],
    queryFn: async () => {
      const end = new Date();
      const start = new Date();
      start.setDate(start.getDate() - 1);
      return await logService.getLogs({
        startDate: start.toISOString(),
        endDate: end.toISOString(),
        limit: 5000
      });
    },
    enabled: typeof window !== 'undefined' && !!localStorage.getItem('token'),
    refetchInterval: 60000,
  });

  // Generate correlation data from logs
  const correlationData = useMemo(() => {
    if (!logsData?.items) {
      return Array.from({ length: 6 }, (_, i) => ({
        time: `${String(i * 4).padStart(2, '0')}:00`,
        malware: 0,
        network: 0,
        auth: 0
      }));
    }

    const hourlyData: Record<string, { malware: number; network: number; auth: number }> = {};
    
    logsData.items.forEach(log => {
      const hour = new Date(log.timestamp).getHours();
      const timeSlot = `${String(Math.floor(hour / 4) * 4).padStart(2, '0')}:00`;
      
      if (!hourlyData[timeSlot]) {
        hourlyData[timeSlot] = { malware: 0, network: 0, auth: 0 };
      }

      const category = log.category?.toLowerCase() || '';
      const level = log.level?.toLowerCase() || '';
      
      if (category.includes('malware') || category.includes('threat')) {
        hourlyData[timeSlot].malware++;
      } else if (category.includes('network') || category.includes('firewall')) {
        hourlyData[timeSlot].network++;
      } else if (category.includes('auth') || category.includes('login') || level.includes('security')) {
        hourlyData[timeSlot].auth++;
      }
    });

    return ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'].map(time => ({
      time,
      malware: hourlyData[time]?.malware || 0,
      network: hourlyData[time]?.network || 0,
      auth: hourlyData[time]?.auth || 0
    }));
  }, [logsData]);

  if (isLoading) {
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
        <div className="h-80 flex items-center justify-center">
          <div className="animate-pulse text-gray-500">Loading correlation data...</div>
        </div>
      </motion.div>
    );
  }

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