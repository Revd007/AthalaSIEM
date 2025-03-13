import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { format } from 'date-fns';

const data = Array.from({ length: 24 }, (_, i) => ({
  time: format(new Date(2024, 0, 1, i), 'HH:mm'),
  inbound: Math.floor(Math.random() * 1000),
  outbound: Math.floor(Math.random() * 800),
}));

export function NetworkTraffic() {
  return (
    <div className="bg-white rounded-lg p-6 shadow-sm">
      <h2 className="text-lg font-semibold mb-4">Network Traffic (24h)</h2>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            <Area
              type="monotone"
              dataKey="inbound"
              stackId="1"
              stroke="#3b82f6"
              fill="#93c5fd"
              name="Inbound"
            />
            <Area
              type="monotone"
              dataKey="outbound"
              stackId="1"
              stroke="#10b981"
              fill="#6ee7b7"
              name="Outbound"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}