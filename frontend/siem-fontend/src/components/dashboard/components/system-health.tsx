import React from 'react';

export const SystemHealth = () => {
  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">System Health</h2>
      <div className="space-y-4">
        <div className="flex justify-between">
          <span>CPU Usage</span>
          <span className="font-bold">45%</span>
        </div>
        <div className="flex justify-between">
          <span>Memory Usage</span>
          <span className="font-bold">60%</span>
        </div>
      </div>
    </div>
  );
};