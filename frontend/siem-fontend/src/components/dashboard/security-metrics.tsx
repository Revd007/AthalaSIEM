import React from 'react';

export const SecurityMetrics = () => {
  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Security Metrics</h2>
      <div className="space-y-4">
        <div className="flex justify-between">
          <span>Security Score</span>
          <span className="font-bold">85%</span>
        </div>
        <div className="flex justify-between">
          <span>Threats Blocked</span>
          <span className="font-bold">1,234</span>
        </div>
      </div>
    </div>
  );
};