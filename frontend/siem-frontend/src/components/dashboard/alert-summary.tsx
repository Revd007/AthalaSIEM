import React from 'react';

export const AlertSummary = () => {
  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Alert Summary</h2>
      <div className="space-y-2">
        <div className="flex justify-between">
          <span>Critical Alerts</span>
          <span className="font-bold text-red-600">5</span>
        </div>
        <div className="flex justify-between">
          <span>Warning Alerts</span>
          <span className="font-bold text-yellow-600">12</span>
        </div>
      </div>
    </div>
  );
};