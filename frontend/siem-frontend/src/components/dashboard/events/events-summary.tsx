import React from 'react';

interface EventsSummaryProps {
  total: number;
  critical: number;
  warning: number;
  normal: number;
}

export function EventsSummary({ total, critical, warning, normal }: EventsSummaryProps) {
  return (
    <>
      <div className="p-4 bg-blue-100 rounded-lg">
        <h4>Total Events</h4>
        <p className="text-2xl font-bold">{total}</p>
      </div>
      <div className="p-4 bg-red-100 rounded-lg">
        <h4>Critical</h4>
        <p className="text-2xl font-bold">{critical}</p>
      </div>
      <div className="p-4 bg-yellow-100 rounded-lg">
        <h4>Warning</h4>
        <p className="text-2xl font-bold">{warning}</p>
      </div>
      <div className="p-4 bg-green-100 rounded-lg">
        <h4>Normal</h4>
        <p className="text-2xl font-bold">{normal}</p>
      </div>
    </>
  );
}