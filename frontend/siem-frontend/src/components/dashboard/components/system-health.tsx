import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle } from '../../ui/card'
import { SystemStatus } from '../../../types/dashboard'

interface SystemHealthProps {
  healthData: SystemStatus;
}

export function SystemHealth({ healthData }: SystemHealthProps) {
  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">System Health</h3>
      <div className="grid gap-4">
        <div className={`p-4 rounded-lg ${getStatusColor(healthData.status)}`}>
          <h4 className="font-medium">{healthData.name}</h4>
          <p className="text-sm">Uptime: {healthData.uptime}</p>
        </div>
        {healthData.components.map((component) => (
          <div
            key={component.name}
            className={`p-4 rounded-lg ${getStatusColor(component.status)}`}
          >
            <h4 className="font-medium">{component.name}</h4>
            <p className="text-sm">Uptime: {component.uptime}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function getStatusColor(status: 'healthy' | 'warning' | 'critical'): string {
  switch (status) {
    case 'healthy':
      return 'bg-green-100 text-green-800';
    case 'warning':
      return 'bg-yellow-100 text-yellow-800';
    case 'critical':
      return 'bg-red-100 text-red-800';
  }
}