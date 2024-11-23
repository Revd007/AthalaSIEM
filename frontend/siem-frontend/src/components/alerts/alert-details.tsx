import React from 'react';
import { Button } from '../ui/button';
import { Card } from '../ui/card';
import { Badge } from '../ui/badge';

interface Alert {
  id: string;
  title: string;
  description: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  status: 'new' | 'investigating' | 'resolved';
  assigned_to?: string;
  created_at: string;
  updated_at: string;
}

interface AlertDetailsProps {
  alert: Alert;
  onClose: () => void;
  onStatusChange: (alertId: string, newStatus: string) => Promise<void>;
  onAssign: (alertId: string, userId: string) => Promise<void>;
}

export default function AlertDetails({ alert, onClose, onStatusChange, onAssign }: AlertDetailsProps) {
  // Function to map severity to badge variant
  const getSeverityVariant = (severity: string) => {
    const severityMap = {
      low: 'info',
      medium: 'warning',
      high: 'destructive',
      critical: 'destructive'
    } as const;
    return severityMap[severity as keyof typeof severityMap];
  };

  // Function to map status to badge variant
  const getStatusVariant = (status: string) => {
    const statusMap = {
      new: 'default',
      investigating: 'warning',
      resolved: 'success'
    } as const;
    return statusMap[status as keyof typeof statusMap];
  };

  return (
    <Card className="p-4">
      <div className="flex justify-between items-start mb-4">
        <h2 className="text-xl font-bold">{alert.title}</h2>
        <Button variant="ghost" onClick={onClose}>×</Button>
      </div>

      <div className="space-y-4">
        <div>
          <label className="font-medium">Severity:</label>
          <Badge variant={getSeverityVariant(alert.severity)}>
            {alert.severity}
          </Badge>
        </div>

        <div>
          <label className="font-medium">Status:</label>
          <Badge variant={getStatusVariant(alert.status)}>
            {alert.status}
          </Badge>
        </div>

        <div>
          <label className="font-medium">Description:</label>
          <p className="mt-1">{alert.description}</p>
        </div>

        <div>
          <label className="font-medium">Assigned To:</label>
          <p className="mt-1">{alert.assigned_to}</p>
        </div>

        <div>
          <label className="font-medium">Created At:</label>
          <p className="mt-1">{alert.created_at}</p>
        </div>

        <div>
          <label className="font-medium">Updated At:</label>
          <p className="mt-1">{alert.updated_at}</p>
        </div>

        <div className="mt-4">
          <Button variant="outline" onClick={() => onStatusChange(alert.id, 'investigating')}>Investigate</Button>
          <Button variant="outline" onClick={() => onStatusChange(alert.id, 'resolved')}>Resolve</Button>
          <Button variant="outline" onClick={() => onAssign(alert.id, 'user_id')}>Assign</Button>
        </div>
      </div>
    </Card>
  );
}

export type { Alert, AlertDetailsProps };