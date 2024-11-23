import React, { useState, useEffect } from 'react';
import { alertService } from '../.././services/alert-service';
import { DataTable } from '../ui/data-table';
import AlertDetails from '../alerts/alert-details';
import { AlertFilters } from '../alerts/alert-filters';
import { Button } from '../ui/button';
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

export function AlertManagement() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);
  const [filters, setFilters] = useState({
    severity: 'all',
    status: 'all',
    timeRange: '24h'
  });

  useEffect(() => {
    loadAlerts();
  }, [filters]);

  const loadAlerts = async () => {
    const data = await alertService.getAlerts(filters);
    setAlerts(data);
  };

  const handleStatusChange = async (alertId: string, newStatus: string) => {
    await alertService.updateAlertStatus(alertId, newStatus);
    loadAlerts();
  };

  const handleAssign = async (alertId: string, userId: string) => {
    await alertService.assignAlert(alertId, userId);
    loadAlerts();
  };

  return (
    <div className="space-y-4">
      <AlertFilters filters={filters} onChange={setFilters} />
      
      <div className="flex space-x-4">
        <div className="w-2/3">
          <DataTable
            data={alerts}
            columns={[
              {
                key: 'severity',
                title: 'Severity',
                render: (alert) => (
                  <Badge variant={alert.severity}>{alert.severity}</Badge>
                )
              },
              { key: 'title', title: 'Title' },
              { key: 'status', title: 'Status' },
              { key: 'created_at', title: 'Created' },
              {
                key: 'actions',
                title: 'Actions',
                render: (alert) => (
                  <div className="space-x-2">
                    <Button onClick={() => setSelectedAlert(alert)}>
                      View
                    </Button>
                    <Button onClick={() => handleStatusChange(alert.id, 'investigating')}>
                      Investigate
                    </Button>
                  </div>
                )
              }
            ]}
          />
        </div>

        {selectedAlert && (
          <div className="w-1/3">
            <AlertDetails
              alert={selectedAlert}
              onClose={() => setSelectedAlert(null)}
              onStatusChange={handleStatusChange}
              onAssign={handleAssign}
            />
          </div>
        )}
      </div>
    </div>
  );
}