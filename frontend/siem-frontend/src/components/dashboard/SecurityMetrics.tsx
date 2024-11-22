import React from 'react';
import { Card } from '../shared/Card';
import { SecurityScoreChart } from './charts/SecurityScoreChart';
import { useEvents } from '@/hooks/useEvents';
import { useAlerts } from '@/hooks/useAlerts';

export const SecurityMetrics: React.FC = () => {
  const { events, isLoading: eventsLoading } = useEvents();
  const { alerts, isLoading: alertsLoading } = useAlerts();

  if (eventsLoading || alertsLoading) {
    return <div>Loading metrics...</div>;
  }

  return (
    <div className="grid grid-cols-metrics gap-4">
      <Card>
        <h3 className="text-text-light dark:text-text-dark font-semibold mb-2">
          Total Alerts
        </h3>
        <div className="flex items-end space-x-2">
          <span className="text-2xl font-bold text-primary-600">
            156
          </span>
          <span className="text-severity-high text-sm">
            +12%
          </span>
        </div>
      </Card>
      
      <Card>
        <h3 className="text-text-light dark:text-text-dark font-semibold mb-2">
          Critical Alerts
        </h3>
        <div className="flex items-end space-x-2">
          <span className="text-2xl font-bold text-severity-critical">
            {alerts.filter(a => a.severity === 'critical').length}
          </span>
          <span className="text-severity-high text-sm">
            +5%
          </span>
        </div>
      </Card>

      <Card>
        <h3 className="text-text-light dark:text-text-dark font-semibold mb-2">
          Security Score
        </h3>
        <div className="flex items-end space-x-2">
          <span className="text-2xl font-bold text-primary-600">
            85
          </span>
          <span className="text-severity-low text-sm">
            +3%
          </span>
        </div>
      </Card>

      <Card>
        <h3 className="text-text-light dark:text-text-dark font-semibold mb-2">
          Active Events
        </h3>
        <div className="flex items-end space-x-2">
          <span className="text-2xl font-bold text-primary-600">
            {events.filter(e => e.status === 'active').length}
          </span>
          <span className="text-text-muted-light dark:text-text-muted-dark text-sm">
            Last 24h
          </span>
        </div>
      </Card>
    </div>
  );
};