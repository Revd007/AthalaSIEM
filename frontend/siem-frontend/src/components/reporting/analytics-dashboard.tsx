import React, { useState, useEffect } from 'react';
import { analyticsService } from '../../services/analytics-service';
import { ChartCard } from '../ui/chart-card';
import { MetricsGrid } from '../ui/metrics-grid';
import { ReportGenerator } from './report-generator';
import { DateRangePicker } from '../ui/date-range-picker';

export function AnalyticsDashboard() {
  const [dateRange, setDateRange] = useState({ start: '', end: '' });
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadAnalytics();
  }, [dateRange]);

  const loadAnalytics = async () => {
    setLoading(true);
    try {
      const [
        threatMetrics,
        eventMetrics,
        systemMetrics,
        userMetrics
      ] = await Promise.all([
        analyticsService.getThreatMetrics(dateRange),
        analyticsService.getEventMetrics(dateRange),
        analyticsService.getSystemMetrics(dateRange),
        analyticsService.getUserMetrics(dateRange)
      ]);

      setMetrics({
        threats: threatMetrics,
        events: eventMetrics,
        system: systemMetrics,
        users: userMetrics
      });
    } catch (error) {
      console.error('Failed to load analytics:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Analytics Dashboard</h2>
        <div className="flex space-x-4">
          <DateRangePicker
            value={dateRange}
            onChange={setDateRange}
          />
          <ReportGenerator
            metrics={metrics}
            dateRange={dateRange}
          />
        </div>
      </div>

      {loading ? (
        <div>Loading analytics...</div>
      ) : (
        <>
          <MetricsGrid
            metrics={[
              {
                title: 'Total Alerts',
                value: metrics.threats.totalAlerts,
                change: metrics.threats.alertsChange,
                trend: 'up'
              },
              {
                title: 'Critical Events',
                value: metrics.events.criticalCount,
                change: metrics.events.criticalChange,
                trend: 'down'
              },
              // ... more metrics
            ]}
          />

          <div className="grid grid-cols-2 gap-4">
            <ChartCard
              title="Threat Distribution"
              chart={metrics.threats.distribution}
            />
            <ChartCard
              title="Event Timeline"
              chart={metrics.events.timeline}
            />
            <ChartCard
              title="System Performance"
              chart={metrics.system.performance}
            />
            <ChartCard
              title="User Activity"
              chart={metrics.users.activity}
            />
          </div>
        </>
      )}
    </div>
  );
}