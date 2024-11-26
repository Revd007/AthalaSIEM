import React from 'react';
import { Card } from '../../ui/card';
import { EventsChart } from '../events/events-chart';
import { RecentEvents } from '../events/recent-events';
import { EventsSummary } from '../events/events-summary';
import { useQuery } from '@tanstack/react-query';

interface EventsData {
  total: number;
  critical: number;
  warning: number;
  normal: number;
  chartData: {
    timestamp: string;
    count: number;
    type: string;
  }[];
  recentEvents: {
    id: string;
    timestamp: string;
    type: string;
    severity: 'critical' | 'warning' | 'normal';
    message: string;
    source: string;
    aiAnalysis: {
      description: string;
      recommendation?: string;
    };
  }[];
}

const fetchEventsData = async (): Promise<EventsData> => {
  const response = await fetch('/api/events/overview');
  if (!response.ok) {
    throw new Error('Failed to fetch events data');
  }
  return response.json();
};

export function EventsOverview() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['eventsOverview'],
    queryFn: fetchEventsData,
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error loading events data</div>;
  if (!data) return null;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <EventsSummary
          total={data.total}
          critical={data.critical}
          warning={data.warning}
          normal={data.normal}
        />
      </div>

      <Card className="p-6">
        <h3 className="text-lg font-medium mb-4">Events Traffic</h3>
        <EventsChart data={data.chartData} />
      </Card>

      <Card className="p-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-medium">Recent Events</h3>
          <button className="text-blue-600 hover:text-blue-800">
            View All
          </button>
        </div>
        <RecentEvents events={data.recentEvents} />
      </Card>
    </div>
  );
}