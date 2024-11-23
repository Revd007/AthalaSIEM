import React, { useEffect, useState } from 'react';
import * as eventService from '../../../services/event-service';
import { Event } from '../../../types/event';
import { DataTable } from '../../ui/data-table';
import { LogFilter } from '../../shared/filters/log-filter';

export const EventsOverview: React.FC = () => {
  const [events, setEvents] = useState<Event[]>([]);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState({
    severity: 'all',
    timeRange: '24h',
    source: 'all'
  });

  useEffect(() => {
    const fetchEvents = async () => {
      try {
        setLoading(true);
        const { events: newEvents } = await eventService.getEvents({
          limit: 100,
          severity: filters.severity !== 'all' ? filters.severity : undefined,
          source: filters.source !== 'all' ? filters.source : undefined
        });
        setEvents(newEvents);
      } catch (error) {
        console.error('Failed to fetch events:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchEvents();
  }, [filters]);

  return (
    <div className="space-y-4">
      <LogFilter filters={filters} setFilters={setFilters} />
      <DataTable
        data={events}
        columns={[
          { key: 'timestamp', title: 'Time' },
          { key: 'event_type', title: 'Type' },
          { key: 'source', title: 'Source' },
          { key: 'severity', title: 'Severity' },
          { key: 'message', title: 'Message' }
        ]}
        loading={loading}
      />
    </div>
  );
};