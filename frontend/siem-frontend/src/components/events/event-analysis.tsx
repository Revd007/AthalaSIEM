import React, { useState } from 'react';
import { eventService } from '../../services/event-service';
import { SearchBar } from '../ui/search-bar';
import { TimelineView } from './timeline-view';
import { EventDetails } from './event-details';
import { FilterPanel } from '../ui/filter-panel';

export function EventAnalysis() {
  const [searchQuery, setSearchQuery] = useState('');
  const [timeRange, setTimeRange] = useState('24h');
  const [filters, setFilters] = useState({
    eventType: [],
    source: [],
    severity: []
  });
  const [events, setEvents] = useState([]);
  const [selectedEvent, setSelectedEvent] = useState(null);

  const handleSearch = async () => {
    const results = await eventService.searchEvents({
      query: searchQuery,
      timeRange,
      ...filters
    });
    setEvents(results);
  };

  const handleEventSelect = async (eventId: string) => {
    const details = await eventService.getEventDetails(eventId);
    setSelectedEvent(details);
  };

  return (
    <div className="flex h-full">
      <div className="w-64 border-r p-4">
        <FilterPanel
          filters={filters}
          onChange={setFilters}
          onTimeRangeChange={setTimeRange}
        />
      </div>

      <div className="flex-1 p-4">
        <SearchBar
          value={searchQuery}
          onChange={setSearchQuery}
          onSearch={handleSearch}
          placeholder="Search events..."
        />

        <div className="mt-4 flex space-x-4">
          <div className="w-2/3">
            <TimelineView
              events={events}
              onEventClick={handleEventSelect}
            />
          </div>

          {selectedEvent && (
            <div className="w-1/3">
              <EventDetails event={selectedEvent} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}