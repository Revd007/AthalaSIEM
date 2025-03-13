'use client'

import { SecurityEventsOverview } from '@/components/SecurityEvents/SecurityEventsOverview'
import { EventsTimeline } from '@/components/SecurityEvents/EventsTimeline'
import { EventsDistribution } from '@/components/SecurityEvents/EventsDistribution'
import { EventsTable } from '@/components/SecurityEvents/EventsTable'
import { EventsFilters } from '@/components/SecurityEvents/EventsFilters'
import { useState } from 'react'

export default function SecurityEventsPage() {
  const [timeRange, setTimeRange] = useState('24h')
  const [filters, setFilters] = useState({
    severity: [],
    eventType: [],
    source: []
  })

  return (
    <div className="p-8 space-y-8">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Security Events</h1>
        <EventsFilters 
          filters={filters} 
          onFiltersChange={setFilters}
          timeRange={timeRange}
          onTimeRangeChange={setTimeRange}
        />
      </div>

      <SecurityEventsOverview timeRange={timeRange} filters={filters} />
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <EventsTimeline />
        <EventsDistribution />
      </div>

      <EventsTable filters={filters} />
    </div>
  )
} 