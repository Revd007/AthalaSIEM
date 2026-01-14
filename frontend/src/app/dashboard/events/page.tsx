'use client'

import { SecurityEventsOverview } from '@/components/SecurityEvents/SecurityEventsOverview'
import { EventsTimeline } from '@/components/SecurityEvents/EventsTimeline'
import { EventsDistribution } from '@/components/SecurityEvents/EventsDistribution'
import { EventsTable } from '@/components/SecurityEvents/EventsTable'
import { EventsFilters } from '@/components/SecurityEvents/EventsFilters'
import { EnhancedLogViewer } from '@/components/Logs/EnhancedLogViewer'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useState } from 'react'
import { FileText, Shield } from 'lucide-react'

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

      <Tabs defaultValue="overview" className="space-y-8">
        <TabsList>
          <TabsTrigger value="overview">
            <Shield className="w-4 h-4 mr-2" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="normalized">
            <FileText className="w-4 h-4 mr-2" />
            Normalized Logs
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-8">
          <SecurityEventsOverview timeRange={timeRange} filters={filters} />
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <EventsTimeline />
            <EventsDistribution />
          </div>

          <EventsTable filters={filters} />
        </TabsContent>

        <TabsContent value="normalized">
          <EnhancedLogViewer />
        </TabsContent>
      </Tabs>
    </div>
  )
} 