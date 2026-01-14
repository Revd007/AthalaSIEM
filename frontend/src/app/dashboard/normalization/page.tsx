'use client'

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useNormalizationStatistics, useNormalizedLogs, type NormalizationFilters } from '@/hooks/useNormalization'
import { NormalizationStats } from '@/components/Normalization/NormalizationStats'
import { EventTypeChart } from '@/components/Normalization/EventTypeChart'
import { SeverityChart } from '@/components/Normalization/SeverityChart'
import { NormalizedLogsTable } from '@/components/Normalization/NormalizedLogsTable'
import { NormalizationFilters as FiltersComponent } from '@/components/Normalization/NormalizationFilters'
import { Database, BarChart3, FileText, Filter } from 'lucide-react'

export default function NormalizationPage() {
  const [filters, setFilters] = useState<NormalizationFilters>({
    page: 1,
    pageSize: 50,
  })
  const [dateRange, setDateRange] = useState<{ start?: string; end?: string }>({
    start: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
    end: new Date().toISOString(),
  })

  const { data: statistics, isLoading: statsLoading } = useNormalizationStatistics(
    dateRange.start,
    dateRange.end
  )
  const { data: logsData, isLoading: logsLoading } = useNormalizedLogs(filters)

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Log Normalization</h1>
          <p className="text-muted-foreground">
            View normalized logs with ECS (Elastic Common Schema) fields
          </p>
        </div>
      </div>

      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList>
          <TabsTrigger value="overview">
            <BarChart3 className="w-4 h-4 mr-2" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="logs">
            <FileText className="w-4 h-4 mr-2" />
            Normalized Logs
          </TabsTrigger>
          <TabsTrigger value="filters">
            <Filter className="w-4 h-4 mr-2" />
            Filters
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {/* Statistics Cards */}
          <NormalizationStats statistics={statistics} isLoading={statsLoading} />

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Event Type Distribution</CardTitle>
                <CardDescription>Distribution of normalized events by type</CardDescription>
              </CardHeader>
              <CardContent>
                <EventTypeChart data={statistics?.eventTypeDistribution || []} />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Severity Distribution</CardTitle>
                <CardDescription>Distribution of normalized events by severity</CardDescription>
              </CardHeader>
              <CardContent>
                <SeverityChart data={statistics?.severityDistribution || []} />
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="logs" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Normalized Logs</CardTitle>
              <CardDescription>
                View logs with ECS fields: timestamp, source_ip, event_type, severity
              </CardDescription>
            </CardHeader>
            <CardContent>
              <NormalizedLogsTable
                data={logsData}
                isLoading={logsLoading}
                onPageChange={(page) => setFilters((prev) => ({ ...prev, page }))}
              />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="filters" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Filter Normalized Logs</CardTitle>
              <CardDescription>
                Filter logs by ECS fields: event_type, source_ip, severity, date range
              </CardDescription>
            </CardHeader>
            <CardContent>
              <FiltersComponent
                filters={filters}
                dateRange={dateRange}
                onFiltersChange={setFilters}
                onDateRangeChange={setDateRange}
              />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
