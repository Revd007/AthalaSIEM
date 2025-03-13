import React, { useState } from 'react'
import { MainLayout } from '../components/layout/main-layout'
import { AlertTable } from '../components/shared/tables/alert-table'
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/card'
import { Input } from '../components/ui/input'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '../components/ui/select'
import { useAlerts } from '../services/alert-service'

export default function AlertsPage() {
  const [filter, setFilter] = useState({
    severity: 'all',
    status: 'all',
    search: ''
  })

  const { data: alerts, isLoading } = useAlerts({
    severity: filter.severity !== 'all' ? filter.severity : undefined,
    status: filter.status !== 'all' ? filter.status : undefined,
    search: filter.search || undefined
  })

  return (
    <MainLayout>
      <div className="space-y-4">
        <h1 className="text-2xl font-bold">Security Alerts</h1>
        
        <Card>
          <CardHeader>
            <CardTitle>Filters</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex gap-4">
              <Input
                placeholder="Search alerts..."
                value={filter.search}
                onChange={(e) => setFilter(prev => ({ ...prev, search: e.target.value }))}
                className="max-w-sm"
              />
              
              <Select
                value={filter.severity}
                onValueChange={(value) => setFilter(prev => ({ ...prev, severity: value }))}
              >
                <SelectTrigger className="w-[180px]">
                  <SelectValue placeholder="Severity" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Severities</SelectItem>
                  <SelectItem value="high">High</SelectItem>
                  <SelectItem value="medium">Medium</SelectItem>
                  <SelectItem value="low">Low</SelectItem>
                </SelectContent>
              </Select>

              <Select
                value={filter.status}
                onValueChange={(value) => setFilter(prev => ({ ...prev, status: value }))}
              >
                <SelectTrigger className="w-[180px]">
                  <SelectValue placeholder="Status" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Status</SelectItem>
                  <SelectItem value="active">Active</SelectItem>
                  <SelectItem value="resolved">Resolved</SelectItem>
                  <SelectItem value="investigating">Investigating</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>

        {isLoading ? (
          <div>Loading...</div>
        ) : (
          <AlertTable alerts={alerts} />
        )}
      </div>
    </MainLayout>
  )
}
