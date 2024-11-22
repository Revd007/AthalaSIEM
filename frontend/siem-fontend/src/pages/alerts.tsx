import React, { useEffect, useState } from 'react'
import { MainLayout } from '../components/layout/main-layout'
import { AlertTable } from '../components/shared/tables/alert-table'
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/card'
import { Input } from '../components/ui/input'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '../components/ui/select'
import api from '../lib/api'

export default function AlertsPage() {
  const [alerts, setAlerts] = useState([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState({
    severity: 'all',
    status: 'all',
    search: ''
  })

  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        const response = await api.get('/alerts', {
          params: {
            severity: filter.severity !== 'all' ? filter.severity : undefined,
            status: filter.status !== 'all' ? filter.status : undefined,
            search: filter.search || undefined
          }
        })
        setAlerts(response.data)
      } catch (error) {
        console.error('Failed to fetch alerts:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchAlerts()
  }, [filter])

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

        {loading ? (
          <div>Loading...</div>
        ) : (
          <AlertTable alerts={alerts} />
        )}
      </div>
    </MainLayout>
  )
}
