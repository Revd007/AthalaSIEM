import React, { useEffect, useState } from 'react'
import { SecurityAlertService, SecurityAlert } from '../../services/security-alert-service'
import { DataTable } from '../ui/data-table'
import { Badge } from '../ui/badge'
import { Button } from '../ui/button'
import { SearchInput } from '../ui/search-input'
import { Select } from '../ui/select'
import { DatePicker } from '../ui/date-picker'
import { AlertCircle, Bell, Shield } from 'lucide-react'

const severityColors = {
  low: 'bg-blue-100 text-blue-800',
  medium: 'bg-yellow-100 text-yellow-800',
  high: 'bg-orange-100 text-orange-800',
  critical: 'bg-red-100 text-red-800',
}

export function SecurityAlerts() {
  const [alerts, setAlerts] = useState<SecurityAlert[]>([])
  const [loading, setLoading] = useState(true)
  const [filters, setFilters] = useState({
    severity: [],
    status: [],
    search: '',
    page: 1,
    limit: 10,
  })
  const [total, setTotal] = useState(0)

  useEffect(() => {
    fetchAlerts()
  }, [filters])

  const fetchAlerts = async () => {
    try {
      setLoading(true)
      const response = await SecurityAlertService.getAlerts(filters)
      setAlerts(response.data)
      setTotal(response.total)
    } catch (error) {
      console.error('Failed to fetch alerts:', error)
    } finally {
      setLoading(false)
    }
  }

  const columns = [
    {
      key: 'severity',
      title: 'Severity',
      render: (alert: SecurityAlert) => (
        <Badge variant="outline" className={severityColors[alert.severity]}>
          {alert.severity}
        </Badge>
      ),
    },
    {
      key: 'title',
      title: 'Title',
      render: (alert: SecurityAlert) => (
        <div className="flex items-center space-x-2">
          {alert.severity === 'critical' && (
            <AlertCircle className="h-4 w-4 text-red-500" />
          )}
          <span>{alert.title}</span>
        </div>
      ),
    },
    {
      key: 'status',
      title: 'Status',
      render: (alert: SecurityAlert) => (
        <Badge
          variant={
            alert.status === 'new'
              ? 'destructive'
              : alert.status === 'investigating'
              ? 'warning'
              : 'success'
          }
        >
          {alert.status}
        </Badge>
      ),
    },
    // ... more columns
  ]

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Security Alerts</h2>
        <Button>
          <Bell className="mr-2 h-4 w-4" />
          Configure Alerts
        </Button>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <SearchInput
          placeholder="Search alerts..."
          onSearch={(value) => setFilters({ ...filters, search: value })}
        />
        <Select
          placeholder="Filter by severity"
          options={[
            { label: 'Critical', value: 'critical' },
            { label: 'High', value: 'high' },
            { label: 'Medium', value: 'medium' },
            { label: 'Low', value: 'low' },
          ]}
          onChange={(value) => setFilters({ ...filters, severity: value })}
          isMulti
        />
        <Select
          placeholder="Filter by status"
          options={[
            { label: 'New', value: 'new' },
            { label: 'Investigating', value: 'investigating' },
            { label: 'Resolved', value: 'resolved' },
            { label: 'Closed', value: 'closed' },
          ]}
          onChange={(value) => setFilters({ ...filters, status: value })}
          isMulti
        />
      </div>

      <DataTable
        data={alerts}
        columns={columns}
        loading={loading}
        pagination={{
          currentPage: filters.page,
          pageSize: filters.limit,
          totalItems: total,
          onPageChange: (page) => setFilters({ ...filters, page }),
        }}
      />
    </div>
  )
}