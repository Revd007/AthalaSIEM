import React from "react"

interface ReportFilterProps {
    filters: {
      type: string
      status: string
      dateRange: string
    }
    setFilters: (filters: any) => void
  }
  
  export function ReportFilter({ filters, setFilters }: ReportFilterProps) {
    return (
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <select
          value={filters.type}
          onChange={(e) => setFilters({ ...filters, type: e.target.value })}
          className="block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
        >
          <option value="all">All Types</option>
          <option value="security">Security</option>
          <option value="compliance">Compliance</option>
          <option value="audit">Audit</option>
        </select>
  
        <select
          value={filters.status}
          onChange={(e) => setFilters({ ...filters, status: e.target.value })}
          className="block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
        >
          <option value="all">All Status</option>
          <option value="completed">Completed</option>
          <option value="processing">Processing</option>
          <option value="failed">Failed</option>
        </select>
  
        <select
          value={filters.dateRange}
          onChange={(e) => setFilters({ ...filters, dateRange: e.target.value })}
          className="block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
        >
          <option value="7d">Last 7 Days</option>
          <option value="30d">Last 30 Days</option>
          <option value="90d">Last 90 Days</option>
          <option value="custom">Custom Range</option>
        </select>
      </div>
    )
  }