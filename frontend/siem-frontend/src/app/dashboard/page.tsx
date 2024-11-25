'use client'

import { Suspense, useState, useEffect } from 'react'
import { Card } from '../../components/ui/card'
import { Calendar } from '../../components/ui/calendar'
import { Button } from '../../components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../../components/ui/select'
import { Popover, PopoverContent, PopoverTrigger } from '../../components/ui/popover'
import { Input } from '../../components/ui/input'
import { LineChart, Line, BarChart, Bar, PieChart, Pie, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, CartesianGrid, Cell } from 'recharts'
import { Search, Calendar as CalendarIcon, Filter, Download, Settings, RefreshCcw } from 'lucide-react'
import { DashboardOverview } from '../../components/dashboard/overview'
import { LoadingSkeleton } from '../../components/ui/loading-skeleton'

export default function DashboardPage() {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) return <LoadingSkeleton />

  return (
    <div className="space-y-6">
      {/* Header with Actions */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-semibold">Dashboard</h1>
          <p className="text-sm text-gray-500">Monitor your security metrics and events</p>
        </div>
        
        <div className="flex items-center gap-4">
          {/* Search */}
          <div className="flex items-center gap-2 relative">
            <Search className="w-4 h-4 absolute left-3 text-gray-500" />
            <Input
              placeholder="Search events..."
              className="w-64 pl-9"
            />
          </div>

          {/* Date Filter */}
          <Popover>
            <PopoverTrigger asChild>
              <Button variant="outline" className="gap-2">
                <CalendarIcon className="w-4 h-4" />
                Filter Date
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-auto p-0">
              <Calendar />
            </PopoverContent>
          </Popover>

          {/* Actions */}
          <Button variant="outline" className="gap-2">
            <Filter className="w-4 h-4" />
            Filters
          </Button>
          <Button variant="outline" className="gap-2">
            <Download className="w-4 h-4" />
            Export
          </Button>
          <Button variant="outline" className="gap-2">
            <RefreshCcw className="w-4 h-4" />
            Refresh
          </Button>
          <Button variant="outline" className="gap-2">
            <Settings className="w-4 h-4" />
            Settings
          </Button>
        </div>
      </div>

      {/* Main Dashboard Content */}
      <Suspense fallback={<LoadingSkeleton />}>
        <DashboardOverview />
      </Suspense>
    </div>
  )
}