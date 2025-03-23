'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { IncidentDashboard } from '@/components/Incidents/IncidentDashboard'
import { IncidentMetrics } from '@/components/Incidents/IncidentMetrics'
import { IncidentFilters } from '@/components/Incidents/IncidentFilters'
import { CreateIncidentModal } from '@/components/Incidents/CreateIncidentModal'
import { Button } from '@/components/ui/button'
import { Plus } from 'lucide-react'
import type { Filters } from '@/components/Incidents/IncidentFilters'

export default function IncidentsPage() {
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false)
  const [filters, setFilters] = useState<Filters>({
    status: [],
    priority: [],
    category: [],
    assignee: []
  })
  const router = useRouter()
  useEffect(() => {
    // Check if user is logged in
    const token = localStorage.getItem('token')
    if (!token) {
      router.push('/login')
    }
  }, [router])

  return (
    <div className="p-8 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Incident Management
        </h1>
        <Button onClick={() => setIsCreateModalOpen(true)}>
          <Plus className="h-4 w-4 mr-2" />
          Create Incident
        </Button>
      </div>

      <IncidentMetrics />
      
      <div className="flex justify-between items-center">
        <IncidentFilters filters={filters} onFiltersChange={setFilters} />
      </div>

      <IncidentDashboard filters={filters} />

      <CreateIncidentModal 
        open={isCreateModalOpen}
        onClose={() => setIsCreateModalOpen(false)}
      />
    </div>
  )
} 