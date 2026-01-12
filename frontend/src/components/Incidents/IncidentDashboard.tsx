'use client'

import { useState } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { IncidentList } from './IncidentList'
import { IncidentTimeline } from './IncidentTimeline'
import { IncidentDetails } from './IncidentDetails'
import type { Incident, TimelineEvent } from '@/types/incident'

interface IncidentDashboardProps {
  filters: {
    status: string[];
    priority: string[];
    category: string[];
    assignee: string[];
  }
}

export function IncidentDashboard({ filters }: IncidentDashboardProps) {
  const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null)

  const handleUpdateIncident = async (updatedIncident: Incident) => {
    try {
      const { api } = await import('@/lib/api');
      const { data } = await api.put<Incident>(`/api/incidents/${updatedIncident.id}`, updatedIncident);
      setSelectedIncident(data);
    } catch (error) {
      console.error('Error updating incident:', error);
    }
  }

  const handleAddTimelineEvent = async (event: TimelineEvent) => {
    if (!selectedIncident) return;
    
    try {
      const { api } = await import('@/lib/api');
      const { data } = await api.post<Incident>(`/api/incidents/${selectedIncident.id}/timeline`, event);
      setSelectedIncident(data);
    } catch (error) {
      console.error('Error adding timeline event:', error);
    }
  }

  return (
    <div className="grid grid-cols-12 gap-6">
      <div className="col-span-12 lg:col-span-5">
        <DashboardCard>
          <IncidentList 
            filters={filters}
            onSelectIncident={setSelectedIncident}
            selectedIncidentId={selectedIncident?.id}
          />
        </DashboardCard>
      </div>

      <div className="col-span-12 lg:col-span-7">
        {selectedIncident ? (
          <div className="space-y-6">
            <DashboardCard>
              <IncidentDetails 
                incident={selectedIncident}
                onUpdate={handleUpdateIncident}
              />
            </DashboardCard>
            <DashboardCard>
              <IncidentTimeline 
                timeline={selectedIncident.timeline}
                onAddEvent={handleAddTimelineEvent}
              />
            </DashboardCard>
          </div>
        ) : (
          <div className="h-full flex items-center justify-center text-gray-500">
            Select an incident to view details
          </div>
        )}
      </div>
    </div>
  )
} 