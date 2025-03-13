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
      const response = await fetch(`/api/incidents/${updatedIncident.id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updatedIncident)
      });
      
      if (!response.ok) {
        throw new Error('Failed to update incident');
      }
      
      const updated = await response.json();
      setSelectedIncident(updated);
    } catch (error) {
      console.error('Error updating incident:', error);
    }
  }

  const handleAddTimelineEvent = async (event: TimelineEvent) => {
    if (!selectedIncident) return;
    
    try {
      const response = await fetch(`/api/incidents/${selectedIncident.id}/timeline`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(event)
      });

      if (!response.ok) {
        throw new Error('Failed to add timeline event');
      }

      const updatedIncident = await response.json();
      setSelectedIncident(updatedIncident);
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