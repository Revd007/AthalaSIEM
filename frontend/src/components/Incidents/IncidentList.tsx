'use client'

import { useState } from 'react'
import { AlertTriangle, Clock, User, Shield } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { incidentService, type Incident } from '@/services/incident-service'
import { Skeleton } from '@/components/ui/skeleton'

interface IncidentListProps {
  filters: {
    status: string[];
    priority: string[];
    category: string[];
    assignee: string[];
  };
  onSelectIncident: (incident: Incident) => void;
  selectedIncidentId?: string;
}

export function IncidentList({ 
  filters, 
  onSelectIncident, 
  selectedIncidentId 
}: IncidentListProps) {
  const { data, isLoading } = useQuery({
    queryKey: ['incidents', filters],
    queryFn: () => incidentService.getIncidents({
      status: filters.status,
      priority: filters.priority,
      category: filters.category,
      assignee: filters.assignee,
      limit: 100
    }),
    refetchInterval: 30000,
  });

  const incidents = data?.items ?? []

  const priorityColors = {
    critical: 'text-red-600 bg-red-50',
    high: 'text-orange-600 bg-orange-50',
    medium: 'text-yellow-600 bg-yellow-50',
    low: 'text-blue-600 bg-blue-50'
  }

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-medium">Active Incidents</h3>
        </div>
        <div className="space-y-2">
          {[1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-24 w-full" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-medium">Active Incidents</h3>
        <div className="text-sm text-gray-500">
          Showing {incidents.length} incidents
        </div>
      </div>

      {incidents.length === 0 ? (
        <div className="text-center text-gray-500 py-8">
          No incidents found
        </div>
      ) : (
        <div className="space-y-2">
          {incidents.map((incident) => (
          <div
            key={incident.id}
            className={`
              p-4 rounded-lg border cursor-pointer transition-colors
              ${selectedIncidentId === incident.id 
                ? 'border-blue-500 bg-blue-50' 
                : 'border-gray-200 hover:border-blue-300'
              }
            `}
            onClick={() => onSelectIncident(incident)}
          >
            <div className="flex items-start justify-between">
              <div>
                <div className="flex items-center space-x-2">
                  <AlertTriangle 
                    className={`h-5 w-5 ${
                      incident.priority === 'critical' ? 'text-red-500' : 'text-yellow-500'
                    }`} 
                  />
                  <h4 className="font-medium">{incident.title}</h4>
                </div>
                <div className="mt-1 text-sm text-gray-500">
                  {incident.description}
                </div>
              </div>
              <span className={`
                px-2 py-1 text-xs rounded-full
                ${priorityColors[incident.priority]}
              `}>
                {incident.priority}
              </span>
            </div>

            <div className="mt-4 flex items-center space-x-4 text-sm text-gray-500">
              <div className="flex items-center">
                <User className="h-4 w-4 mr-1" />
                {incident.assignee}
              </div>
              <div className="flex items-center">
                <Clock className="h-4 w-4 mr-1" />
                {new Date(incident.createdAt).toLocaleString()}
              </div>
              <div className="flex items-center">
                <Shield className="h-4 w-4 mr-1" />
                {incident.status}
              </div>
            </div>
          </div>
        ))}
        </div>
      )}
    </div>
  )
} 