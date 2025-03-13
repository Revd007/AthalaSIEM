'use client'

import { useState } from 'react'
import { AlertTriangle, Clock, User, Shield } from 'lucide-react'
import type { Incident } from '@/types/incident'

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
  // Mock data - replace with API call
  const incidents: Incident[] = [
    {
      id: '1',
      title: 'Ransomware Attack Attempt',
      description: 'Multiple ransomware indicators detected',
      status: 'investigating',
      priority: 'critical',
      category: 'security',
      assignee: 'Sarah Chen',
      reporter: 'System',
      createdAt: '2024-03-15T10:30:00Z',
      updatedAt: '2024-03-15T10:35:00Z',
      timeline: [],
      affectedSystems: ['WS-001', 'WS-002'],
      tags: ['ransomware', 'malware'],
      metrics: {
        mttd: 2.5,
        mtta: 5
      }
    }
    // Add more mock incidents
  ]

  const priorityColors = {
    critical: 'text-red-600 bg-red-50',
    high: 'text-orange-600 bg-orange-50',
    medium: 'text-yellow-600 bg-yellow-50',
    low: 'text-blue-600 bg-blue-50'
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-medium">Active Incidents</h3>
        <div className="text-sm text-gray-500">
          Showing {incidents.length} incidents
        </div>
      </div>

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
    </div>
  )
} 