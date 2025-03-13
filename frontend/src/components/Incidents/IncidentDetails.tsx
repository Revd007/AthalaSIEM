'use client'

import { 
  AlertTriangle, 
  Clock, 
  User, 
  Tag, 
  Server,
  MessageSquare,
  Edit,
  Trash2
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import type { Incident } from '@/types/incident'

interface IncidentDetailsProps {
  incident: Incident;
  onUpdate: (incident: Incident) => Promise<void>;
}

export function IncidentDetails({ incident, onUpdate }: IncidentDetailsProps) {
  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-xl font-semibold">{incident.title}</h2>
          <p className="mt-1 text-gray-500">{incident.description}</p>
        </div>
        <div className="flex space-x-2">
          <Button variant="outline" size="sm">
            <Edit className="h-4 w-4 mr-2" />
            Edit
          </Button>
          <Button variant="outline" size="sm" className="text-red-600 hover:text-red-700">
            <Trash2 className="h-4 w-4 mr-2" />
            Delete
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-4">
          <div>
            <label className="text-sm font-medium text-gray-500">Status</label>
            <div className="mt-1 flex items-center space-x-2">
              <AlertTriangle className="h-5 w-5 text-yellow-500" />
              <span className="capitalize">{incident.status}</span>
            </div>
          </div>

          <div>
            <label className="text-sm font-medium text-gray-500">Assignee</label>
            <div className="mt-1 flex items-center space-x-2">
              <User className="h-5 w-5 text-gray-400" />
              <span>{incident.assignee}</span>
            </div>
          </div>

          <div>
            <label className="text-sm font-medium text-gray-500">Created</label>
            <div className="mt-1 flex items-center space-x-2">
              <Clock className="h-5 w-5 text-gray-400" />
              <span>{new Date(incident.createdAt).toLocaleString()}</span>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <div>
            <label className="text-sm font-medium text-gray-500">Category</label>
            <div className="mt-1 capitalize">{incident.category}</div>
          </div>

          <div>
            <label className="text-sm font-medium text-gray-500">Tags</label>
            <div className="mt-1 flex flex-wrap gap-2">
              {incident.tags.map((tag) => (
                <span 
                  key={tag}
                  className="px-2 py-1 bg-gray-100 text-gray-700 rounded-full text-sm"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>

          <div>
            <label className="text-sm font-medium text-gray-500">Affected Systems</label>
            <div className="mt-1 space-y-1">
              {incident.affectedSystems.map((system) => (
                <div key={system} className="flex items-center space-x-2">
                  <Server className="h-4 w-4 text-gray-400" />
                  <span className="text-sm">{system}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="pt-4 border-t">
        <Button className="w-full">
          <MessageSquare className="h-4 w-4 mr-2" />
          Add Comment
        </Button>
      </div>
    </div>
  )
} 