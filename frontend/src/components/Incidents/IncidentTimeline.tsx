'use client'

import { useState } from 'react'
import { Clock, Plus, AlertCircle, Search, Shield, Activity } from 'lucide-react'
import { Button } from '@/components/ui/button'
import type { TimelineEvent } from '@/types/incident'

interface IncidentTimelineProps {
  timeline: TimelineEvent[]
  onAddEvent: (event: TimelineEvent) => Promise<void>
}

type EventType = 'detection' | 'investigation' | 'response' | 'resolution';

interface NewEvent {
  action: string;
  type: EventType;
}

export function IncidentTimeline({ timeline, onAddEvent }: IncidentTimelineProps) {
  const [isAddingEvent, setIsAddingEvent] = useState(false)
  const [newEvent, setNewEvent] = useState<NewEvent>({
    action: '',
    type: 'investigation'
  })

  const eventTypeIcons = {
    detection: AlertCircle,
    investigation: Search,
    response: Shield,
    resolution: Activity
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      await onAddEvent({
        id: crypto.randomUUID(),
        time: new Date().toISOString(),
        user: 'Current User', // Replace with actual user
        ...newEvent
      })
      setIsAddingEvent(false)
      setNewEvent({ action: '', type: 'investigation' })
    } catch (error) {
      console.error('Failed to add event:', error)
    }
  }

  const handleTypeChange = (value: string) => {
    setNewEvent({ ...newEvent, type: value as EventType })
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-medium">Timeline</h3>
        <Button 
          size="sm" 
          variant="outline"
          onClick={() => setIsAddingEvent(true)}
        >
          <Plus className="h-4 w-4 mr-2" />
          Add Event
        </Button>
      </div>

      {isAddingEvent && (
        <form onSubmit={handleSubmit} className="space-y-4 p-4 border rounded-lg">
          <div>
            <label className="block text-sm font-medium mb-1">
              Event Type
            </label>
            <select
              value={newEvent.type}
              onChange={(e) => handleTypeChange(e.target.value)}
              className="w-full rounded-md border p-2"
            >
              <option value="detection">Detection</option>
              <option value="investigation">Investigation</option>
              <option value="response">Response</option>
              <option value="resolution">Resolution</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">
              Action Description
            </label>
            <textarea
              value={newEvent.action}
              onChange={(e) => setNewEvent({ ...newEvent, action: e.target.value })}
              className="w-full rounded-md border p-2"
              rows={3}
              required
            />
          </div>
          <div className="flex justify-end space-x-2">
            <Button 
              type="button" 
              variant="outline"
              onClick={() => setIsAddingEvent(false)}
            >
              Cancel
            </Button>
            <Button type="submit">
              Add Event
            </Button>
          </div>
        </form>
      )}

      <div className="space-y-4">
        {timeline.map((event) => {
          const Icon = eventTypeIcons[event.type]
          return (
            <div key={event.id} className="flex items-start space-x-3">
              <div className="flex-shrink-0">
                <Icon className="h-5 w-5 text-gray-400" />
              </div>
              <div className="flex-1 space-y-1">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium">{event.action}</p>
                  <div className="flex items-center text-sm text-gray-500">
                    <Clock className="h-4 w-4 mr-1" />
                    {new Date(event.time).toLocaleString()}
                  </div>
                </div>
                <p className="text-sm text-gray-500">
                  by {event.user}
                </p>
                {event.details && (
                  <p className="text-sm text-gray-600 mt-1">
                    {event.details}
                  </p>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
} 