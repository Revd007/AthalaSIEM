import { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '../../ui/dialog';
import { Badge } from '../../ui/badge';

interface Event {
  id: string;
  timestamp: string;
  type: string;
  severity: 'critical' | 'warning' | 'normal';
  message: string;
  source: string;
  aiAnalysis: {
    description: string;
    recommendation?: string;
  };
}

interface RecentEventsProps {
  events: Event[];
}

export function RecentEvents({ events }: RecentEventsProps) {
  const [selectedEvent, setSelectedEvent] = useState<Event | null>(null);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-100 text-red-800';
      case 'warning':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-green-100 text-green-800';
    }
  };

  return (
    <>
      <div className="divide-y">
        {events.map((event) => (
          <div
            key={event.id}
            className="py-4 cursor-pointer hover:bg-gray-50"
            onClick={() => setSelectedEvent(event)}
          >
            <div className="flex justify-between items-start">
              <div>
                <div className="flex items-center gap-2">
                  <Badge className={getSeverityColor(event.severity)}>
                    {event.severity}
                  </Badge>
                  <span className="text-sm text-gray-500">
                    {new Date(event.timestamp).toLocaleString()}
                  </span>
                </div>
                <p className="mt-1">{event.message}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      <Dialog
        open={!!selectedEvent}
        onOpenChange={() => setSelectedEvent(null)}
      >
        <DialogContent className="sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle>Event Details</DialogTitle>
          </DialogHeader>
          
          {selectedEvent && (
            <div className="space-y-4">
              <div>
                <h4 className="font-medium">Event Information</h4>
                <div className="mt-2 space-y-2">
                  <p><span className="font-medium">Type:</span> {selectedEvent.type}</p>
                  <p><span className="font-medium">Source:</span> {selectedEvent.source}</p>
                  <p><span className="font-medium">Timestamp:</span> {new Date(selectedEvent.timestamp).toLocaleString()}</p>
                  <p><span className="font-medium">Message:</span> {selectedEvent.message}</p>
                </div>
              </div>

              <div>
                <h4 className="font-medium">AI Analysis</h4>
                <div className="mt-2 space-y-2">
                  <p>{selectedEvent.aiAnalysis.description}</p>
                  {selectedEvent.aiAnalysis.recommendation && (
                    <div className="mt-4">
                      <h5 className="font-medium">Recommendation</h5>
                      <p className="text-green-700">{selectedEvent.aiAnalysis.recommendation}</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}