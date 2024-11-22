import React from 'react';
import { EventsOverview } from '../../dashboard/components/EventsOverview';
import { AlertsSummary } from '../../dashboard/components/AlertsSummary';
import { SystemHealth } from '../../dashboard/components/SystemHealth';
import { ThreatMap } from '../../dashboard/components/ThreatMap';

export const SecurityDashboard: React.FC = () => {
  return (
    <div className="grid grid-cols-2 gap-4">
      <EventsOverview />
      <AlertsSummary />
      <SystemHealth />
      <ThreatMap />
    </div>
  );
};