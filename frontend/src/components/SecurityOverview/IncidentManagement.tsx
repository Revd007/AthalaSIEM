import React from 'react';
import { PredictiveAnalytics } from '../ThreatHunting/PredictiveAnalytics';
import { AutomatedPlaybooks } from '../IncidentResponse/AutomatedPlaybooks';

export function IncidentManagement() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
      <PredictiveAnalytics />
      <AutomatedPlaybooks />
    </div>
  );
}