import React from 'react';
import { ThreatHuntingDashboard } from '../ThreatHunting/ThreatHuntingDashboard';
import { AIThreatAnalysis } from '../ThreatHunting/AIThreatAnalysis';

export function ThreatAnalysis() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
      <ThreatHuntingDashboard />
      <AIThreatAnalysis />
    </div>
  );
}