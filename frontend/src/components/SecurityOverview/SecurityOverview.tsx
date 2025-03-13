import React from 'react';
import { ThreatAnalysis } from './ThreatAnalysis';
import { IncidentManagement } from './IncidentManagement';
import { ComplianceAndCollaboration } from './ComplianceAndCollaboration';

export function SecurityOverview() {
  return (
    <div className="mt-8 space-y-8">
      <ThreatAnalysis />
      <IncidentManagement />
      <ComplianceAndCollaboration />
    </div>
  );
}