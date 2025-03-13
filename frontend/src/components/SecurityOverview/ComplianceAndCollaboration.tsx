import React from 'react';
import { IncidentWorkflow } from '../IncidentResponse/IncidentWorkflow';
import { RealTimeCollaboration } from '../Collaboration/RealTimeCollaboration';
import { LogStream } from '../Dashboard/LogStream';
import { ComplianceReport } from '../Compliance/ComplianceReport';

export function ComplianceAndCollaboration() {
  return (
    <>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <IncidentWorkflow />
        <RealTimeCollaboration />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <LogStream />
        <ComplianceReport />
      </div>
    </>
  );
}