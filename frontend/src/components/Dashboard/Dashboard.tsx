import React, { useState } from 'react';
import { MetricsGrid } from './MetricsGrid';
import { SecurityEvents } from './SecurityEvents';
import { NetworkTraffic } from './NetworkTraffic';
import { ThreatHunting } from './ThreatHunting';
import { AIThreatAnalyzer } from '../AI/SecurityAnalysis/AIThreatAnalyzer';
import { PredictiveAnalytics } from '../ThreatHunting/PredictiveAnalytics';
import { AutomatedPlaybooks } from '../IncidentResponse/AutomatedPlaybooks';
import { IncidentWorkflow } from '../IncidentResponse/IncidentWorkflow';
import { RealTimeCollaboration } from '../Collaboration/RealTimeCollaboration';
import { ComplianceReport } from '../Compliance/ComplianceReport';
import { RecentAlerts } from './RecentAlerts';
import { RealTimeDashboard } from './RealTimeDashboard';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export function Dashboard() {
  const [activeTab, setActiveTab] = useState('realtime')

  return (
    <div className="flex-1 bg-gray-50 dark:bg-gray-900 p-8">
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList>
          <TabsTrigger value="realtime">Real-Time</TabsTrigger>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="realtime" className="space-y-6">
          <RealTimeDashboard />
        </TabsContent>

        <TabsContent value="overview" className="space-y-8">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-8">Security Overview</h1>
          
          <div className="space-y-8">
            <MetricsGrid />
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <SecurityEvents />
              <RecentAlerts />
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <NetworkTraffic />
              <ThreatHunting />
            </div>
          </div>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-8">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-8">Advanced Analytics</h1>
          
          <div className="space-y-8">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <AIThreatAnalyzer />
              <PredictiveAnalytics />
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <AutomatedPlaybooks />
              <IncidentWorkflow />
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <RealTimeCollaboration />
              <ComplianceReport />
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}