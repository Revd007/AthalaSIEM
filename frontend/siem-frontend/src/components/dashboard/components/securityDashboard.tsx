import React, { useEffect, useState } from 'react';
import { EventsOverview } from './events-overview';
import { AlertSummary } from '../alert-summary';
import { SystemHealth } from './system-health';
import { ThreatMap } from './threat-map';
import { aiService } from '../../../services/ai-service';
import { SystemMonitor } from '../system-monitor';

export const SecurityDashboard: React.FC = () => {
  const [threatAnalysis, setThreatAnalysis] = useState<ThreatAnalysis | null>(null);
  const [anomalyData, setAnomalyData] = useState<AnomalyResult | null>(null);

  useEffect(() => {
    const fetchAIInsights = async () => {
      try {
        // Get latest event for threat analysis
        const latestEvent = await eventsApi.getLatestEvent();
        const threatResult = await aiService.analyzeThreat(latestEvent);
        setThreatAnalysis(threatResult);

        // Get system metrics for anomaly detection
        const metrics = await monitoringService.getSystemMetrics();
        const anomalyResult = await aiService.detectAnomalies(metrics);
        setAnomalyData(anomalyResult);
      } catch (error) {
        console.error('Error fetching AI insights:', error);
      }
    };

    fetchAIInsights();
    const interval = setInterval(fetchAIInsights, 300000); // Update every 5 minutes
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="grid grid-cols-2 gap-4">
      <EventsOverview />
      <AlertSummary threatData={threatAnalysis} />
      <SystemHealth anomalyData={anomalyData} />
      <ThreatMap />
      <SystemMonitor />
    </div>
  );
};