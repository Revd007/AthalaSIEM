import React, { useEffect, useState } from 'react';
import { aiService } from '../../services/ai-service';
import { Card } from '../ui/card';
import { Badge } from '../ui/badge';
import { LineChart } from '../ui/line-chart';

export function AIInsights() {
  const [insights, setInsights] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchInsights = async () => {
      try {
        const [threatAnalysis, anomalyDetection] = await Promise.all([
          aiService.getLatestThreats(),
          aiService.getAnomalyStats()
        ]);

        setInsights({
          threats: threatAnalysis,
          anomalies: anomalyDetection
        });
      } catch (error) {
        console.error('Failed to fetch AI insights:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchInsights();
    const interval = setInterval(fetchInsights, 300000); // 5 minutes
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return <div>Loading AI insights...</div>;
  }

  return (
    <div className="grid grid-cols-2 gap-4">
      <Card>
        <h3 className="text-lg font-medium mb-4">Threat Analysis</h3>
        {insights?.threats.map((threat: any) => (
          <div key={threat.id} className="mb-2 flex items-center justify-between">
            <span>{threat.description}</span>
            <Badge
              variant={threat.severity === 'high' ? 'destructive' : 'warning'}
            >
              {threat.confidence}%
            </Badge>
          </div>
        ))}
      </Card>

      <Card>
        <h3 className="text-lg font-medium mb-4">Anomaly Detection</h3>
        <LineChart
          data={insights?.anomalies.timeline || []}
          lines={[
            { key: 'score', name: 'Anomaly Score', color: '#ef4444' }
          ]}
        />
      </Card>
    </div>
  );
}