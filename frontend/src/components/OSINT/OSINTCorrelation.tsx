'use client'

import { useMemo } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { Link2, AlertTriangle } from 'lucide-react'
import { Progress } from '@/components/ui/progress'
import { useQuery } from '@tanstack/react-query'
import { useAlerts } from '@/services/alert-service'
import { logService } from '@/services/log-service'
import { Skeleton } from '@/components/ui/skeleton'

interface CorrelationData {
  osintFinding: string
  prediction: string
  confidence: number
  impact: string
  recommendations: string[]
}

export function OSINTCorrelation() {
  // Fetch alerts for correlation analysis
  const { data: alertsData, isLoading: alertsLoading } = useAlerts({
    limit: 50,
    sortField: 'Timestamp',
    sortDirection: 'desc'
  });

  // Fetch logs for additional correlation
  const { data: logsData, isLoading: logsLoading } = useQuery({
    queryKey: ['osint-correlation-logs'],
    queryFn: async () => {
      const end = new Date();
      const start = new Date();
      start.setDate(start.getDate() - 7);
      return logService.getLogs({
        startDate: start.toISOString(),
        endDate: end.toISOString(),
        limit: 500
      });
    },
    refetchInterval: 60000,
  });

  // Generate correlations from alerts and logs
  const correlations: CorrelationData[] = useMemo(() => {
    if (!alertsData?.items && !logsData?.items) return [];

    const correlations: CorrelationData[] = [];
    const alerts = alertsData?.items || [];
    const logs = logsData?.items || [];

    // Look for credential-related alerts
    const credentialAlerts = alerts.filter(a => 
      a.message?.toLowerCase().includes('credential') ||
      a.message?.toLowerCase().includes('login') ||
      a.message?.toLowerCase().includes('authentication')
    );
    if (credentialAlerts.length >= 3) {
      correlations.push({
        osintFinding: 'Multiple Authentication Failures',
        prediction: 'Account Takeover Attempt',
        confidence: Math.min(95, 70 + credentialAlerts.length * 3),
        impact: 'High',
        recommendations: [
          'Force password reset for affected accounts',
          'Enable MFA immediately',
          'Monitor suspicious login patterns',
          'Review access logs'
        ]
      });
    }

    // Look for data exposure patterns
    const dataAlerts = alerts.filter(a => 
      a.message?.toLowerCase().includes('data') ||
      a.message?.toLowerCase().includes('leak') ||
      a.message?.toLowerCase().includes('exfil')
    );
    if (dataAlerts.length > 0) {
      correlations.push({
        osintFinding: 'Data Exposure Indicators',
        prediction: 'Potential Data Breach',
        confidence: Math.min(92, 65 + dataAlerts.length * 5),
        impact: 'Critical',
        recommendations: [
          'Identify and secure exposed data',
          'Review data access policies',
          'Update security controls',
          'Notify affected parties if needed'
        ]
      });
    }

    // Look for network anomalies
    const networkLogs = logs.filter(l => 
      l.message?.toLowerCase().includes('network') ||
      l.message?.toLowerCase().includes('connection') ||
      l.message?.toLowerCase().includes('firewall')
    );
    if (networkLogs.length > 50) {
      correlations.push({
        osintFinding: 'Unusual Network Activity',
        prediction: 'Network Intrusion',
        confidence: Math.min(88, 60 + Math.floor(networkLogs.length / 10)),
        impact: 'High',
        recommendations: [
          'Review firewall rules',
          'Check for unauthorized access',
          'Monitor network traffic patterns',
          'Update IDS/IPS signatures'
        ]
      });
    }

    // Look for malware indicators
    const malwareAlerts = alerts.filter(a => 
      a.message?.toLowerCase().includes('malware') ||
      a.message?.toLowerCase().includes('virus') ||
      a.message?.toLowerCase().includes('trojan')
    );
    if (malwareAlerts.length > 0) {
      correlations.push({
        osintFinding: 'Malware Indicators Detected',
        prediction: 'Active Malware Infection',
        confidence: 90,
        impact: 'Critical',
        recommendations: [
          'Isolate affected systems',
          'Run full system scans',
          'Update antivirus definitions',
          'Review endpoint protection'
        ]
      });
    }

    return correlations;
  }, [alertsData, logsData]);

  const isLoading = alertsLoading || logsLoading;

  if (isLoading) {
    return (
      <DashboardCard title="OSINT-Prediction Correlation" icon={Link2}>
        <div className="space-y-4">
          {[1, 2, 3].map(i => (
            <Skeleton key={i} className="h-32 w-full" />
          ))}
        </div>
      </DashboardCard>
    );
  }

  return (
    <DashboardCard title="OSINT-Prediction Correlation" icon={Link2}>
      <div className="space-y-6">
        {correlations.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            No significant correlations detected. System is operating normally.
          </div>
        ) : (
          correlations.map((correlation, index) => (
            <div key={index} className="p-4 border rounded-lg">
              <div className="flex items-start justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <AlertTriangle className="w-5 h-5 text-orange-500" />
                    <h3 className="font-medium">{correlation.prediction}</h3>
                  </div>
                  <p className="text-sm text-gray-500 mt-1">
                    Based on: {correlation.osintFinding}
                  </p>
                </div>
                <div className="text-right">
                  <span className="text-sm font-medium text-blue-500">
                    {correlation.confidence}% confidence
                  </span>
                  <p className="text-xs text-gray-500">
                    Impact: {correlation.impact}
                  </p>
                </div>
              </div>
              
              <div className="mt-4">
                <h4 className="text-sm font-medium mb-2">Recommendations</h4>
                <ul className="space-y-2">
                  {correlation.recommendations.map((rec, idx) => (
                    <li key={idx} className="text-sm text-gray-600 flex items-center gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                      {rec}
                    </li>
                  ))}
                </ul>
              </div>

              <div className="mt-4">
                <div className="flex justify-between text-sm mb-1">
                  <span>Correlation Strength</span>
                  <span>{correlation.confidence}%</span>
                </div>
                <Progress value={correlation.confidence} className="h-2" />
              </div>
            </div>
          ))
        )}
      </div>
    </DashboardCard>
  )
}
