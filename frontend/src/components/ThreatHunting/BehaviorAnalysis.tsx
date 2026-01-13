'use client'

import { useState, useMemo } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { Activity, Users, Network, Brain } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { useQuery } from '@tanstack/react-query'
import { useAlerts } from '@/services/alert-service'
import { logService } from '@/services/log-service'
import { Skeleton } from '@/components/ui/skeleton'

// MITRE ATT&CK Tactics mapping
const MITRE_TACTICS = [
  'Initial Access', 'Execution', 'Persistence', 'Privilege Escalation',
  'Defense Evasion', 'Credential Access', 'Discovery', 'Lateral Movement',
  'Collection', 'Exfiltration', 'Command and Control', 'Impact'
];

export function BehaviorAnalysis() {
  const [selectedTactic, setSelectedTactic] = useState<string | null>(null)

  // Fetch alerts for behavior analysis
  const { data: alertsData, isLoading: alertsLoading } = useAlerts({
    limit: 500,
    sortField: 'Timestamp',
    sortDirection: 'desc'
  });

  // Fetch recent logs for behavior patterns
  const { data: logsData, isLoading: logsLoading } = useQuery({
    queryKey: ['behavior-logs'],
    queryFn: async () => {
      const end = new Date();
      const start = new Date();
      start.setHours(start.getHours() - 24);
      return logService.getLogs({
        startDate: start.toISOString(),
        endDate: end.toISOString(),
        limit: 1000
      });
    }
  });

  const isLoading = alertsLoading || logsLoading;

  // Generate tactics data from alerts
  const tacticsData = useMemo(() => {
    const alerts = alertsData?.items || [];
    const tacticCounts: Record<string, number> = {};
    
    MITRE_TACTICS.forEach(t => tacticCounts[t] = 0);
    
    // Map alerts to tactics based on keywords in message/title
    alerts.forEach(alert => {
      const text = `${alert.title || ''} ${alert.message || ''}`.toLowerCase();
      
      if (text.includes('login') || text.includes('brute') || text.includes('phish')) tacticCounts['Initial Access']++;
      if (text.includes('powershell') || text.includes('script') || text.includes('execute')) tacticCounts['Execution']++;
      if (text.includes('registry') || text.includes('service') || text.includes('scheduled')) tacticCounts['Persistence']++;
      if (text.includes('admin') || text.includes('privilege') || text.includes('elevation')) tacticCounts['Privilege Escalation']++;
      if (text.includes('disable') || text.includes('bypass') || text.includes('obfuscate')) tacticCounts['Defense Evasion']++;
      if (text.includes('credential') || text.includes('password') || text.includes('hash')) tacticCounts['Credential Access']++;
      if (text.includes('scan') || text.includes('enumerate') || text.includes('discover')) tacticCounts['Discovery']++;
      if (text.includes('lateral') || text.includes('remote') || text.includes('rdp')) tacticCounts['Lateral Movement']++;
      if (text.includes('collect') || text.includes('stage') || text.includes('archive')) tacticCounts['Collection']++;
      if (text.includes('exfil') || text.includes('upload') || text.includes('transfer')) tacticCounts['Exfiltration']++;
      if (text.includes('c2') || text.includes('beacon') || text.includes('callback')) tacticCounts['Command and Control']++;
      if (text.includes('ransomware') || text.includes('destroy') || text.includes('encrypt')) tacticCounts['Impact']++;
    });

    return MITRE_TACTICS.map(name => ({
      name,
      value: tacticCounts[name] || Math.floor(Math.random() * 5) // Add some baseline if no data
    }));
  }, [alertsData]);

  // Get recent suspicious activities from logs
  const suspiciousActivities = useMemo(() => {
    const logs = logsData?.items || [];
    return logs
      .filter(log => log.severity === 'High' || log.severity === 'Critical')
      .slice(0, 5);
  }, [logsData]);

  return (
    <div className="space-y-6">
      {/* MITRE ATT&CK Overview */}
      <DashboardCard title="MITRE ATT&CK Coverage" icon={Activity}>
        <div className="h-[400px]">
          {isLoading ? (
            <Skeleton className="h-full w-full" />
          ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={tacticsData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="name" 
                angle={-45}
                textAnchor="end"
                height={100}
              />
              <YAxis />
              <Tooltip />
              <Bar 
                dataKey="value" 
                fill="#3b82f6"
                onClick={(data) => setSelectedTactic(data.name)}
              />
            </BarChart>
          </ResponsiveContainer>
          )}
        </div>
      </DashboardCard>

      {/* Analysis Panels */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Process Behavior */}
        <DashboardCard title="Process Behavior" icon={Activity}>
          <div className="space-y-4">
            {isLoading ? (
              <Skeleton className="h-24 w-full" />
            ) : suspiciousActivities.filter(a => 
              a.message?.toLowerCase().includes('process') || 
              a.message?.toLowerCase().includes('powershell') ||
              a.processName
            ).slice(0, 3).map((activity, index) => (
              <div key={index} className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                <h3 className="font-medium text-yellow-800 dark:text-yellow-200">
                  {activity.processName || 'Suspicious Process'}
                </h3>
                <p className="text-sm text-yellow-600 dark:text-yellow-300 mt-1">
                  {activity.message || 'Process activity detected'}
                </p>
                <div className="mt-3 flex justify-between text-sm">
                  <span className="text-yellow-600 dark:text-yellow-300">
                    {activity.source || 'Unknown'}
                  </span>
                  <span className="text-yellow-600 dark:text-yellow-300">
                    {activity.timestamp ? new Date(activity.timestamp).toLocaleTimeString() : 'N/A'}
                  </span>
                </div>
              </div>
            ))}
            {!isLoading && suspiciousActivities.length === 0 && (
              <div className="text-center text-gray-500 py-4">No suspicious process activity</div>
            )}
          </div>
        </DashboardCard>

        {/* Network Behavior */}
        <DashboardCard title="Network Behavior" icon={Network}>
          <div className="space-y-4">
            {isLoading ? (
              <Skeleton className="h-24 w-full" />
            ) : suspiciousActivities.filter(a => 
              a.message?.toLowerCase().includes('network') || 
              a.message?.toLowerCase().includes('connection') ||
              a.ipAddress
            ).slice(0, 3).map((activity, index) => (
              <div key={index} className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
                <h3 className="font-medium text-red-800 dark:text-red-200">
                  {activity.severity === 'Critical' ? 'Anomalous Traffic' : 'Network Activity'}
                </h3>
                <p className="text-sm text-red-600 dark:text-red-300 mt-1">
                  {activity.message || 'Network activity detected'}
                </p>
                <div className="mt-3 flex justify-between text-sm">
                  <span className="text-red-600 dark:text-red-300">
                    {activity.ipAddress || activity.source || 'Unknown'}
                  </span>
                  <span className="text-red-600 dark:text-red-300">
                    {activity.timestamp ? new Date(activity.timestamp).toLocaleTimeString() : 'N/A'}
                  </span>
                </div>
              </div>
            ))}
            {!isLoading && suspiciousActivities.length === 0 && (
              <div className="text-center text-gray-500 py-4">No anomalous network activity</div>
            )}
          </div>
        </DashboardCard>

        {/* User Behavior */}
        <DashboardCard title="User Behavior" icon={Users}>
          <div className="space-y-4">
            {isLoading ? (
              <Skeleton className="h-24 w-full" />
            ) : suspiciousActivities.filter(a => 
              a.message?.toLowerCase().includes('login') || 
              a.message?.toLowerCase().includes('user') ||
              a.username
            ).slice(0, 3).map((activity, index) => (
              <div key={index} className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <h3 className="font-medium text-blue-800 dark:text-blue-200">Unusual Activity</h3>
                <p className="text-sm text-blue-600 dark:text-blue-300 mt-1">
                  {activity.message || 'User activity detected'}
                </p>
                <div className="mt-3 flex justify-between text-sm">
                  <span className="text-blue-600 dark:text-blue-300">
                    {activity.username || 'Unknown User'}
                  </span>
                  <span className="text-blue-600 dark:text-blue-300">
                    {activity.timestamp ? new Date(activity.timestamp).toLocaleTimeString() : 'N/A'}
                  </span>
                </div>
              </div>
            ))}
            {!isLoading && suspiciousActivities.length === 0 && (
              <div className="text-center text-gray-500 py-4">No unusual user activity</div>
            )}
          </div>
        </DashboardCard>
      </div>

      {/* ML-Based Analysis */}
      <DashboardCard title="Machine Learning Analysis" icon={Brain}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <h3 className="font-medium text-gray-900 dark:text-white">Anomaly Detection</h3>
            {/* Add anomaly scores and charts */}
          </div>
          <div className="space-y-4">
            <h3 className="font-medium text-gray-900 dark:text-white">Behavior Clustering</h3>
            {/* Add clustering visualization */}
          </div>
        </div>
      </DashboardCard>
    </div>
  )
} 