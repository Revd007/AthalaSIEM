import React from 'react';
import { Brain, Zap, AlertTriangle } from 'lucide-react';

const aiInsights = [
  {
    id: 1,
    title: 'Anomaly Detection',
    description: 'Unusual pattern of failed login attempts detected from multiple IPs',
    confidence: 89,
    recommendation: 'Investigate potential brute force attack',
    severity: 'high',
  },
  {
    id: 2,
    title: 'Behavioral Analysis',
    description: 'User accessing sensitive files outside normal working hours',
    confidence: 75,
    recommendation: 'Review user activity and implement time-based access controls',
    severity: 'medium',
  },
  {
    id: 3,
    title: 'Threat Intelligence',
    description: 'Communication with known malicious IP addresses',
    confidence: 95,
    recommendation: 'Block identified IPs and scan affected systems',
    severity: 'critical',
  },
];

export function AIThreatAnalysis() {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Brain className="h-6 w-6 text-purple-500" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">AI Threat Analysis</h2>
        </div>
        <span className="text-sm text-gray-500">Updated 2 minutes ago</span>
      </div>

      <div className="space-y-4">
        {aiInsights.map((insight) => (
          <div key={insight.id} className="border dark:border-gray-700 rounded-lg p-4">
            <div className="flex items-start space-x-4">
              {insight.severity === 'critical' ? (
                <AlertTriangle className="h-5 w-5 text-red-500 mt-1" />
              ) : (
                <Zap className="h-5 w-5 text-yellow-500 mt-1" />
              )}
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <h3 className="font-medium text-gray-900 dark:text-white">{insight.title}</h3>
                  <span className="text-sm text-gray-500">
                    Confidence: {insight.confidence}%
                  </span>
                </div>
                <p className="mt-1 text-sm text-gray-600 dark:text-gray-300">{insight.description}</p>
                <div className="mt-2 flex items-center space-x-2">
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Recommendation:
                  </span>
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    {insight.recommendation}
                  </span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}