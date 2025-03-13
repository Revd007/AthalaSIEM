import React from 'react';
import { Brain, AlertTriangle, Activity } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const behaviorData = [
  { time: '00:00', normalScore: 95, userScore: 92 },
  { time: '04:00', normalScore: 95, userScore: 65 },
  { time: '08:00', normalScore: 90, userScore: 88 },
  { time: '12:00', normalScore: 92, userScore: 45 },
  { time: '16:00', normalScore: 94, userScore: 91 },
  { time: '20:00', normalScore: 93, userScore: 90 },
];

const anomalies = [
  {
    id: 1,
    user: 'john.doe',
    activity: 'Unusual file access pattern',
    severity: 'high',
    confidence: 89,
    details: 'Multiple sensitive file accesses outside normal working hours',
  },
  {
    id: 2,
    user: 'system.admin',
    activity: 'Privilege escalation attempt',
    severity: 'critical',
    confidence: 95,
    details: 'Unauthorized attempt to gain elevated permissions',
  },
];

export function AIBehavioralAnalysis() {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Brain className="h-6 w-6 text-purple-500" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">AI Behavioral Analysis</h2>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">User Behavior Score</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={behaviorData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="normalScore"
                  stroke="#10b981"
                  strokeWidth={2}
                  name="Normal Behavior"
                />
                <Line
                  type="monotone"
                  dataKey="userScore"
                  stroke="#ef4444"
                  strokeWidth={2}
                  name="Current Behavior"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Detected Anomalies</h3>
          <div className="space-y-4">
            {anomalies.map((anomaly) => (
              <div
                key={anomaly.id}
                className={`p-4 rounded-lg ${
                  anomaly.severity === 'critical' ? 'bg-red-50' : 'bg-yellow-50'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <AlertTriangle className={`h-5 w-5 ${
                      anomaly.severity === 'critical' ? 'text-red-500' : 'text-yellow-500'
                    }`} />
                    <span className="font-medium">{anomaly.user}</span>
                  </div>
                  <span className="text-sm">
                    {anomaly.confidence}% confidence
                  </span>
                </div>
                <p className="mt-2 text-sm font-medium">{anomaly.activity}</p>
                <p className="mt-1 text-sm text-gray-600">{anomaly.details}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}