import React from 'react';
import { Brain, Shield, AlertTriangle, TrendingUp } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const mockAIData = {
  anomalyScores: [
    { time: '00:00', score: 0.2 },
    { time: '04:00', score: 0.8 },
    { time: '08:00', score: 0.3 },
    { time: '12:00', score: 0.4 },
    { time: '16:00', score: 0.9 },
    { time: '20:00', score: 0.5 },
  ],
  insights: [
    {
      id: 1,
      type: 'anomaly',
      title: 'Unusual Authentication Pattern',
      description: 'Multiple failed login attempts detected from unusual locations',
      severity: 'high',
      confidence: 0.89,
    },
    {
      id: 2,
      type: 'prediction',
      title: 'DDoS Attack Prediction',
      description: 'High probability of DDoS attack in the next 24 hours',
      severity: 'critical',
      confidence: 0.92,
    },
  ],
};

export function AISecurityFeatures() {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Brain className="h-6 w-6 text-purple-500" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">AI Security Analysis</h2>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Anomaly Detection</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mockAIData.anomalyScores}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="score"
                  stroke="#8b5cf6"
                  strokeWidth={2}
                  dot={{ r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">AI Insights</h3>
          <div className="space-y-4">
            {mockAIData.insights.map(insight => (
              <div
                key={insight.id}
                className={`p-4 rounded-lg ${
                  insight.severity === 'critical' ? 'bg-red-50 dark:bg-red-900/30' : 'bg-yellow-50 dark:bg-yellow-900/30'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    {insight.type === 'anomaly' ? (
                      <AlertTriangle className={`h-5 w-5 ${
                        insight.severity === 'critical' ? 'text-red-500' : 'text-yellow-500'
                      }`} />
                    ) : (
                      <TrendingUp className="h-5 w-5 text-purple-500" />
                    )}
                    <h4 className="font-medium">{insight.title}</h4>
                  </div>
                  <span className="text-sm">
                    {(insight.confidence * 100).toFixed(0)}% confidence
                  </span>
                </div>
                <p className="mt-2 text-sm">{insight.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}