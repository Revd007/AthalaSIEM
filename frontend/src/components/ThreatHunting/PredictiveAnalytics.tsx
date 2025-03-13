import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Brain, TrendingUp, AlertOctagon } from 'lucide-react';

const predictions = [
  { time: '12:00', actual: 45, predicted: 42 },
  { time: '13:00', actual: 52, predicted: 50 },
  { time: '14:00', actual: 48, predicted: 45 },
  { time: '15:00', actual: null, predicted: 58 },
  { time: '16:00', actual: null, predicted: 62 },
  { time: '17:00', actual: null, predicted: 55 },
];

const riskFactors = [
  {
    title: 'Increased Attack Surface',
    probability: 78,
    impact: 'High',
    description: 'Multiple new endpoints added without security baseline',
  },
  {
    title: 'Credential Compromise Risk',
    probability: 65,
    impact: 'Critical',
    description: 'Pattern suggests potential credential harvesting attempt',
  },
];

export function PredictiveAnalytics() {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Brain className="h-6 w-6 text-purple-500" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Predictive Analytics</h2>
        </div>
        <span className="text-sm text-gray-500">Next 3 hours forecast</span>
      </div>

      <div className="h-64 mb-6">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={predictions}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            <Line
              type="monotone"
              dataKey="actual"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={{ r: 4 }}
              name="Actual Events"
            />
            <Line
              type="monotone"
              dataKey="predicted"
              stroke="#ef4444"
              strokeDasharray="5 5"
              strokeWidth={2}
              dot={{ r: 4 }}
              name="Predicted Events"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="space-y-4">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white flex items-center">
          <TrendingUp className="h-5 w-5 mr-2 text-orange-500" />
          Emerging Risk Factors
        </h3>
        {riskFactors.map((risk, index) => (
          <div key={index} className="border dark:border-gray-700 rounded-lg p-4">
            <div className="flex items-start justify-between">
              <div>
                <div className="flex items-center space-x-2">
                  <AlertOctagon className="h-5 w-5 text-red-500" />
                  <h4 className="font-medium text-gray-900 dark:text-white">{risk.title}</h4>
                </div>
                <p className="mt-1 text-sm text-gray-600 dark:text-gray-300">{risk.description}</p>
              </div>
              <div className="text-right">
                <div className="text-sm font-medium text-gray-900 dark:text-white">
                  {risk.probability}% probability
                </div>
                <div className="text-sm text-red-500">Impact: {risk.impact}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}