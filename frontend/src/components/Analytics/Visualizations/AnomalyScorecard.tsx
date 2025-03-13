import React from 'react';
import { AlertTriangle, TrendingUp, TrendingDown } from 'lucide-react';
import { motion } from 'framer-motion';

interface AnomalyScore {
  category: string;
  score: number;
  trend: 'up' | 'down' | 'stable';
  change: number;
}

const anomalyScores: AnomalyScore[] = [
  { category: 'Network Traffic', score: 85, trend: 'up', change: 12 },
  { category: 'User Behavior', score: 92, trend: 'down', change: 5 },
  { category: 'System Access', score: 78, trend: 'up', change: 8 },
  { category: 'Data Transfer', score: 88, trend: 'stable', change: 0 },
];

export function AnomalyScorecard() {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4"
    >
      {anomalyScores.map((score, index) => (
        <motion.div
          key={score.category}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
          className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm"
        >
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-gray-600 dark:text-gray-300">
              {score.category}
            </h3>
            <AlertTriangle className={`h-5 w-5 ${
              score.score > 90 ? 'text-green-500' : 
              score.score > 80 ? 'text-yellow-500' : 'text-red-500'
            }`} />
          </div>
          <div className="mt-2 flex items-center justify-between">
            <div className="text-2xl font-semibold text-gray-900 dark:text-white">
              {score.score}
            </div>
            <div className={`flex items-center ${
              score.trend === 'up' ? 'text-green-500' :
              score.trend === 'down' ? 'text-red-500' : 'text-gray-500'
            }`}>
              {score.trend === 'up' && <TrendingUp className="h-4 w-4 mr-1" />}
              {score.trend === 'down' && <TrendingDown className="h-4 w-4 mr-1" />}
              <span className="text-sm">
                {score.change > 0 ? `+${score.change}%` : 
                 score.change < 0 ? `${score.change}%` : 'Stable'}
              </span>
            </div>
          </div>
        </motion.div>
      ))}
    </motion.div>
  );
}