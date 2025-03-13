import React from 'react';
import { Brain, AlertTriangle, Shield } from 'lucide-react';
import { useAIAnalysis } from '../../../hooks/useAIAnalysis';
import { SecurityThreat } from '../../../types/security';

export function AISecurityAnalyzer() {
  const { threats, isAnalyzing, analyze } = useAIAnalysis();

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Brain className="h-6 w-6 text-purple-500" />
          <h2 className="text-xl font-semibold">AI Security Analysis</h2>
        </div>
        <button 
          onClick={() => analyze()}
          className="px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600"
          disabled={isAnalyzing}
        >
          {isAnalyzing ? 'Analyzing...' : 'Run Analysis'}
        </button>
      </div>

      <div className="space-y-4">
        {threats.map((threat) => (
          <ThreatCard key={threat.id} threat={threat} />
        ))}
      </div>
    </div>
  );
}

function ThreatCard({ threat }: { threat: SecurityThreat }) {
  return (
    <div className="border dark:border-gray-700 rounded-lg p-4">
      <div className="flex items-start space-x-3">
        {threat.severity === 'critical' ? (
          <AlertTriangle className="h-5 w-5 text-red-500" />
        ) : (
          <Shield className="h-5 w-5 text-yellow-500" />
        )}
        <div>
          <h3 className="font-medium">{threat.title}</h3>
          <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">
            {threat.description}
          </p>
          <div className="mt-2 flex items-center space-x-2">
            <span className="text-sm text-gray-500">
              Confidence: {threat.confidence}%
            </span>
            <span className="text-sm text-gray-500">
              Impact: {threat.impact}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}