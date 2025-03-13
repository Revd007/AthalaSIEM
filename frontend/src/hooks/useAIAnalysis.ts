import { useState, useCallback } from 'react';
import { SecurityThreat } from '../types/security';
import { analyzeSecurityData } from '../utils/aiAnalysis';

export function useAIAnalysis() {
  const [threats, setThreats] = useState<SecurityThreat[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const analyze = useCallback(async () => {
    setIsAnalyzing(true);
    try {
      const results = await analyzeSecurityData();
      setThreats(results);
    } catch (error) {
      console.error('AI analysis failed:', error);
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  return { threats, isAnalyzing, analyze };
}