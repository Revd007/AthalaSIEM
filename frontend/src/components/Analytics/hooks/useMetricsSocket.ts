import { useState, useEffect } from 'react';

interface MetricsUpdate {
  deviceId: string;
  metrics: {
    cpu: number;
    memory: number;
    disk: number;
    network: number;
  };
  timestamp: string;
}

export function useMetricsSocket() {
  const [metrics, setMetrics] = useState<MetricsUpdate[]>([]);

  useEffect(() => {
    // Simulate real-time updates
    const interval = setInterval(() => {
      const update: MetricsUpdate = {
        deviceId: `device-${Math.floor(Math.random() * 3) + 1}`,
        metrics: {
          cpu: Math.floor(Math.random() * 100),
          memory: Math.floor(Math.random() * 100),
          disk: Math.floor(Math.random() * 100),
          network: Math.floor(Math.random() * 100),
        },
        timestamp: new Date().toISOString(),
      };
      setMetrics(prev => [...prev, update].slice(-10));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  return metrics;
}