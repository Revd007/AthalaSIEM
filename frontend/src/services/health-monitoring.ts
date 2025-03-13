import { useQuery } from '@tanstack/react-query';

interface SystemHealth {
  status: 'healthy' | 'degraded' | 'critical';
  components: {
    database: ComponentHealth;
    ai_engine: ComponentHealth;
    collectors: ComponentHealth;
    api: ComponentHealth;
  };
  metrics: {
    cpu_usage: number;
    memory_usage: number;
    disk_usage: number;
    event_processing_rate: number;
  };
}

interface ComponentHealth {
  status: 'up' | 'down';
  latency: number;
  last_check: string;
  details?: any;
}

export function useSystemHealth() {
  return useQuery<SystemHealth>({
    queryKey: ['system-health'],
    queryFn: async () => {
      const response = await fetch('/system/health');
      return response.json();
    },
    refetchInterval: 30000 // Refresh every 30 seconds
  });
}

export function useComponentMetrics(component: string, timeRange: string) {
  return useQuery({
    queryKey: ['component-metrics', component, timeRange],
    queryFn: async () => {
      const response = await fetch(`/system/metrics/${component}?timeRange=${timeRange}`);
      return response.json();
    },
    refetchInterval: 60000 // Refresh every minute
  });
}