import { axiosInstance } from '../lib/axios';

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

export const healthMonitoringService = {
  async getSystemHealth(): Promise<SystemHealth> {
    const response = await axiosInstance.get('/system/health');
    return response.data;
  },

  async getComponentMetrics(component: string, timeRange: string): Promise<any[]> {
    const response = await axiosInstance.get(`/system/metrics/${component}`, {
      params: { timeRange }
    });
    return response.data;
  }
};