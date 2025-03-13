import { axiosInstance } from '../lib/axios';

interface SystemConfig {
  ai_settings: {
    model_type: string;
    threshold: number;
    enabled_features: string[];
  };
  collection_settings: {
    sources: string[];
    intervals: Record<string, number>;
  };
  retention_settings: {
    events_retention_days: number;
    alerts_retention_days: number;
  };
}

export const systemConfigService = {
  async getConfiguration(): Promise<SystemConfig> {
    const response = await axiosInstance.get('/system/config');
    return response.data;
  },

  async updateConfiguration(config: Partial<SystemConfig>): Promise<SystemConfig> {
    const response = await axiosInstance.put('/system/config', config);
    return response.data;
  },

  async verifyInstallation(): Promise<{
    status: 'success' | 'failed';
    checks: Record<string, boolean>;
    issues?: string[];
  }> {
    const response = await axiosInstance.get('/system/verify-installation');
    return response.data;
  }
};