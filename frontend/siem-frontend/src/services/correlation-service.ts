import { axiosInstance } from '../lib/axios';

interface CorrelationRule {
  id: string;
  name: string;
  conditions: any[];
  actions: any[];
  enabled: boolean;
}

export const correlationService = {
  async getRules(): Promise<CorrelationRule[]> {
    const response = await axiosInstance.get('/correlation/rules');
    return response.data;
  },

  async createRule(rule: Partial<CorrelationRule>): Promise<CorrelationRule> {
    const response = await axiosInstance.post('/correlation/rules', rule);
    return response.data;
  },

  async getCorrelatedEvents(alertId: string): Promise<any[]> {
    const response = await axiosInstance.get(`/correlation/events/${alertId}`);
    return response.data;
  }
};