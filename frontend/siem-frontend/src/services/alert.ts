import { axiosInstance } from '../axios';
import { Alert, AlertResponse } from '../types/alert';

export const alertsApi = {
  getAlerts: async (): Promise<AlertResponse> => {
    const response = await axiosInstance.get('/alerts');
    return response.data;
  },

  acknowledgeAlert: async (alertId: string): Promise<Alert> => {
    const response = await axiosInstance.post(`/alerts/${alertId}/acknowledge`);
    return response.data;
  },

  updateAlertStatus: async (alertId: string, status: string): Promise<Alert> => {
    const response = await axiosInstance.put(`/alerts/${alertId}/status`, { status });
    return response.data;
  }
};