import { axiosInstance } from '../lib/axios';

interface AlertFilters {
  severity?: string;
  status?: string;
  timeRange?: string;
}

export const alertService = {
  async getAlerts(filters: AlertFilters) {
    const response = await axiosInstance.get('/alerts', { params: filters });
    return response.data;
  },

  async updateAlertStatus(alertId: string, newStatus: string) {
    const response = await axiosInstance.patch(`/alerts/${alertId}/status`, {
      status: newStatus
    });
    return response.data;
  },

  async assignAlert(alertId: string, userId: string) {
    const response = await axiosInstance.patch(`/alerts/${alertId}/assign`, {
      user_id: userId
    });
    return response.data;
  },

  async getAlertDetails(alertId: string) {
    const response = await axiosInstance.get(`/alerts/${alertId}`);
    return response.data;
  }
};