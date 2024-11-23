import { axiosInstance } from '../lib/axios';

interface AuditLog {
  id: string;
  user_id: string;
  action: string;
  resource_type: string;
  resource_id: string;
  details: any;
  ip_address: string;
  timestamp: string;
}

export const auditService = {
  async getAuditLogs(params: {
    startDate?: string;
    endDate?: string;
    userId?: string;
    action?: string;
    resourceType?: string;
  }): Promise<AuditLog[]> {
    const response = await axiosInstance.get('/audit-logs', { params });
    return response.data;
  },

  async createAuditEntry(data: Partial<AuditLog>): Promise<AuditLog> {
    const response = await axiosInstance.post('/audit-logs', data);
    return response.data;
  }
};