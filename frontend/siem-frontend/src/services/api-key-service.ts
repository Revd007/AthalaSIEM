import { axiosInstance } from '../lib/axios';

interface APIKey {
  id: string;
  key: string;
  name: string;
  user_id: string;
  is_active: boolean;
  expires_at: string;
  created_at: string;
  last_used_at: string;
}

export const apiKeyService = {
  async getAPIKeys(): Promise<APIKey[]> {
    const response = await axiosInstance.get('/api-keys');
    return response.data;
  },

  async createAPIKey(name: string): Promise<APIKey> {
    const response = await axiosInstance.post('/api-keys', { name });
    return response.data;
  },

  async revokeAPIKey(id: string): Promise<void> {
    await axiosInstance.delete(`/api-keys/${id}`);
  }
};