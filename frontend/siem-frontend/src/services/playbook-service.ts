import { axiosInstance } from '../lib/axios';
import { PlaybookTemplate } from '../components/automation/playbook-editor';

interface PlaybookRun {
  id: string;
  alert_id: string;
  playbook_id: string;
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  start_time: string;
  end_time?: string;
  results: any;
}

export const playbookService = {
  async getTemplates(): Promise<PlaybookTemplate[]> {
    const response = await axiosInstance.get('/playbooks/templates');
    return response.data;
  },

  async getTemplate(id: string): Promise<PlaybookTemplate> {
    const response = await axiosInstance.get(`/playbooks/templates/${id}`);
    return response.data;
  },

  async createTemplate(template: Partial<PlaybookTemplate>): Promise<PlaybookTemplate> {
    const response = await axiosInstance.post('/playbooks/templates', template);
    return response.data;
  },

  async updateTemplate(id: string, template: Partial<PlaybookTemplate>): Promise<PlaybookTemplate> {
    const response = await axiosInstance.put(`/playbooks/templates/${id}`, template);
    return response.data;
  },

  async deleteTemplate(id: string): Promise<void> {
    await axiosInstance.delete(`/playbooks/templates/${id}`);
  },

  async executePlaybook(templateId: string, context: any): Promise<void> {
    await axiosInstance.post(`/playbooks/execute/${templateId}`, { context });
  }
};