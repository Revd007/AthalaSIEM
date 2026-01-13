import { api } from '@/lib/api';

export interface Incident {
  id: string;
  title: string;
  description: string;
  status: 'new' | 'investigating' | 'resolved' | 'dismissed';
  priority: 'low' | 'medium' | 'high' | 'critical';
  category: string;
  assignee?: string;
  reporter?: string;
  createdAt: string;
  updatedAt: string;
  timeline?: any[];
  affectedSystems?: string[];
  tags?: string[];
  metrics?: {
    mttd?: number;
    mtta?: number;
  };
}

export interface IncidentQueryParams {
  status?: string[];
  priority?: string[];
  category?: string[];
  assignee?: string[];
  limit?: number;
  offset?: number;
}

export const incidentService = {
  async getIncidents(params: IncidentQueryParams = {}): Promise<{ items: Incident[]; totalCount: number }> {
    const queryString = new URLSearchParams();
    
    if (params.status) params.status.forEach(s => queryString.append('status', s));
    if (params.priority) params.priority.forEach(p => queryString.append('priority', p));
    if (params.category) params.category.forEach(c => queryString.append('category', c));
    if (params.assignee) params.assignee.forEach(a => queryString.append('assignee', a));
    if (params.limit) queryString.append('limit', params.limit.toString());
    if (params.offset) queryString.append('offset', params.offset.toString());
    
    try {
      // For now, return empty array since incidents endpoint may not exist yet
      // This will be updated when backend incidents API is ready
      return { items: [], totalCount: 0 };
    } catch (error) {
      console.error('Error fetching incidents:', error);
      return { items: [], totalCount: 0 };
    }
  },

  async getIncidentById(id: string): Promise<Incident | null> {
    try {
      const { data } = await api.get<Incident>(`/api/incidents/${id}`);
      return data ?? null;
    } catch (error) {
      console.error('Error fetching incident:', error);
      return null;
    }
  },

  async createIncident(incident: Partial<Incident>): Promise<Incident | null> {
    try {
      const { data } = await api.post<Incident>('/api/incidents', incident);
      return data ?? null;
    } catch (error) {
      console.error('Error creating incident:', error);
      return null;
    }
  },

  async updateIncident(id: string, updates: Partial<Incident>): Promise<Incident | null> {
    try {
      const { data } = await api.put<Incident>(`/api/incidents/${id}`, updates);
      return data ?? null;
    } catch (error) {
      console.error('Error updating incident:', error);
      return null;
    }
  }
};
