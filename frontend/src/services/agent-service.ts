import { ApiResponse } from '@/types/api';
import { api, endpoints } from '../lib/api';
import type { NewAgentConfig } from '../types/agent';
import type { Agent, AgentStatus, AgentHealthReport, LogQueryParams, PaginatedResult, LogEntry } from '../types/agent';
import { authService } from './auth-service';

interface AgentResponse {
    id: string;
    name: string;
    status: string;
    installationCommand: string;
    message: string;
    apiKey?: string;
}

interface InstallerInfo {
    fileName: string;
    contentType: string;
    size: number;
    downloadUrl: string;
    serverUrl: string;
}

interface DownloadTokenResponse {
  token: string;
  expiresAt: string;
  downloadUrl: string;
}

interface SecureDownloadResponse {
  downloadUrl: string;
}

export const agentService = {
  async addAgent(agentConfig: NewAgentConfig): Promise<AgentResponse> {
    try {
      const response = await api.post<AgentResponse>('/api/agents/add-agent', agentConfig);
      if (!response.data) throw new Error('Failed to register agent');
      return response.data;
    } catch (error) {
      console.error('Agent registration error:', error);
      throw error;
    }
  },

  async getAgents(): Promise<Agent[]> {
    const response = await api.get<Agent[]>('/api/agents');
    return response.data ?? [];
  },

  async getAgentStatus(agentId: string): Promise<Agent> {
    const response = await api.get<Agent>(`/api/agents/${agentId}/status`);
    if (!response.data) throw new Error('Failed to get agent status');
    return response.data;
  },

  async configureAgent(agentId: string, config: Partial<Agent>): Promise<Agent> {
    const response = await api.put<Agent>(`/api/agents/${agentId}`, config);
    if (!response.data) throw new Error('Failed to configure agent');
    return response.data;
  },

  async getInstallerToken(type: string = 'windows'): Promise<DownloadTokenResponse> {
    try {
      const response = await api.post<DownloadTokenResponse>('/api/auth/installer-token', { type: type });
      
      if (!response.data) throw new Error('Failed to get installer token');
      return response.data;
    } catch (error) {
      console.error('Error getting installer token:', error);
      throw error;
    }
  },

  async getInstallerInfo(type: string): Promise<InstallerInfo> {
    try {
      // First get a download token
      const tokenResponse = await this.getInstallerToken(type);
      
      // Use the token to get installer info
      const response = await api.get<InstallerInfo>(`/api/agents/installer-info/${type}?token=${encodeURIComponent(tokenResponse.token)}`);
      
      if (!response.data) throw new Error('Failed to get installer info');
      return response.data;
    } catch (error) {
      console.error('Error getting installer info:', error);
      
      // Rethrow with more context if needed
      if (error instanceof Error) {
        if (error.message.includes('Failed to get installer token')) {
          throw new Error(`Failed to get installer info: ${error.message}`);
        }
      }
      
      throw error;
    }
  },

  async getSecureDownloadUrl(): Promise<string> {
    try {
      const response = await api.get<SecureDownloadResponse>('/api/auth/secure-download-url');
      
      if (!response.data || !response.data.downloadUrl) {
        throw new Error('Failed to get secure download URL');
      }
      
      console.log('Secure download URL obtained:', response.data.downloadUrl);
      return response.data.downloadUrl;
    } catch (error) {
      console.error('Error getting secure download URL:', error);
      throw error;
    }
  },

  async downloadAgentInstaller(type: string): Promise<void> {
    try {
      // Use the API client which already has the correct base URL
      const response = await api.get<SecureDownloadResponse>('/api/auth/secure-download-url');
      
      if (!response.data || !response.data.downloadUrl) {
        throw new Error('Failed to get secure download URL');
      }
      
      const downloadUrl = response.data.downloadUrl;
      console.log('Secure download URL obtained:', downloadUrl);
      
      // Make a direct fetch to the full URL
      const downloadResponse = await fetch(downloadUrl, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
          'Accept': 'application/octet-stream'
        },
        credentials: 'include'
      });

      if (!downloadResponse.ok) {
        throw new Error(`Download failed with status: ${downloadResponse.status}`);
      }
      
      // Get the blob from the response
      const blob = await downloadResponse.blob();
      
      // Create a download link
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      
      // Extract filename from Content-Disposition header
      const disposition = downloadResponse.headers.get('Content-Disposition');
      const filename = disposition?.match(/filename="?([^"]+)"?/)?.[1] || 'AthalaAgent-Setup.exe';
      
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      
      // Clean up
      setTimeout(() => {
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }, 100);

    } catch (error) {
      console.error('Download failed:', error);
      throw error;
    }
  },

  async getAgentHealth(agentId: string): Promise<AgentHealthReport> {
    const response = await api.get<AgentHealthReport>(`/api/agents/${agentId}/health`);
    return response.data;
  },

  async getAgentLogs(agentId: string, params: LogQueryParams): Promise<PaginatedResult<LogEntry>> {
    const queryString = new URLSearchParams();
    
    if (params.startDate) queryString.append('startDate', params.startDate.toISOString());
    if (params.endDate) queryString.append('endDate', params.endDate.toISOString());
    if (params.severity) queryString.append('severity', params.severity);
    if (params.source) queryString.append('source', params.source);
    if (params.page) queryString.append('page', params.page.toString());
    if (params.pageSize) queryString.append('pageSize', params.pageSize.toString());
    
    const response = await api.get<PaginatedResult<LogEntry>>(`/api/logs/agent/${agentId}?${queryString}`);
    return response.data;
  },

  async restartAgent(agentId: string): Promise<boolean> {
    try {
      await api.post(`/api/agents/${agentId}/restart`);
      return true;
    } catch (error) {
      console.error('Failed to restart agent:', error);
      return false;
    }
  },

  async deleteAgent(agentId: string): Promise<boolean> {
    try {
      await api.delete(`/api/agents/${agentId}`);
      return true;
    } catch (error) {
      console.error('Failed to delete agent:', error);
      throw error;
    }
  }
}; 