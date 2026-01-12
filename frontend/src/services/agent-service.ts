import { ApiResponse } from '@/types/api';
import { api, endpoints, queryClient } from '../lib/api';
import type { 
  NewAgentConfig, 
  Agent, 
  AgentStatus, 
  AgentHealthReport, 
  LogQueryParams, 
  PaginatedResult, 
  LogEntry,
  AgentMetrics,
  AgentAlert,
  Severity
} from '../types/agent';
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

interface DownloadUrlResponse {
  downloadUrl: string;
}

interface DeploymentTokenResponse {
  token: string;
  expiresAt: string;
  downloadUrl: string;
}

interface DeploymentTokenConfig {
  name?: string;
  group?: string;
  serverAddress?: string;
}

export const agentService = {
  async addAgent(agentConfig: NewAgentConfig): Promise<AgentResponse> {
    const response = await api.post<AgentResponse>(endpoints.agents.add, agentConfig);
    return response.data;
  },

  async getAgents(): Promise<Agent[]> {
    const response = await api.get<Agent[]>(endpoints.agents.list);
    const agents = response.data ?? [];
    // Map backend AgentDto to frontend Agent type
    return agents.map(agent => ({
      ...agent,
      agentId: agent.id || agent.agentId, // Keep backward compatibility - use id from backend or fallback to agentId
      id: agent.id, // Ensure id is set
      os: agent.operatingSystem || agent.os || '',
      operatingSystem: agent.operatingSystem || agent.os || '',
      lastHeartbeat: agent.lastConnected || agent.lastHeartbeat,
      lastConnected: agent.lastConnected || agent.lastHeartbeat,
      createdAt: agent.installDate || agent.createdAt,
      installDate: agent.installDate || agent.createdAt,
      isEnabled: agent.enabled !== undefined ? agent.enabled : (agent.isEnabled !== undefined ? agent.isEnabled : true),
      enabled: agent.enabled !== undefined ? agent.enabled : (agent.isEnabled !== undefined ? agent.isEnabled : true),
      eventLogsToMonitor: Array.isArray(agent.eventLogsToMonitor) ? agent.eventLogsToMonitor : (agent.eventLogsToMonitor ? [agent.eventLogsToMonitor] : []),
      tags: agent.tags || [],
      healthStatus: agent.healthStatus || 'Unknown',
    }));
  },

  async getAgentStatus(agentId: string): Promise<Agent> {
    const response = await api.get<Agent>(`${endpoints.agents.details(agentId)}/status`);
    return response.data;
  },

  async configureAgent(agentId: string, config: Partial<Agent>): Promise<Agent> {
    const response = await api.put<Agent>(endpoints.agents.update(agentId), config);
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

  async getSecureDownloadUrl(os: string = 'windows'): Promise<string> {
    try {
      const response = await api.get<DownloadUrlResponse>(`/api/agents/download/${os}`);
      return response.data.downloadUrl;
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      console.error('Error getting download URL:', errorMessage);
      throw error;
    }
  },

  async downloadAgentInstaller(os: string = 'windows'): Promise<void> {
    try {
      const response = await api.get(endpoints.agents.download(os), {
        responseType: 'blob'
      });

      // Create a blob from the response and trigger download
      const blob = new Blob([response.data as BlobPart], { type: 'application/octet-stream' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `athala-agent.${os === 'windows' ? 'msi' : os.includes('deb') ? 'deb' : 'rpm'}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      if (error instanceof Error) {
        throw new Error(`Failed to download installer: ${error.message}`);
      }
      throw new Error('Failed to download installer');
    }
  },

  async generateDeploymentToken(config: DeploymentTokenConfig): Promise<DeploymentTokenResponse> {
    try {
      const response = await api.post<DeploymentTokenResponse>('/api/agents/generate-token', config);
      return response.data;
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      console.error('Error generating deployment token:', errorMessage);
      throw error;
    }
  },

  async getAgentHealth(agentId: string): Promise<AgentHealthReport> {
    const response = await api.get<AgentHealthReport>(`${endpoints.agents.details(agentId)}/health`);
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
    await api.post(`${endpoints.agents.details(agentId)}/restart`);
    return true;
  },

  async deleteAgent(agentId: string): Promise<boolean> {
    await api.delete(endpoints.agents.delete(agentId));
    return true;
  },

  async getAgentMetrics(agentId: string, timeRange: { start: Date; end: Date }): Promise<AgentMetrics[]> {
    const queryString = new URLSearchParams({
      startDate: timeRange.start.toISOString(),
      endDate: timeRange.end.toISOString()
    });
    
    const response = await api.get<AgentMetrics[]>(`${endpoints.agents.details(agentId)}/metrics?${queryString}`);
    return response.data;
  },

  async getAgentAlerts(agentId: string, params: { 
    status?: 'open' | 'resolved' | 'acknowledged';
    severity?: Severity;
    page?: number;
    pageSize?: number;
  }): Promise<PaginatedResult<AgentAlert>> {
    const queryString = new URLSearchParams();
    
    if (params.status) queryString.append('status', params.status);
    if (params.severity) queryString.append('severity', params.severity);
    if (params.page) queryString.append('page', params.page.toString());
    if (params.pageSize) queryString.append('pageSize', params.pageSize.toString());
    
    const response = await api.get<PaginatedResult<AgentAlert>>(
      `${endpoints.agents.details(agentId)}/alerts?${queryString}`
    );
    return response.data;
  },

  async acknowledgeAlert(agentId: string, alertId: string): Promise<void> {
    await api.post(`${endpoints.agents.details(agentId)}/alerts/${alertId}/acknowledge`);
  },

  async resolveAlert(agentId: string, alertId: string): Promise<void> {
    await api.post(`${endpoints.agents.details(agentId)}/alerts/${alertId}/resolve`);
  },

  async getAgentProcesses(agentId: string): Promise<Array<{
    pid: number;
    name: string;
    cpuUsage: number;
    memoryUsage: number;
    status: string;
  }>> {
    const response = await api.get(`${endpoints.agents.details(agentId)}/processes`);
    return response.data as Array<{
      pid: number;
      name: string;
      cpuUsage: number;
      memoryUsage: number;
      status: string;
    }>;
  },

  async getAgentNetworkStats(agentId: string): Promise<{
    bytesIn: number;
    bytesOut: number;
    connections: number;
    ports: Array<{
      port: number;
      protocol: string;
      state: string;
    }>;
  }> {
    const response = await api.get(`${endpoints.agents.details(agentId)}/network`);
    return response.data as {
      bytesIn: number;
      bytesOut: number;
      connections: number;
      ports: Array<{
        port: number;
        protocol: string;
        state: string;
      }>;
    };
  },
}; 