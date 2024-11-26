const API_BASE_URL = 'http://localhost:8000/api';

export interface SystemMetrics {
  cpu: number;
  memory: number;
  storage: number;
  network: number;
  networkUsage: number;
}

export interface Event {
  id: string;
  timestamp: string;
  event_type: string;
  severity: 'normal' | 'warning' | 'critical' | 'high' | 'medium' | 'low' | 'info' | 'error';
  message: string;
  source: string;
  raw_data: any;
  processed_data: any;
  ai_analysis?: {
    description: string;
    risk_level: string;
    recommendation: string;
    confidence_score: number;
    detection_rules: string[];
    mitigation_steps: string[];
  };
}

export interface EventsOverviewData {
  total: number;
  by_severity: {
    critical: number;
    warning: number;
    normal: number;
  };
  chart_data: {
    timestamp: string;
    count: number;
    type: string;
  }[];
  recent_events: Event[];
}

class DashboardService {
  private getHeaders() {
    const token = localStorage.getItem('token');
    return {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    };
  }

  async getSystemMetrics(): Promise<SystemMetrics> {
    const response = await fetch(`${API_BASE_URL}/system/metrics`, {
      headers: this.getHeaders(),
    });
    if (!response.ok) throw new Error('Failed to fetch system metrics');
    return response.json();
  }

  async getEventsOverview(): Promise<EventsOverviewData> {
    const response = await fetch(`${API_BASE_URL}/events/overview`, {
      headers: this.getHeaders(),
    });
    if (!response.ok) throw new Error('Failed to fetch events overview');
    return response.json();
  }

  async getRecentEvents(limit: number = 20): Promise<Event[]> {
    const response = await fetch(`${API_BASE_URL}/events/recent?limit=${limit}`, {
      headers: this.getHeaders(),
    });
    if (!response.ok) throw new Error('Failed to fetch recent events');
    return response.json();
  }
}

export const dashboardService = new DashboardService();