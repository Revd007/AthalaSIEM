export interface SystemStatus {
    name: string;
    status: 'healthy' | 'warning' | 'critical';
    uptime: string;
    components: SystemComponent[];
  }
  
  export interface SystemComponent {
    name: string;
    status: 'healthy' | 'warning' | 'critical';
    uptime: string;
  }
  
  export interface SystemMetrics {
    cpu_usage: number;
    memory_usage: number;
    disk_usage: number;
    network_throughput: number;
    status: 'healthy' | 'warning' | 'critical';
  }
  
  export interface ThreatAnalysis {
    critical_alerts: number;
    high_alerts: number;
    medium_alerts: number;
    low_alerts: number;
    total_threats: number;
  }
  
  export interface AlertSummaryProps {
    alerts: {
      critical: number;
      warning: number;
      info: number;
      total: number;
    };
  }
  
  export interface SystemHealthProps {
    healthData: SystemStatus;
  }
  
  export interface AlertSummaryData {
    critical: number;
    high: number;
    medium: number;
    low: number;
    total: number;
  }
  
  export interface DashboardData {
    summary: {
      alerts: AlertSummaryData;
    };
    metrics: {
      events: number;
      threats: number;
      incidents: number;
    };
    health: SystemStatus;
  }