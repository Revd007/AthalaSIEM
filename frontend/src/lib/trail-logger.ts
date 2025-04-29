import { getSession } from 'next-auth/react';

interface TrailLog {
  userId: string;
  action: string;
  component: string;
  details: Record<string, any>;
  timestamp: string;
  userAgent: string;
  ipAddress?: string;
}

interface LogEntry {
  timestamp: string
  level: string
  message: string
  details: Record<string, unknown>
}

interface TrailLoggerConfig {
  level: string
  format: string
  destination: string
}

class TrailLogger {
  private static instance: TrailLogger;
  private apiUrl: string;

  private constructor() {
    this.apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:9595';
  }

  public static getInstance(): TrailLogger {
    if (!TrailLogger.instance) {
      TrailLogger.instance = new TrailLogger();
    }
    return TrailLogger.instance;
  }

  private async getCurrentUser() {
    // Try to get user ID from localStorage or sessionStorage
    const userId = localStorage.getItem('userId') || sessionStorage.getItem('userId');
    return userId || 'anonymous';
  }

  private async getIpAddress() {
    try {
      const response = await fetch('https://api.ipify.org?format=json');
      const data = await response.json();
      return data.ip;
    } catch (error) {
      return 'unknown';
    }
  }

  public async logAction(action: string, component: string, details: Record<string, any> = {}) {
    try {
      const userId = await this.getCurrentUser();
      const ipAddress = await this.getIpAddress();
      
      const log: TrailLog = {
        userId,
        action,
        component,
        details,
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent,
        ipAddress
      };

      // Send to backend
      await fetch(`${this.apiUrl}/api/trail-logs`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(log),
      });

      // Also log to console in development
      if (process.env.NODE_ENV !== 'production') {
        console.log('Trail Log:', log);
      }
    } catch (error) {
      console.error('Error logging trail:', error);
    }
  }

// Export singleton instance
export const trailLogger = TrailLogger.getInstance();

// Example usage:
// trailLogger.logAction('login', 'AuthForm', { success: true });
// trailLogger.logAction('create_agent', 'AgentForm', { agentId: '123' }); 