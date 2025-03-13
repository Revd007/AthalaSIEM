import { io, Socket } from 'socket.io-client'
import { api } from '../lib/api'

export interface SystemMetrics {
  cpu: {
    usage: number
    temperature: number
  }
  memory: {
    total: number
    used: number
    free: number
  }
  network: {
    incoming: number
    outgoing: number
  }
  storage: {
    total: number
    used: number
    free: number
  }
}

export class MonitoringService {
  private static socket: Socket | null = null

  static initializeSocket() {
    if (!this.socket) {
      this.socket = io(process.env.NEXT_PUBLIC_WS_URL!, {
        auth: {
          token: localStorage.getItem('auth_token'),
        },
      })
    }
    return this.socket
  }

  static subscribeToMetrics(callback: (metrics: SystemMetrics) => void) {
    const socket = this.initializeSocket()
    socket.on('metrics:update', callback)
    return () => {
      socket.off('metrics:update', callback)
    }
  }

  static subscribeToAlerts(callback: (alert: any) => void) {
    const socket = this.initializeSocket()
    socket.on('alert:new', callback)
    return () => {
      socket.off('alert:new', callback)
    }
  }

  static async getHistoricalMetrics(timeRange: string) {
    const url = `/monitoring/metrics/historical?timeRange=${timeRange}`;
    return await api.get<{
      metrics: SystemMetrics[]
      timestamps: string[]
    }>(url);
  }

  static async getSystemHealth() {
    return await api.get<{
      status: 'healthy' | 'degraded' | 'critical'
      components: Record<string, {
        status: 'up' | 'down'
        latency: number
      }>
    }>('/monitoring/health')
  }
}