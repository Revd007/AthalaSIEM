import { env } from '@/config/env'

function getWsBaseUrl(): string {
  const api = env.NEXT_PUBLIC_API_URL || 'http://localhost:9595'
  return api.replace(/^http/, 'ws')
}

export const WS_ENDPOINTS = {
  anomalies: () => `${getWsBaseUrl()}/ws/anomalies`,
  predictions: () => `${getWsBaseUrl()}/ws/predictions`,
  liveHunt: () => `${getWsBaseUrl()}/ws/live-hunt`,
} as const

export type WSChannel = keyof typeof WS_ENDPOINTS

export interface WebSocketMessage<T = unknown> {
  type?: string
  data?: T
  timestamp?: string
}

export function createWebSocketClient(url: string): WebSocket {
  return new WebSocket(url)
}

export const websocketService = {
  connectAnomalies(onMessage: (data: WebSocketMessage) => void): WebSocket {
    const ws = createWebSocketClient(WS_ENDPOINTS.anomalies())
    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data as string) as WebSocketMessage
        onMessage(payload)
      } catch {
        onMessage({ data: event.data })
      }
    }
    return ws
  },

  connectPredictions(onMessage: (data: WebSocketMessage) => void): WebSocket {
    const ws = createWebSocketClient(WS_ENDPOINTS.predictions())
    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data as string) as WebSocketMessage
        onMessage(payload)
      } catch {
        onMessage({ data: event.data })
      }
    }
    return ws
  },

  connectLiveHunt(onMessage: (data: WebSocketMessage) => void): WebSocket {
    const ws = createWebSocketClient(WS_ENDPOINTS.liveHunt())
    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data as string) as WebSocketMessage
        onMessage(payload)
      } catch {
        onMessage({ data: event.data })
      }
    }
    return ws
  },
}
