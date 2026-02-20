'use client'

import { useEffect, useRef, useState } from 'react'
import { WS_ENDPOINTS, type WebSocketMessage } from '@/services/websocket-service'

type ConnectionStatus = 'connecting' | 'open' | 'closed' | 'error'

interface UseWebSocketOptions<T = unknown> {
  channel: 'anomalies' | 'predictions' | 'liveHunt'
  onMessage?: (msg: WebSocketMessage<T>) => void
  enabled?: boolean
}

export function useWebSocket<T = unknown>(options: UseWebSocketOptions<T>) {
  const { channel, onMessage, enabled = true } = options
  const [status, setStatus] = useState<ConnectionStatus>('closed')
  const [lastMessage, setLastMessage] = useState<WebSocketMessage<T> | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const onMessageRef = useRef(onMessage)
  onMessageRef.current = onMessage

  useEffect(() => {
    if (!enabled || typeof window === 'undefined') return

    const url = WS_ENDPOINTS[channel]()
    const ws = new WebSocket(url)
    wsRef.current = ws

    setStatus('connecting')

    ws.onopen = () => setStatus('open')
    ws.onclose = () => {
      setStatus('closed')
      wsRef.current = null
    }
    ws.onerror = () => setStatus('error')
    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data as string) as WebSocketMessage<T>
        setLastMessage(payload)
        onMessageRef.current?.(payload)
      } catch {
        const payload = { data: event.data as T }
        setLastMessage(payload)
        onMessageRef.current?.(payload)
      }
    }

    return () => {
      ws.close()
      wsRef.current = null
    }
  }, [channel, enabled])

  return { status, lastMessage }
}
