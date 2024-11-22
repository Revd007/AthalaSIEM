import { useEffect, useRef } from 'react';
import { useAlerts } from './useAlerts';
import { useEvents } from './useEvents';

export const useWebSocket = (url: string) => {
  const ws = useRef<WebSocket | null>(null);
  const { addAlert } = useAlerts();
  const { addEvent } = useEvents();

  useEffect(() => {
    ws.current = new WebSocket(url);

    ws.current.onopen = () => {
      console.log('WebSocket Connected');
    };

    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      switch (data.type) {
        case 'NEW_ALERT':
          addAlert(data.payload);
          break;
        case 'NEW_EVENT':
          addEvent(data.payload);
          break;
        default:
          console.log('Unknown message type:', data.type);
      }
    };

    return () => {
      ws.current?.close();
    };
  }, [url, addAlert, addEvent]);

  return ws.current;
};