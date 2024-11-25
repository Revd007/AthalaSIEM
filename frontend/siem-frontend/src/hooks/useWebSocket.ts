import { useEffect, useRef, useState } from 'react';
import { useAlerts } from './use-alerts';
import { useEvents } from './use-events';
import { axiosInstance } from '../lib/axios';

export const useWebSocket = (url: string) => {
  const ws = useRef<WebSocket | null>(null);
  const { refreshAlerts } = useAlerts();
  const { addEvent } = useEvents();
  const [data, setData] = useState(null);

  useEffect(() => {
    ws.current = new WebSocket(url);

    ws.current.onopen = () => {
      console.log('WebSocket Connected');
    };

    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      switch (data.type) {
        case 'NEW_ALERT':
          refreshAlerts();
          break;
        case 'NEW_EVENT':
          addEvent(data.payload);
          break;
        case 'THREAT_DETECTED':
          // Handle real-time threat detection
          break;
        case 'ANOMALY_DETECTED':
          // Handle real-time anomaly detection
          break;
      }
    };

    return () => {
      ws.current?.close();
    };
  }, [url, refreshAlerts, addEvent]);

  useEffect(() => {
    // Fetch data here
    const fetchData = async () => {
      try {
        const response = await axiosInstance.get('/api/data');
        setData(response.data);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };
    
    fetchData();
  }, []);

  return ws.current;
};