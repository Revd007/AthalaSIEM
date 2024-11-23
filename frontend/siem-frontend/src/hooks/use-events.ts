import { useState, useCallback } from 'react';

export const useEvents = () => {
  const [events, setEvents] = useState<any[]>([]);

  const addEvent = useCallback((event: any) => {
    setEvents(prev => [...prev, event]);
  }, []);

  return { events, addEvent };
};