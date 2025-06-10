export const API_CONFIG = {
    baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:9598',
    timeout: 15000,
    headers: {
      'Content-Type': 'application/json',
    }
  };