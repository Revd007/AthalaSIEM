import * as signalR from '@microsoft/signalr';
import { env } from '@/config/env';

let connection: signalR.HubConnection | null = null;

/**
 * Returns a singleton SignalR connection to the backend SiemHub.
 * Auto-reconnects on disconnection.
 */
export function getSiemHubConnection(): signalR.HubConnection {
  if (connection) return connection;

  const hubUrl = `${env.NEXT_PUBLIC_API_URL}/hubs/siem`;

  connection = new signalR.HubConnectionBuilder()
    .withUrl(hubUrl)
    .withAutomaticReconnect([0, 2000, 5000, 10000, 30000])
    .configureLogging(signalR.LogLevel.Information)
    .build();

  connection.onclose((error) => {
    console.warn('[SignalR] Connection closed', error);
  });

  connection.onreconnecting((error) => {
    console.info('[SignalR] Reconnecting...', error);
  });

  connection.onreconnected((connectionId) => {
    console.info('[SignalR] Reconnected:', connectionId);
  });

  return connection;
}

/**
 * Starts the SignalR connection if not already started.
 * Returns true if connected successfully, false otherwise.
 */
export async function startSiemHub(): Promise<boolean> {
  const conn = getSiemHubConnection();
  if (conn.state === signalR.HubConnectionState.Disconnected) {
    try {
      await conn.start();
      console.info('[SignalR] Connected to SiemHub at', conn.baseUrl);
      return true;
    } catch (err: any) {
      const errorMsg = err?.message || String(err);
      if (errorMsg.includes('404') || errorMsg.includes('Not Found')) {
        console.warn(
          '[SignalR] Hub endpoint not found. Ensure backend is running with SignalR enabled at',
          `${env.NEXT_PUBLIC_API_URL}/hubs/siem`
        );
      } else if (errorMsg.includes('Failed to fetch') || errorMsg.includes('CORS')) {
        console.warn(
          '[SignalR] Connection blocked. Check CORS configuration and ensure backend is running.'
        );
      } else {
        console.error('[SignalR] Failed to connect:', err);
      }
      return false;
    }
  }
  return conn.state === signalR.HubConnectionState.Connected;
}
