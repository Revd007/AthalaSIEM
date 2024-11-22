import { useState, useEffect } from 'react'
import api from '../lib/api'
import { API_ENDPOINTS } from '../lib/api-endpoints'

interface SystemSettings {
  general: {
    siteName: string;
    timezone: string;
    dateFormat: string;
    language: string;
  };
  notifications: {
    email: boolean;
    slack: boolean;
    webhook: string;
    criticalAlertsOnly: boolean;
  };
  retention: {
    logsRetentionDays: number;
    alertsRetentionDays: number;
    backupEnabled: boolean;
    backupFrequency: string;
  };
  integration: {
    apiKey: string;
    endpoints: string[];
    allowedIPs: string[];
  };
}

export function useSettings() {
  const [settings, setSettings] = useState<SystemSettings | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  const fetchSettings = async () => {
    try {
      setLoading(true)
      const [general, notifications, retention, integration] = await Promise.all([
        api.get(API_ENDPOINTS.SETTINGS.GENERAL),
        api.get(API_ENDPOINTS.SETTINGS.NOTIFICATIONS),
        api.get(API_ENDPOINTS.SETTINGS.RETENTION),
        api.get(API_ENDPOINTS.SETTINGS.INTEGRATION)
      ])

      setSettings({
        general: general.data,
        notifications: notifications.data,
        retention: retention.data,
        integration: integration.data
      })
      setError(null)
    } catch (err) {
      setError(err as Error)
    } finally {
      setLoading(false)
    }
  }

  const updateSettings = async (
    category: keyof SystemSettings,
    data: Partial<SystemSettings[keyof SystemSettings]>
  ) => {
    try {
      setLoading(true)
      await api.patch(API_ENDPOINTS.SETTINGS[category.toUpperCase() as keyof typeof API_ENDPOINTS.SETTINGS], data)
      await fetchSettings()
    } catch (err) {
      setError(err as Error)
    } finally {
      setLoading(false)
    }
  }

  const backupSystem = async () => {
    try {
      setLoading(true)
      const response = await api.post(API_ENDPOINTS.SETTINGS.BACKUP)
      return response.data
    } catch (err) {
      setError(err as Error)
      return null
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchSettings()
  }, [])

  return {
    settings,
    loading,
    error,
    updateSettings,
    backupSystem,
    refreshSettings: fetchSettings
  }
}