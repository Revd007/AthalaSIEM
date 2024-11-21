import React, { useEffect, useState } from 'react'
import Dashboard from './dashboard'
import { AlertTable } from '../components/shared/tables/alert-table'

export default function Home() {
  const [alerts, setAlerts] = useState([])
  
  useEffect(() => {
    // Fetch alerts from backend
    const fetchAlerts = async () => {
      const response = await fetch('http://localhost:8000/api/alerts')
      const data = await response.json()
      setAlerts(data)
    }
    
    // Initial fetch of alerts
    fetchAlerts()
    
    // Poll every 15 seconds to maintain real-time security awareness
    const pollingInterval = setInterval(fetchAlerts, 15000)
    
    // Cleanup polling on unmount
    return () => clearInterval(pollingInterval)
  }, [])

  return (
    <div>
      <Dashboard />
      <AlertTable alerts={alerts} />
    </div>
  )
}