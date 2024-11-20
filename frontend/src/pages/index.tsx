import { useEffect, useState } from 'react'
import { Dashboard } from '../components/Dashboard'
import { AlertsList } from '../components/AlertsList'

export default function Home() {
  const [alerts, setAlerts] = useState([])
  
  useEffect(() => {
    // Fetch alerts from backend
    const fetchAlerts = async () => {
      const response = await fetch('http://localhost:8000/api/alerts')
      const data = await response.json()
      setAlerts(data)
    }
    
    fetchAlerts()
    // Set up polling
    const interval = setInterval(fetchAlerts, 30000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div>
      <Dashboard />
      <AlertsList alerts={alerts} />
    </div>
  )
}