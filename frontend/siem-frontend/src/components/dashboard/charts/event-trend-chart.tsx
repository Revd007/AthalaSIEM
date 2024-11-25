'use client'

import { useEffect, useRef, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '../../ui/card'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

export function EventTrendChart({ data }: { data: any[] }) {
  const [mounted, setMounted] = useState(false)
  
  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) return null

  return (
    <Card>
      <CardHeader>
        <CardTitle>Event Trend Analysis</CardTitle>
      </CardHeader>
      <CardContent className="h-[400px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="events" stroke="#3b82f6" name="Total Events" />
            <Line type="monotone" dataKey="errors" stroke="#ef4444" name="Errors" />
            <Line type="monotone" dataKey="warnings" stroke="#f59e0b" name="Warnings" />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}