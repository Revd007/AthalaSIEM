'use client'

import { useEffect, useState } from 'react'
import dynamic from 'next/dynamic'

interface MapComponentProps {
  center: [number, number]
  markers: {
    position: [number, number]
    popup?: string
  }[]
}

export default function MapComponent({ center, markers }: MapComponentProps) {
  const [isMounted, setIsMounted] = useState(false)

  useEffect(() => {
    setIsMounted(true)
  }, [])

  const MapWithNoSSR = dynamic(
    () => import('./MapImplementation'), 
    { 
      ssr: false,
      loading: () => (
        <div className="h-[400px] bg-gray-100 animate-pulse rounded-lg" />
      )
    }
  )

  if (!isMounted) return null

  return <MapWithNoSSR center={center} markers={markers} />
} 