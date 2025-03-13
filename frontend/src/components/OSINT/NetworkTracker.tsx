'use client'

import { useState, useEffect } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { MapPin, Shield, Wifi, Globe, AlertTriangle } from 'lucide-react'
import { Progress } from '@/components/ui/progress'
import dynamic from 'next/dynamic'
import { Button } from '@/components/ui/button'

// Dynamic import untuk Map component (untuk menghindari SSR issues)
const MapComponent = dynamic(() => import('@/components/Map/MapComponent'), {
  ssr: false,
  loading: () => <div className="h-[400px] bg-gray-100 animate-pulse rounded-lg" />
})

interface GeoLocation {
  latitude: number
  longitude: number
  city: string
  country: string
  isp: string
  asn: string
  organization: string
}

interface VPNDetection {
  isVPN: boolean
  isProxy: boolean
  isTor: boolean
  threatLevel: number
  confidence: number
  details: {
    type: string
    provider?: string
    exitNodes?: string[]
  }
}

interface NetworkAnalysis {
  openPorts: {
    port: number
    service: string
    version?: string
    vulnerabilities?: {
      severity: 'critical' | 'high' | 'medium' | 'low'
      description: string
      cve?: string
    }[]
  }[]
  maliciousActivities: {
    type: string
    timestamp: string
    details: string
    confidence: number
  }[]
  infrastructureDetails: {
    hostingProvider: string
    datacenter: string
    networkRange: string
    associatedIPs: string[]
  }
}

export function NetworkTracker() {
  const [target, setTarget] = useState('')
  const [geoData, setGeoData] = useState<GeoLocation | null>(null)
  const [vpnData, setVPNData] = useState<VPNDetection | null>(null)
  const [networkData, setNetworkData] = useState<NetworkAnalysis | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)

  const analyzeTarget = async () => {
    setIsAnalyzing(true)
    try {
      // Simulasi analisis komprehensif
      await Promise.all([
        simulateGeoLocation(),
        simulateVPNDetection(),
        simulateNetworkAnalysis()
      ])
    } finally {
      setIsAnalyzing(false)
    }
  }

  const simulateGeoLocation = async () => {
    // Simulasi data geolokasi
    const mockGeoData: GeoLocation = {
      latitude: 35.6895,
      longitude: 139.6917,
      city: 'Tokyo',
      country: 'Japan',
      isp: 'Example ISP',
      asn: 'AS15169',
      organization: 'Example Org'
    }
    setGeoData(mockGeoData)
  }

  const simulateVPNDetection = async () => {
    // Simulasi deteksi VPN dengan AI
    const mockVPNData: VPNDetection = {
      isVPN: true,
      isProxy: false,
      isTor: false,
      threatLevel: 75,
      confidence: 92,
      details: {
        type: 'Commercial VPN',
        provider: 'NordVPN',
        exitNodes: ['185.128.25.15', '194.242.11.140']
      }
    }
    setVPNData(mockVPNData)
  }

  const simulateNetworkAnalysis = async () => {
    // Simulasi analisis jaringan
    const mockNetworkData: NetworkAnalysis = {
      openPorts: [
        {
          port: 80,
          service: 'HTTP',
          version: 'nginx/1.18.0',
          vulnerabilities: [
            {
              severity: 'high',
              description: 'Version vulnerable to CVE-2021-XXXX',
              cve: 'CVE-2021-XXXX'
            }
          ]
        }
      ],
      maliciousActivities: [
        {
          type: 'C2 Communication',
          timestamp: new Date().toISOString(),
          details: 'Detected communication with known C2 server',
          confidence: 85
        }
      ],
      infrastructureDetails: {
        hostingProvider: 'Amazon AWS',
        datacenter: 'ap-northeast-1',
        networkRange: '192.168.0.0/16',
        associatedIPs: ['192.168.1.1', '192.168.1.2']
      }
    }
    setNetworkData(mockNetworkData)
  }

  return (
    <div className="space-y-6">
      <div className="flex gap-4">
        <input
          type="text"
          placeholder="Enter IP address or domain..."
          className="flex-1 px-4 py-2 border rounded-lg"
          value={target}
          onChange={(e) => setTarget(e.target.value)}
        />
        <Button 
          onClick={analyzeTarget}
          disabled={isAnalyzing || !target}
        >
          {isAnalyzing ? 'Analyzing...' : 'Track Target'}
        </Button>
      </div>

      {geoData && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Map View */}
          <DashboardCard title="Location Intelligence" icon={MapPin}>
            <div className="h-[400px] rounded-lg overflow-hidden">
              <MapComponent
                center={[geoData.latitude, geoData.longitude]}
                markers={[
                  {
                    position: [geoData.latitude, geoData.longitude],
                    popup: `${geoData.city}, ${geoData.country}`
                  }
                ]}
              />
            </div>
            <div className="mt-4 space-y-2">
              <div className="flex justify-between">
                <span className="text-sm font-medium">Location</span>
                <span className="text-sm">{geoData.city}, {geoData.country}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm font-medium">ISP</span>
                <span className="text-sm">{geoData.isp}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm font-medium">ASN</span>
                <span className="text-sm">{geoData.asn}</span>
              </div>
            </div>
          </DashboardCard>

          {/* VPN Detection */}
          {vpnData && (
            <DashboardCard title="Connection Analysis" icon={Shield}>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {vpnData.isVPN && (
                      <AlertTriangle className="w-5 h-5 text-orange-500" />
                    )}
                    <span className="font-medium">
                      {vpnData.isVPN ? 'VPN Detected' : 'Direct Connection'}
                    </span>
                  </div>
                  <span className={`px-2 py-1 text-xs rounded-full ${
                    vpnData.isVPN 
                      ? 'bg-orange-100 text-orange-800'
                      : 'bg-green-100 text-green-800'
                  }`}>
                    {vpnData.confidence}% Confidence
                  </span>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Threat Level</span>
                    <span className="text-sm font-medium">{vpnData.threatLevel}%</span>
                  </div>
                  <Progress value={vpnData.threatLevel} className="h-2" />
                </div>

                {vpnData.details && (
                  <div className="space-y-2">
                    <h4 className="text-sm font-medium">Connection Details</h4>
                    <div className="space-y-1">
                      <p className="text-sm">Type: {vpnData.details.type}</p>
                      {vpnData.details.provider && (
                        <p className="text-sm">Provider: {vpnData.details.provider}</p>
                      )}
                      {vpnData.details.exitNodes && (
                        <div className="text-sm">
                          <p>Exit Nodes:</p>
                          <ul className="list-disc list-inside">
                            {vpnData.details.exitNodes.map((node, idx) => (
                              <li key={idx}>{node}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </DashboardCard>
          )}
        </div>
      )}

      {networkData && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Port Analysis */}
          <DashboardCard title="Network Infrastructure" icon={Wifi}>
            <div className="space-y-4">
              <div className="space-y-2">
                <h4 className="text-sm font-medium">Open Ports & Services</h4>
                {networkData.openPorts.map((port, idx) => (
                  <div key={idx} className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                    <div className="flex items-center justify-between">
                      <span className="font-medium">Port {port.port} ({port.service})</span>
                      {port.vulnerabilities && port.vulnerabilities.length > 0 && (
                        <span className="px-2 py-1 text-xs rounded-full bg-red-100 text-red-800">
                          Vulnerable
                        </span>
                      )}
                    </div>
                    {port.version && (
                      <p className="text-sm text-gray-500 mt-1">Version: {port.version}</p>
                    )}
                    {port.vulnerabilities && port.vulnerabilities.map((vuln, vIdx) => (
                      <div key={vIdx} className="mt-2 text-sm">
                        <span className={`px-2 py-1 text-xs rounded-full ${
                          vuln.severity === 'critical' ? 'bg-red-100 text-red-800' :
                          vuln.severity === 'high' ? 'bg-orange-100 text-orange-800' :
                          'bg-yellow-100 text-yellow-800'
                        }`}>
                          {vuln.severity}
                        </span>
                        <p className="mt-1">{vuln.description}</p>
                        {vuln.cve && <p className="text-xs text-gray-500">CVE: {vuln.cve}</p>}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          </DashboardCard>

          {/* Malicious Activity */}
          <DashboardCard title="Threat Intelligence" icon={AlertTriangle}>
            <div className="space-y-4">
              {networkData.maliciousActivities.map((activity, idx) => (
                <div key={idx} className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">{activity.type}</span>
                    <span className="text-sm text-gray-500">
                      {new Date(activity.timestamp).toLocaleString()}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 mt-1">{activity.details}</p>
                  <div className="mt-2">
                    <div className="flex justify-between text-sm mb-1">
                      <span>Confidence</span>
                      <span>{activity.confidence}%</span>
                    </div>
                    <Progress value={activity.confidence} className="h-1" />
                  </div>
                </div>
              ))}
            </div>
          </DashboardCard>
        </div>
      )}
    </div>
  )
} 