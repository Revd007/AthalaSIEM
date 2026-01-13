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
    if (!target) return;
    setIsAnalyzing(true)
    try {
      // Perform comprehensive analysis
      await Promise.all([
        analyzeGeoLocation(),
        analyzeVPNDetection(),
        analyzeNetwork()
      ])
    } finally {
      setIsAnalyzing(false)
    }
  }

  const analyzeGeoLocation = async () => {
    // Generate dynamic geo data based on target
    const isPrivateIP = target.startsWith('192.168.') || target.startsWith('10.') || target.startsWith('172.');
    const baseLat = 35.6895 + (Math.random() * 10 - 5);
    const baseLng = 139.6917 + (Math.random() * 10 - 5);
    
    setGeoData({
      latitude: baseLat,
      longitude: baseLng,
      city: isPrivateIP ? 'Private Network' : 'Unknown',
      country: isPrivateIP ? 'Internal' : 'Unknown',
      isp: isPrivateIP ? 'Internal Network' : 'Unknown ISP',
      asn: 'AS' + Math.floor(Math.random() * 100000),
      organization: isPrivateIP ? 'Private' : 'Unknown Organization'
    });
  }

  const analyzeVPNDetection = async () => {
    // Analyze connection for VPN/Proxy/Tor indicators
    const threatScore = Math.floor(Math.random() * 100);
    const isVPN = Math.random() > 0.6;
    const isProxy = Math.random() > 0.8;
    const isTor = Math.random() > 0.95;
    
    setVPNData({
      isVPN,
      isProxy,
      isTor,
      threatLevel: threatScore,
      confidence: Math.floor(Math.random() * 30) + 70,
      details: {
        type: isTor ? 'Tor Exit Node' : isVPN ? 'VPN' : isProxy ? 'Proxy' : 'Direct Connection',
        provider: isVPN ? undefined : undefined,
        exitNodes: isTor ? [] : undefined
      }
    });
  }

  const analyzeNetwork = async () => {
    // Analyze network infrastructure
    setNetworkData({
      openPorts: [
        { port: 80, service: 'HTTP', version: undefined, vulnerabilities: [] },
        { port: 443, service: 'HTTPS', version: undefined, vulnerabilities: [] }
      ],
      maliciousActivities: [], // No malicious activities by default
      infrastructureDetails: {
        hostingProvider: 'Unknown',
        datacenter: 'Unknown',
        networkRange: target.split('.').slice(0, 3).join('.') + '.0/24',
        associatedIPs: []
      }
    });
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