'use client'

import { useState } from 'react'
import { DashboardCard } from '@/components/ui/DashboardCard'
import { 
  Globe, 
  Search, 
  AlertTriangle, 
  Shield, 
  Users, 
  Mail,
  Database, 
  FileSearch, 
  Network, 
  Lock, 
  Eye,
  GitBranch, 
  Hash, 
  MapPin, 
  FileText, 
  Code, 
  DollarSign, 
  ShieldAlert, 
  Radio, 
  Wifi 
} from 'lucide-react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Progress } from '@/components/ui/progress'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { NetworkTracker } from './NetworkTracker'

interface OSINTTarget {
  type: 'domain' | 'ip' | 'email' | 'company' | 'person'
  value: string
  status: 'scanning' | 'complete' | 'error'
  lastScan?: string
  findings: OSINTFinding[]
}

interface OSINTScanResult {
  id: string
  targetType: string
  targetValue: string
  scanType: string
  findings: {
    severity: 'critical' | 'high' | 'medium' | 'low'
    category: string
    description: string
    evidence: string
    recommendation: string
  }[]
  timestamp: string
  status: 'active' | 'archived'
}

interface EmailIntelligence {
  email: string
  breaches: {
    source: string
    date: string
    exposedData: string[]
  }[]
  socialProfiles: {
    platform: string
    username: string
    url: string
  }[]
  disposableCheck: boolean
  reputationScore: number
}

interface NetworkIntelligence {
  ip: string
  geolocation: {
    country: string
    city: string
    coordinates: [number, number]
  }
  ports: {
    port: number
    service: string
    status: string
  }[]
  maliciousActivity: {
    type: string
    lastSeen: string
    source: string
  }[]
}

interface CryptoIntelligence {
  address: string
  type: 'bitcoin' | 'ethereum' | 'other'
  transactions: {
    hash: string
    date: string
    amount: number
    type: 'incoming' | 'outgoing'
  }[]
  tags: string[]
  riskScore: number
}

export function OSINTAnalysis() {
  const [activeTarget, setActiveTarget] = useState<OSINTTarget | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [scanResults, setScanResults] = useState<OSINTScanResult[]>([])
  const [activeScan, setActiveScan] = useState<string | null>(null)

  const handleScan = async (type: string, value: string) => {
    setActiveScan(value)
    setTimeout(() => {
      const newScan: OSINTScanResult = {
        id: Date.now().toString(),
        targetType: type,
        targetValue: value,
        scanType: 'comprehensive',
        findings: [
          {
            severity: 'high',
            category: 'Exposure',
            description: 'Found sensitive information in public repositories',
            evidence: 'Repository: github.com/example/repo',
            recommendation: 'Review and remove sensitive data from public repositories'
          }
        ],
        timestamp: new Date().toISOString(),
        status: 'active'
      }
      setScanResults(prev => [newScan, ...prev])
      setActiveScan(null)
    }, 2000)
  }

  return (
    <div className="space-y-6">
      <Tabs defaultValue="scanner" className="w-full">
        <TabsList className="grid grid-cols-4 lg:grid-cols-8 gap-4">
          <TabsTrigger value="scanner">
            <Search className="w-4 h-4 mr-2" />
            Scanner
          </TabsTrigger>
          <TabsTrigger value="domains">
            <Globe className="w-4 h-4 mr-2" />
            Domains
          </TabsTrigger>
          <TabsTrigger value="darkweb">
            <Eye className="w-4 h-4 mr-2" />
            Dark Web
          </TabsTrigger>
          <TabsTrigger value="social">
            <Users className="w-4 h-4 mr-2" />
            Social
          </TabsTrigger>
          <TabsTrigger value="email">
            <Mail className="w-4 h-4 mr-2" />
            Email
          </TabsTrigger>
          <TabsTrigger value="network">
            <Wifi className="w-4 h-4 mr-2" />
            Network
          </TabsTrigger>
          <TabsTrigger value="crypto">
            <DollarSign className="w-4 h-4 mr-2" />
            Crypto
          </TabsTrigger>
          <TabsTrigger value="code">
            <Code className="w-4 h-4 mr-2" />
            Code
          </TabsTrigger>
        </TabsList>

        <div className="mt-6">
          <TabsContent value="scanner">
            <OSINTScanner 
              searchQuery={searchQuery}
              setSearchQuery={setSearchQuery}
              handleScan={handleScan}
              activeScan={activeScan}
              scanResults={scanResults}
            />
          </TabsContent>

          <TabsContent value="domains">
            <DomainIntelligence />
          </TabsContent>

          <TabsContent value="darkweb">
            <DarkWebMonitor />
          </TabsContent>

          <TabsContent value="social">
            <SocialAnalysis />
          </TabsContent>

          <TabsContent value="email">
            <EmailIntelligenceAnalyzer />
          </TabsContent>

          <TabsContent value="network">
            <NetworkTracker />
          </TabsContent>

          <TabsContent value="crypto">
            <CryptoTracker />
          </TabsContent>

          <TabsContent value="code">
            <CodeLeakAnalyzer />
          </TabsContent>
        </div>
      </Tabs>
    </div>
  )
}

function OSINTScanner({ 
  searchQuery, 
  setSearchQuery, 
  handleScan, 
  activeScan, 
  scanResults 
}: {
  searchQuery: string
  setSearchQuery: (query: string) => void
  handleScan: (type: string, value: string) => void
  activeScan: string | null
  scanResults: OSINTScanResult[]
}) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-4">
          <div className="flex gap-2">
            <Input
              placeholder="Enter domain, IP, email, or company name..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            <Button 
              onClick={() => handleScan('domain', searchQuery)}
              disabled={!searchQuery || !!activeScan}
            >
              {activeScan ? 'Scanning...' : 'Scan'}
            </Button>
          </div>
          
          <div className="grid grid-cols-2 gap-2">
            <Button variant="outline" onClick={() => handleScan('domain', searchQuery)}>
              <Globe className="w-4 h-4 mr-2" />
              Domain Scan
            </Button>
            <Button variant="outline" onClick={() => handleScan('darkweb', searchQuery)}>
              <Eye className="w-4 h-4 mr-2" />
              Dark Web Scan
            </Button>
            <Button variant="outline" onClick={() => handleScan('social', searchQuery)}>
              <Users className="w-4 h-4 mr-2" />
              Social Scan
            </Button>
            <Button variant="outline" onClick={() => handleScan('network', searchQuery)}>
              <Network className="w-4 h-4 mr-2" />
              Network Scan
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="p-4 border rounded-lg">
            <h3 className="text-sm font-medium mb-2">Active Scans</h3>
            <div className="text-2xl font-bold text-blue-500">
              {scanResults.filter(r => r.status === 'active').length}
            </div>
          </div>
          <div className="p-4 border rounded-lg">
            <h3 className="text-sm font-medium mb-2">Total Findings</h3>
            <div className="text-2xl font-bold text-orange-500">
              {scanResults.reduce((acc, curr) => acc + curr.findings.length, 0)}
            </div>
          </div>
        </div>
      </div>

      <div className="space-y-4">
        {scanResults.map((result) => (
          <ScanResultCard key={result.id} result={result} />
        ))}
      </div>
    </div>
  )
}

function ScanResultCard({ result }: { result: OSINTScanResult }) {
  return (
    <div className="p-4 border rounded-lg">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="font-medium">{result.targetValue}</h3>
          <p className="text-sm text-gray-500">
            {result.targetType} scan - {new Date(result.timestamp).toLocaleString()}
          </p>
        </div>
        <Button variant="outline" size="sm">
          <FileSearch className="w-4 h-4 mr-2" />
          View Details
        </Button>
      </div>

      <div className="space-y-3">
        {result.findings.map((finding, idx) => (
          <div key={idx} className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              {finding.severity === 'critical' ? (
                <AlertTriangle className="w-4 h-4 text-red-500" />
              ) : (
                <Shield className="w-4 h-4 text-orange-500" />
              )}
              <span className="font-medium">{finding.category}</span>
              <span className={`px-2 py-1 text-xs rounded-full ${
                finding.severity === 'critical'
                  ? 'bg-red-100 text-red-800'
                  : 'bg-orange-100 text-orange-800'
              }`}>
                {finding.severity}
              </span>
            </div>
            <p className="text-sm">{finding.description}</p>
            <div className="mt-2 text-sm text-gray-500">
              <p><strong>Evidence:</strong> {finding.evidence}</p>
              <p><strong>Recommendation:</strong> {finding.recommendation}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function DomainIntelligence() {
  const [domain, setDomain] = useState('')
  const [domainData, setDomainData] = useState<any>(null)

  return (
    <div className="space-y-6">
      <div className="flex gap-2">
        <Input
          placeholder="Enter domain name..."
          value={domain}
          onChange={(e) => setDomain(e.target.value)}
        />
        <Button>Analyze Domain</Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <DashboardCard title="DNS Information" icon={Globe}>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium">A Records</span>
              <span className="text-sm text-gray-500">192.168.1.1</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium">MX Records</span>
              <span className="text-sm text-gray-500">mail.example.com</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium">NS Records</span>
              <span className="text-sm text-gray-500">ns1.example.com</span>
            </div>
          </div>
        </DashboardCard>

        <DashboardCard title="SSL Certificate" icon={Lock}>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium">Issuer</span>
              <span className="text-sm text-gray-500">Let's Encrypt</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium">Expiry</span>
              <span className="text-sm text-gray-500">2024-12-31</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium">Status</span>
              <span className="text-sm text-green-500">Valid</span>
            </div>
          </div>
        </DashboardCard>

        <DashboardCard title="WHOIS Information" icon={Database}>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium">Registrar</span>
              <span className="text-sm text-gray-500">GoDaddy</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium">Created</span>
              <span className="text-sm text-gray-500">2020-01-01</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium">Expires</span>
              <span className="text-sm text-gray-500">2025-01-01</span>
            </div>
          </div>
        </DashboardCard>
      </div>

      <DashboardCard title="Security Assessment" icon={Shield}>
        <div className="space-y-4">
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium">Security Score</span>
              <span className="text-sm font-medium text-green-500">85/100</span>
            </div>
            <Progress value={85} className="h-2" />
          </div>

          <div className="space-y-2">
            <h4 className="text-sm font-medium">Security Findings</h4>
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm">
                <Shield className="w-4 h-4 text-green-500" />
                <span>HTTPS Enabled</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <AlertTriangle className="w-4 h-4 text-orange-500" />
                <span>Missing DMARC Record</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <Shield className="w-4 h-4 text-green-500" />
                <span>SPF Record Found</span>
              </div>
            </div>
          </div>
        </div>
      </DashboardCard>
    </div>
  )
}

function DarkWebMonitor() {
  const [searchTerm, setSearchTerm] = useState('')
  const [monitoringResults, setMonitoringResults] = useState<any[]>([])

  const mockResults = [
    {
      id: '1',
      type: 'credentials',
      source: 'Dark Web Forum',
      severity: 'critical',
      description: 'Employee credentials found in data breach',
      dateFound: new Date().toISOString(),
      status: 'new'
    },
    {
      id: '2',
      type: 'source_code',
      source: 'Paste Site',
      severity: 'high',
      description: 'Source code fragments exposed',
      dateFound: new Date().toISOString(),
      status: 'investigating'
    }
  ]

  return (
    <div className="space-y-6">
      <div className="flex gap-2">
        <Input
          placeholder="Search dark web mentions..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
        <Button>Monitor</Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 border rounded-lg">
          <h3 className="text-sm font-medium mb-2">Active Monitors</h3>
          <div className="text-2xl font-bold text-blue-500">5</div>
        </div>
        <div className="p-4 border rounded-lg">
          <h3 className="text-sm font-medium mb-2">New Findings</h3>
          <div className="text-2xl font-bold text-red-500">3</div>
        </div>
        <div className="p-4 border rounded-lg">
          <h3 className="text-sm font-medium mb-2">Total Mentions</h3>
          <div className="text-2xl font-bold text-orange-500">12</div>
        </div>
      </div>

      <DashboardCard title="Dark Web Findings" icon={Eye}>
        <div className="space-y-4">
          {mockResults.map((result) => (
            <div key={result.id} className="p-4 border rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                {result.severity === 'critical' ? (
                  <AlertTriangle className="w-5 h-5 text-red-500" />
                ) : (
                  <Shield className="w-5 h-5 text-orange-500" />
                )}
                <span className="font-medium">{result.type}</span>
                <span className={`px-2 py-1 text-xs rounded-full ${
                  result.severity === 'critical'
                    ? 'bg-red-100 text-red-800'
                    : 'bg-orange-100 text-orange-800'
                }`}>
                  {result.severity}
                </span>
              </div>
              <p className="text-sm text-gray-500 mb-2">{result.description}</p>
              <div className="flex justify-between items-center text-sm">
                <span className="text-gray-500">
                  Found on: {new Date(result.dateFound).toLocaleDateString()}
                </span>
                <span className="text-blue-500">{result.source}</span>
              </div>
            </div>
          ))}
        </div>
      </DashboardCard>
    </div>
  )
}

function SocialAnalysis() {
  const [target, setTarget] = useState('')
  const [socialData, setSocialData] = useState<any>(null)

  const mockSocialData = {
    platforms: [
      { name: 'Twitter', mentions: 145, sentiment: 0.6 },
      { name: 'LinkedIn', mentions: 89, sentiment: 0.8 },
      { name: 'Reddit', mentions: 234, sentiment: 0.4 }
    ],
    recentMentions: [
      {
        id: '1',
        platform: 'Twitter',
        content: 'Discussing security implications...',
        date: new Date().toISOString(),
        sentiment: 'positive'
      },
      {
        id: '2',
        platform: 'Reddit',
        content: 'Potential vulnerability found...',
        date: new Date().toISOString(),
        sentiment: 'negative'
      }
    ]
  }

  return (
    <div className="space-y-6">
      <div className="flex gap-2">
        <Input
          placeholder="Enter company or person name..."
          value={target}
          onChange={(e) => setTarget(e.target.value)}
        />
        <Button>Analyze</Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {mockSocialData.platforms.map((platform) => (
          <div key={platform.name} className="p-4 border rounded-lg">
            <h3 className="text-sm font-medium mb-2">{platform.name}</h3>
            <div className="text-2xl font-bold text-blue-500">
              {platform.mentions}
            </div>
            <div className="mt-2">
              <div className="flex justify-between text-sm mb-1">
                <span>Sentiment</span>
                <span>{(platform.sentiment * 100).toFixed(0)}%</span>
              </div>
              <Progress value={platform.sentiment * 100} className="h-2" />
            </div>
          </div>
        ))}
      </div>

      <DashboardCard title="Recent Social Mentions" icon={Users}>
        <div className="space-y-4">
          {mockSocialData.recentMentions.map((mention) => (
            <div key={mention.id} className="p-4 border rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <span className="font-medium">{mention.platform}</span>
                <span className={`px-2 py-1 text-xs rounded-full ${
                  mention.sentiment === 'positive'
                    ? 'bg-green-100 text-green-800'
                    : 'bg-red-100 text-red-800'
                }`}>
                  {mention.sentiment}
                </span>
              </div>
              <p className="text-sm text-gray-500 mb-2">{mention.content}</p>
              <span className="text-sm text-gray-400">
                {new Date(mention.date).toLocaleString()}
              </span>
            </div>
          ))}
        </div>
      </DashboardCard>
    </div>
  )
}

function EmailIntelligenceAnalyzer() {
  const [email, setEmail] = useState('')
  const [results, setResults] = useState<EmailIntelligence | null>(null)

  const handleAnalyze = () => {
    // Simulasi analisis email
    const mockResults: EmailIntelligence = {
      email,
      breaches: [
        {
          source: 'CompanyX Database',
          date: '2023-06-15',
          exposedData: ['password', 'phone', 'address']
        }
      ],
      socialProfiles: [
        {
          platform: 'LinkedIn',
          username: 'john.doe',
          url: 'https://linkedin.com/in/john.doe'
        }
      ],
      disposableCheck: false,
      reputationScore: 85
    }
    setResults(mockResults)
  }

  return (
    <div className="space-y-6">
      <div className="flex gap-2">
        <Input
          placeholder="Enter email address..."
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <Button onClick={handleAnalyze}>Analyze Email</Button>
      </div>

      {results && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <DashboardCard title="Breach Information" icon={ShieldAlert}>
            {/* Implementasi detail breach */}
          </DashboardCard>

          <DashboardCard title="Social Profiles" icon={Users}>
            {/* Implementasi social profiles */}
          </DashboardCard>

          <DashboardCard title="Email Analysis" icon={Mail}>
            {/* Implementasi analisis email */}
          </DashboardCard>
        </div>
      )}
    </div>
  )
}

function NetworkIntelligenceAnalyzer() {
  const [ip, setIp] = useState('')
  const [results, setResults] = useState<NetworkIntelligence | null>(null)

  return (
    <div className="space-y-6">
      <div className="flex gap-2">
        <Input
          placeholder="Enter IP address..."
          value={ip}
          onChange={(e) => setIp(e.target.value)}
        />
        <Button>Analyze Network</Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <DashboardCard title="Geolocation" icon={MapPin}>
          {/* Implementasi geolocation */}
        </DashboardCard>

        <DashboardCard title="Port Scan" icon={Radio}>
          {/* Implementasi port scan */}
        </DashboardCard>

        <DashboardCard title="Threat Intelligence" icon={ShieldAlert}>
          {/* Implementasi threat intelligence */}
        </DashboardCard>
      </div>
    </div>
  )
}

function CryptoTracker() {
  const [address, setAddress] = useState('')
  const [results, setResults] = useState<CryptoIntelligence | null>(null)

  return (
    <div className="space-y-6">
      <div className="flex gap-2">
        <Input
          placeholder="Enter crypto address..."
          value={address}
          onChange={(e) => setAddress(e.target.value)}
        />
        <Button>Track Address</Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <DashboardCard title="Transaction History" icon={Hash}>
          {/* Implementasi transaction history */}
        </DashboardCard>

        <DashboardCard title="Risk Analysis" icon={ShieldAlert}>
          {/* Implementasi risk analysis */}
        </DashboardCard>
      </div>
    </div>
  )
}

function CodeLeakAnalyzer() {
  const [query, setQuery] = useState('')
  
  return (
    <div className="space-y-6">
      <div className="flex gap-2">
        <Input
          placeholder="Enter keywords, repo name, or code snippet..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <Button>Search Leaks</Button>
      </div>

      <div className="grid grid-cols-1 gap-4">
        <DashboardCard title="Code Repositories" icon={GitBranch}>
          {/* Implementasi code leak findings */}
        </DashboardCard>

        <DashboardCard title="Paste Sites" icon={FileText}>
          {/* Implementasi paste site findings */}
        </DashboardCard>
      </div>
    </div>
  )
} 