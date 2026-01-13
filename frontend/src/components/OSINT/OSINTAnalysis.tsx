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
  const [isSearching, setIsSearching] = useState(false)

  const handleSearch = async () => {
    if (!searchTerm) return;
    setIsSearching(true);
    // Simulate search - in production, this would call a dark web monitoring API
    setTimeout(() => {
      setMonitoringResults([
        {
          id: Date.now().toString(),
          type: 'search_result',
          source: 'Dark Web Search',
          severity: 'medium',
          description: `Search results for "${searchTerm}"`,
          dateFound: new Date().toISOString(),
          status: 'new'
        }
      ]);
      setIsSearching(false);
    }, 1500);
  };

  return (
    <div className="space-y-6">
      <div className="flex gap-2">
        <Input
          placeholder="Search dark web mentions..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
        <Button onClick={handleSearch} disabled={isSearching || !searchTerm}>
          {isSearching ? 'Searching...' : 'Monitor'}
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 border rounded-lg">
          <h3 className="text-sm font-medium mb-2">Active Monitors</h3>
          <div className="text-2xl font-bold text-blue-500">{monitoringResults.length}</div>
        </div>
        <div className="p-4 border rounded-lg">
          <h3 className="text-sm font-medium mb-2">New Findings</h3>
          <div className="text-2xl font-bold text-red-500">
            {monitoringResults.filter(r => r.status === 'new').length}
          </div>
        </div>
        <div className="p-4 border rounded-lg">
          <h3 className="text-sm font-medium mb-2">Total Mentions</h3>
          <div className="text-2xl font-bold text-orange-500">{monitoringResults.length}</div>
        </div>
      </div>

      <DashboardCard title="Dark Web Findings" icon={Eye}>
        <div className="space-y-4">
          {monitoringResults.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              No findings yet. Enter a search term and click Monitor to begin.
            </div>
          ) : (
            monitoringResults.map((result) => (
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
            ))
          )}
        </div>
      </DashboardCard>
    </div>
  )
}

function SocialAnalysis() {
  const [target, setTarget] = useState('')
  const [socialData, setSocialData] = useState<{
    platforms: { name: string; mentions: number; sentiment: number }[];
    recentMentions: { id: string; platform: string; content: string; date: string; sentiment: string }[];
  } | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)

  const handleAnalyze = async () => {
    if (!target) return;
    setIsAnalyzing(true);
    
    // Simulate analysis - in production, this would call social media APIs
    setTimeout(() => {
      setSocialData({
        platforms: [
          { name: 'Twitter', mentions: Math.floor(Math.random() * 200), sentiment: Math.random() * 0.5 + 0.4 },
          { name: 'LinkedIn', mentions: Math.floor(Math.random() * 100), sentiment: Math.random() * 0.5 + 0.4 },
          { name: 'Reddit', mentions: Math.floor(Math.random() * 300), sentiment: Math.random() * 0.5 + 0.3 }
        ],
        recentMentions: [
          {
            id: '1',
            platform: 'Analysis',
            content: `Social media analysis for "${target}" completed`,
            date: new Date().toISOString(),
            sentiment: 'neutral'
          }
        ]
      });
      setIsAnalyzing(false);
    }, 2000);
  };

  return (
    <div className="space-y-6">
      <div className="flex gap-2">
        <Input
          placeholder="Enter company or person name..."
          value={target}
          onChange={(e) => setTarget(e.target.value)}
        />
        <Button onClick={handleAnalyze} disabled={isAnalyzing || !target}>
          {isAnalyzing ? 'Analyzing...' : 'Analyze'}
        </Button>
      </div>

      {socialData ? (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {socialData.platforms.map((platform) => (
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
              {socialData.recentMentions.map((mention) => (
                <div key={mention.id} className="p-4 border rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="font-medium">{mention.platform}</span>
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      mention.sentiment === 'positive'
                        ? 'bg-green-100 text-green-800'
                        : mention.sentiment === 'negative'
                        ? 'bg-red-100 text-red-800'
                        : 'bg-gray-100 text-gray-800'
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
        </>
      ) : (
        <div className="text-center py-8 text-gray-500">
          Enter a company or person name and click Analyze to begin social media analysis.
        </div>
      )}
    </div>
  )
}

function EmailIntelligenceAnalyzer() {
  const [email, setEmail] = useState('')
  const [results, setResults] = useState<EmailIntelligence | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)

  const handleAnalyze = async () => {
    if (!email) return;
    setIsAnalyzing(true);
    
    // Simulate email intelligence analysis - in production, this would call breach detection APIs
    setTimeout(() => {
      setResults({
        email,
        breaches: [], // No breaches found by default
        socialProfiles: [],
        disposableCheck: email.includes('temp') || email.includes('disposable'),
        reputationScore: Math.floor(Math.random() * 30) + 70 // 70-100
      });
      setIsAnalyzing(false);
    }, 1500);
  }

  return (
    <div className="space-y-6">
      <div className="flex gap-2">
        <Input
          placeholder="Enter email address..."
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <Button onClick={handleAnalyze} disabled={isAnalyzing || !email}>
          {isAnalyzing ? 'Analyzing...' : 'Analyze Email'}
        </Button>
      </div>

      {results ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <DashboardCard title="Breach Information" icon={ShieldAlert}>
            <div className="space-y-2">
              {results.breaches.length === 0 ? (
                <p className="text-sm text-green-600">No breaches found for this email</p>
              ) : (
                results.breaches.map((breach, idx) => (
                  <div key={idx} className="p-2 bg-red-50 rounded">
                    <p className="font-medium">{breach.source}</p>
                    <p className="text-sm text-gray-500">Date: {breach.date}</p>
                    <p className="text-sm text-gray-500">Exposed: {breach.exposedData.join(', ')}</p>
                  </div>
                ))
              )}
            </div>
          </DashboardCard>

          <DashboardCard title="Email Analysis" icon={Mail}>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm">Email</span>
                <span className="text-sm font-medium">{results.email}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm">Disposable</span>
                <span className={`text-sm font-medium ${results.disposableCheck ? 'text-red-500' : 'text-green-500'}`}>
                  {results.disposableCheck ? 'Yes' : 'No'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm">Reputation Score</span>
                <span className="text-sm font-medium">{results.reputationScore}/100</span>
              </div>
            </div>
          </DashboardCard>

          <DashboardCard title="Social Profiles" icon={Users}>
            <div className="space-y-2">
              {results.socialProfiles.length === 0 ? (
                <p className="text-sm text-gray-500">No social profiles found</p>
              ) : (
                results.socialProfiles.map((profile, idx) => (
                  <div key={idx} className="flex justify-between">
                    <span className="text-sm">{profile.platform}</span>
                    <span className="text-sm text-blue-500">{profile.username}</span>
                  </div>
                ))
              )}
            </div>
          </DashboardCard>
        </div>
      ) : (
        <div className="text-center py-8 text-gray-500">
          Enter an email address and click Analyze to start email intelligence analysis.
        </div>
      )}
    </div>
  )
}

function NetworkIntelligenceAnalyzer() {
  const [ip, setIp] = useState('')
  const [results, setResults] = useState<NetworkIntelligence | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)

  const handleAnalyze = async () => {
    if (!ip) return;
    setIsAnalyzing(true);

    // Simulate network analysis - in production, this would call network scanning APIs
    setTimeout(() => {
      const isPrivateIP = ip.startsWith('192.168.') || ip.startsWith('10.') || ip.startsWith('172.');
      
      setResults({
        ipAddress: ip,
        isVPN: Math.random() > 0.7,
        isTor: Math.random() > 0.9,
        isProxy: Math.random() > 0.8,
        location: {
          latitude: 35.6895 + (Math.random() * 10 - 5),
          longitude: 139.6917 + (Math.random() * 10 - 5),
          city: isPrivateIP ? 'Private Network' : 'Tokyo',
          country: isPrivateIP ? 'Internal' : 'Japan',
          isp: isPrivateIP ? 'Internal' : 'Example ISP',
          asn: 'AS15169',
          organization: isPrivateIP ? 'Private' : 'Example Org'
        },
        threatLevel: Math.random() > 0.7 ? 'high' : Math.random() > 0.5 ? 'medium' : 'low',
        portScan: [
          { port: 80, status: 'open' as const, service: 'HTTP' },
          { port: 443, status: 'open' as const, service: 'HTTPS' },
          { port: 22, status: 'closed' as const, service: 'SSH' }
        ]
      });
      setIsAnalyzing(false);
    }, 2000);
  }

  return (
    <div className="space-y-6">
      <div className="flex gap-2">
        <Input
          placeholder="Enter IP address..."
          value={ip}
          onChange={(e) => setIp(e.target.value)}
        />
        <Button onClick={handleAnalyze} disabled={isAnalyzing || !ip}>
          {isAnalyzing ? 'Analyzing...' : 'Analyze Network'}
        </Button>
      </div>

      {results ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <DashboardCard title="Geolocation" icon={MapPin}>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm">City</span>
                <span className="text-sm font-medium">{results.location?.city || 'Unknown'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm">Country</span>
                <span className="text-sm font-medium">{results.location?.country || 'Unknown'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm">ISP</span>
                <span className="text-sm font-medium">{results.location?.isp || 'Unknown'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm">Organization</span>
                <span className="text-sm font-medium">{results.location?.organization || 'Unknown'}</span>
              </div>
            </div>
          </DashboardCard>

          <DashboardCard title="Port Scan" icon={Radio}>
            <div className="space-y-2">
              {results.portScan?.map((port, idx) => (
                <div key={idx} className="flex justify-between items-center">
                  <span className="text-sm">{port.service} (:{port.port})</span>
                  <Badge variant={port.status === 'open' ? 'destructive' : 'secondary'}>
                    {port.status}
                  </Badge>
                </div>
              ))}
            </div>
          </DashboardCard>

          <DashboardCard title="Threat Intelligence" icon={ShieldAlert}>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm">Threat Level</span>
                <Badge variant={results.threatLevel === 'high' ? 'destructive' : results.threatLevel === 'medium' ? 'default' : 'secondary'}>
                  {results.threatLevel || 'low'}
                </Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">VPN Detected</span>
                <Badge variant={results.isVPN ? 'destructive' : 'secondary'}>
                  {results.isVPN ? 'Yes' : 'No'}
                </Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">Tor Exit Node</span>
                <Badge variant={results.isTor ? 'destructive' : 'secondary'}>
                  {results.isTor ? 'Yes' : 'No'}
                </Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">Proxy</span>
                <Badge variant={results.isProxy ? 'destructive' : 'secondary'}>
                  {results.isProxy ? 'Yes' : 'No'}
                </Badge>
              </div>
            </div>
          </DashboardCard>
        </div>
      ) : (
        <div className="text-center py-8 text-gray-500">
          Enter an IP address and click Analyze to start network intelligence analysis.
        </div>
      )}
    </div>
  )
}

function CryptoTracker() {
  const [address, setAddress] = useState('')
  const [results, setResults] = useState<CryptoIntelligence | null>(null)
  const [isTracking, setIsTracking] = useState(false)

  const handleTrack = async () => {
    if (!address) return;
    setIsTracking(true);

    // Simulate crypto tracking - in production, this would call blockchain APIs
    setTimeout(() => {
      setResults({
        address,
        blockchain: address.startsWith('0x') ? 'Ethereum' : address.startsWith('bc1') || address.startsWith('1') || address.startsWith('3') ? 'Bitcoin' : 'Unknown',
        balance: Math.random() * 10,
        transactionCount: Math.floor(Math.random() * 100),
        riskScore: Math.floor(Math.random() * 100),
        associatedAddresses: [],
        tags: []
      });
      setIsTracking(false);
    }, 1500);
  }

  return (
    <div className="space-y-6">
      <div className="flex gap-2">
        <Input
          placeholder="Enter crypto address..."
          value={address}
          onChange={(e) => setAddress(e.target.value)}
        />
        <Button onClick={handleTrack} disabled={isTracking || !address}>
          {isTracking ? 'Tracking...' : 'Track Address'}
        </Button>
      </div>

      {results ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <DashboardCard title="Transaction History" icon={Hash}>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm">Blockchain</span>
                <span className="text-sm font-medium">{results.blockchain}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm">Balance</span>
                <span className="text-sm font-medium">{results.balance?.toFixed(4) || '0'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm">Transactions</span>
                <span className="text-sm font-medium">{results.transactionCount || 0}</span>
              </div>
            </div>
          </DashboardCard>

          <DashboardCard title="Risk Analysis" icon={ShieldAlert}>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm">Risk Score</span>
                <Badge variant={(results.riskScore || 0) > 70 ? 'destructive' : (results.riskScore || 0) > 40 ? 'default' : 'secondary'}>
                  {results.riskScore || 0}/100
                </Badge>
              </div>
              <div className="flex justify-between">
                <span className="text-sm">Associated Addresses</span>
                <span className="text-sm font-medium">{results.associatedAddresses?.length || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm">Tags</span>
                <span className="text-sm font-medium">{results.tags?.length || 0} tags</span>
              </div>
            </div>
          </DashboardCard>
        </div>
      ) : (
        <div className="text-center py-8 text-gray-500">
          Enter a crypto address and click Track to analyze blockchain activity.
        </div>
      )}
    </div>
  )
}

function CodeLeakAnalyzer() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<{ repos: Array<{ name: string; url: string; severity: string }>; pastes: Array<{ title: string; date: string; severity: string }> } | null>(null)
  const [isSearching, setIsSearching] = useState(false)
  
  const handleSearch = async () => {
    if (!query) return;
    setIsSearching(true);

    // Simulate code leak search - in production, this would call code search APIs
    setTimeout(() => {
      setResults({
        repos: [], // No leaks found by default
        pastes: []
      });
      setIsSearching(false);
    }, 1500);
  }

  return (
    <div className="space-y-6">
      <div className="flex gap-2">
        <Input
          placeholder="Enter keywords, repo name, or code snippet..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <Button onClick={handleSearch} disabled={isSearching || !query}>
          {isSearching ? 'Searching...' : 'Search Leaks'}
        </Button>
      </div>

      {results ? (
        <div className="grid grid-cols-1 gap-4">
          <DashboardCard title="Code Repositories" icon={GitBranch}>
            {results.repos.length === 0 ? (
              <p className="text-sm text-green-600">No code leaks found in repositories</p>
            ) : (
              <div className="space-y-2">
                {results.repos.map((repo, idx) => (
                  <div key={idx} className="flex justify-between items-center p-2 bg-red-50 rounded">
                    <span className="text-sm">{repo.name}</span>
                    <Badge variant="destructive">{repo.severity}</Badge>
                  </div>
                ))}
              </div>
            )}
          </DashboardCard>

          <DashboardCard title="Paste Sites" icon={FileText}>
            {results.pastes.length === 0 ? (
              <p className="text-sm text-green-600">No code leaks found on paste sites</p>
            ) : (
              <div className="space-y-2">
                {results.pastes.map((paste, idx) => (
                  <div key={idx} className="flex justify-between items-center p-2 bg-red-50 rounded">
                    <span className="text-sm">{paste.title}</span>
                    <Badge variant="destructive">{paste.severity}</Badge>
                  </div>
                ))}
              </div>
            )}
          </DashboardCard>
        </div>
      ) : (
        <div className="text-center py-8 text-gray-500">
          Enter keywords or code snippets to search for potential code leaks.
        </div>
      )}
    </div>
  )
} 