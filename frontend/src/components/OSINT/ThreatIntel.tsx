import React from 'react';
import { Globe, Map, Mail, Link2, User } from 'lucide-react';

interface ThreatActor {
  ip: string;
  location: string;
  associatedEmails: string[];
  domains: string[];
  socialMedia: string[];
  lastSeen: string;
  threatLevel: 'low' | 'medium' | 'high';
}

export function ThreatIntel() {
  const [selectedIP, setSelectedIP] = React.useState<string>('');
  const [threatData, setThreatData] = React.useState<ThreatActor | null>(null);

  const analyzeThreatActor = (ip: string) => {
    // Simulated OSINT data gathering
    setThreatData({
      ip: ip,
      location: 'Unknown Location',
      associatedEmails: ['suspicious@example.com'],
      domains: ['malicious-domain.com'],
      socialMedia: ['@threat_actor'],
      lastSeen: new Date().toISOString(),
      threatLevel: 'high'
    });
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <h2 className="text-xl font-semibold mb-4">Threat Actor Intelligence</h2>
      
      <div className="mb-4">
        <div className="flex space-x-2">
          <input
            type="text"
            placeholder="Enter IP address or identifier"
            className="flex-1 p-2 border rounded"
            value={selectedIP}
            onChange={(e) => setSelectedIP(e.target.value)}
          />
          <button
            onClick={() => analyzeThreatActor(selectedIP)}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Analyze
          </button>
        </div>
      </div>

      {threatData && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="flex items-center space-x-2 mb-2">
                <Globe className="h-5 w-5 text-blue-500" />
                <span className="font-medium">Location</span>
              </div>
              <p>{threatData.location}</p>
            </div>
            
            <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="flex items-center space-x-2 mb-2">
                <Mail className="h-5 w-5 text-blue-500" />
                <span className="font-medium">Associated Emails</span>
              </div>
              <ul className="space-y-1">
                {threatData.associatedEmails.map(email => (
                  <li key={email}>{email}</li>
                ))}
              </ul>
            </div>
          </div>

          <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
            <div className="flex items-center space-x-2 mb-2">
              <Link2 className="h-5 w-5 text-blue-500" />
              <span className="font-medium">Connected Infrastructure</span>
            </div>
            <ul className="space-y-1">
              {threatData.domains.map(domain => (
                <li key={domain}>{domain}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}