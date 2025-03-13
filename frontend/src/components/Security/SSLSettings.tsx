import React, { useState } from 'react';
import { Shield, Lock, RefreshCw } from 'lucide-react';

export function SSLSettings() {
  const [sslConfig, setSSLConfig] = useState({
    enabled: true,
    protocol: 'TLS 1.3',
    cipherSuites: ['ECDHE-ECDSA-AES256-GCM-SHA384', 'ECDHE-RSA-AES256-GCM-SHA384'],
    certificateExpiry: '2025-03-15',
    hsts: true,
    ocspStapling: true,
  });

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Lock className="h-6 w-6 text-green-500" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">SSL/TLS Configuration</h2>
        </div>
        <button className="flex items-center px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600">
          <RefreshCw className="h-4 w-4 mr-2" />
          Update Certificate
        </button>
      </div>

      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-gray-700 dark:text-gray-300">SSL Status</span>
              <span className="flex items-center text-green-500">
                <Shield className="h-4 w-4 mr-1" />
                Active
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-700 dark:text-gray-300">Protocol Version</span>
              <span className="text-gray-900 dark:text-gray-100">{sslConfig.protocol}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-700 dark:text-gray-300">Certificate Expiry</span>
              <span className="text-gray-900 dark:text-gray-100">{sslConfig.certificateExpiry}</span>
            </div>
          </div>

          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-gray-700 dark:text-gray-300">HSTS</span>
              <span className={`px-2 py-1 rounded-full text-sm ${sslConfig.hsts ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                {sslConfig.hsts ? 'Enabled' : 'Disabled'}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-700 dark:text-gray-300">OCSP Stapling</span>
              <span className={`px-2 py-1 rounded-full text-sm ${sslConfig.ocspStapling ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                {sslConfig.ocspStapling ? 'Enabled' : 'Disabled'}
              </span>
            </div>
          </div>
        </div>

        <div className="mt-6">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Cipher Suites</h3>
          <div className="space-y-2">
            {sslConfig.cipherSuites.map((cipher, index) => (
              <div key={index} className="flex items-center space-x-2 bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                <Lock className="h-4 w-4 text-green-500" />
                <span className="text-sm text-gray-700 dark:text-gray-300">{cipher}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}