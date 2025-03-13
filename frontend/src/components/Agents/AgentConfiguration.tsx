import React, { useState } from 'react';
import { Settings, Save, RefreshCw } from 'lucide-react';
import { Agent } from '../../types/agent';

interface AgentConfig {
  collectionInterval: number;
  logLevel: 'debug' | 'info' | 'warn' | 'error';
  enableFIM: boolean;
  fimPaths: string[];
  enableNetworkMonitoring: boolean;
  enableProcessMonitoring: boolean;
  retentionDays: number;
}

const defaultConfig: AgentConfig = {
  collectionInterval: 60,
  logLevel: 'info',
  enableFIM: true,
  fimPaths: ['/etc', '/var/www', '/usr/local/bin'],
  enableNetworkMonitoring: true,
  enableProcessMonitoring: true,
  retentionDays: 30,
};

export function AgentConfiguration({ agent }: { agent: Agent }) {
  const [config, setConfig] = useState<AgentConfig>(defaultConfig);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Settings className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
            Agent Configuration: {agent.name}
          </h2>
        </div>
        <div className="flex space-x-2">
          <button className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 flex items-center">
            <Save className="h-4 w-4 mr-2" />
            Save Changes
          </button>
          <button className="px-4 py-2 border rounded-lg hover:bg-gray-50 flex items-center">
            <RefreshCw className="h-4 w-4 mr-2" />
            Reset
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Collection Interval (seconds)
            </label>
            <input
              type="number"
              value={config.collectionInterval}
              onChange={(e) => setConfig({ ...config, collectionInterval: parseInt(e.target.value) })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Log Level
            </label>
            <select
              value={config.logLevel}
              onChange={(e) => setConfig({ ...config, logLevel: e.target.value as any })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            >
              <option value="debug">Debug</option>
              <option value="info">Info</option>
              <option value="warn">Warning</option>
              <option value="error">Error</option>
            </select>
          </div>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Monitoring Features
            </label>
            <div className="mt-2 space-y-2">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={config.enableFIM}
                  onChange={(e) => setConfig({ ...config, enableFIM: e.target.checked })}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">
                  File Integrity Monitoring
                </span>
              </label>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={config.enableNetworkMonitoring}
                  onChange={(e) => setConfig({ ...config, enableNetworkMonitoring: e.target.checked })}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">
                  Network Monitoring
                </span>
              </label>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={config.enableProcessMonitoring}
                  onChange={(e) => setConfig({ ...config, enableProcessMonitoring: e.target.checked })}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">
                  Process Monitoring
                </span>
              </label>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Data Retention (days)
            </label>
            <input
              type="number"
              value={config.retentionDays}
              onChange={(e) => setConfig({ ...config, retentionDays: parseInt(e.target.value) })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            />
          </div>
        </div>
      </div>

      {config.enableFIM && (
        <div className="mt-6">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
            FIM Monitored Paths
          </label>
          <div className="mt-2">
            <textarea
              value={config.fimPaths.join('\n')}
              onChange={(e) => setConfig({ ...config, fimPaths: e.target.value.split('\n') })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              rows={4}
            />
          </div>
        </div>
      )}
    </div>
  );
}