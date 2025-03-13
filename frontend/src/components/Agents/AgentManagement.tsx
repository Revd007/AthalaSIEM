import React, { useState } from 'react';
import { Monitor, Server, Cloud, Activity, AlertTriangle } from 'lucide-react';
import { Agent, AgentType } from '../../types/agent';

const mockAgents: Agent[] = [
  {
    id: '1',
    name: 'WIN-DC01',
    type: 'windows',
    status: 'active',
    version: '2.1.0',
    lastHeartbeat: '2024-03-15T10:30:00Z',
    ipAddress: '192.168.1.10',
    osInfo: {
      platform: 'Windows',
      version: 'Server 2019',
      architecture: 'x64',
    },
    metrics: {
      cpuUsage: 45,
      memoryUsage: 60,
      diskUsage: 75,
    },
  },
  {
    id: '2',
    name: 'aws-collector',
    type: 'cloud',
    status: 'active',
    version: '2.0.0',
    lastHeartbeat: '2024-03-15T10:29:00Z',
    ipAddress: '10.0.1.100',
    cloudInfo: {
      provider: 'aws',
      region: 'us-east-1',
      instanceId: 'i-0123456789',
    },
    metrics: {
      cpuUsage: 30,
      memoryUsage: 45,
      diskUsage: 50,
    },
  },
];

const agentIcons: Record<AgentType, React.ElementType> = {
  windows: Monitor,
  linux: Server,
  cloud: Cloud,
};

export function AgentManagement() {
  const [agents] = useState<Agent[]>(mockAgents);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Server className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Agent Management</h2>
        </div>
        <button className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600">
          Deploy New Agent
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {agents.map(agent => {
          const Icon = agentIcons[agent.type];
          return (
            <div
              key={agent.id}
              className="border dark:border-gray-700 rounded-lg p-4 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700"
              onClick={() => setSelectedAgent(agent)}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-center space-x-3">
                  <Icon className="h-5 w-5 text-blue-500" />
                  <div>
                    <h3 className="font-medium text-gray-900 dark:text-white">{agent.name}</h3>
                    <p className="text-sm text-gray-500">{agent.ipAddress}</p>
                  </div>
                </div>
                <span className={`px-2 py-1 text-xs rounded-full ${
                  agent.status === 'active' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                }`}>
                  {agent.status}
                </span>
              </div>

              <div className="mt-4 grid grid-cols-3 gap-4">
                <div className="text-center">
                  <div className="text-sm font-medium text-gray-900 dark:text-white">CPU</div>
                  <div className="text-sm text-gray-500">{agent.metrics.cpuUsage}%</div>
                </div>
                <div className="text-center">
                  <div className="text-sm font-medium text-gray-900 dark:text-white">Memory</div>
                  <div className="text-sm text-gray-500">{agent.metrics.memoryUsage}%</div>
                </div>
                <div className="text-center">
                  <div className="text-sm font-medium text-gray-900 dark:text-white">Disk</div>
                  <div className="text-sm text-gray-500">{agent.metrics.diskUsage}%</div>
                </div>
              </div>

              {agent.metrics.cpuUsage > 80 && (
                <div className="mt-4 flex items-center space-x-2 text-yellow-600">
                  <AlertTriangle className="h-4 w-4" />
                  <span className="text-sm">High CPU Usage</span>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}