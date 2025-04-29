"use client";

import React, { useState, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { Agent } from '../../types/agent';
import { agentService } from '../../services/agent-service';
import { Button } from '../ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Switch } from '../ui/switch';
import { Textarea } from '../ui/textarea';
import { useToast } from '../ui/use-toast';
import { useQueryClient } from '@tanstack/react-query';

// Create the schema
const agentConfigSchema = z.object({
  name: z.string().min(1, 'Name is required'),
  hostname: z.string().min(1, 'Hostname is required'),
  ipAddress: z.string().ip('Invalid IP address'),
  port: z.number().min(1).max(65535),
  isEnabled: z.boolean().default(true),
  collectEventLogs: z.boolean().default(false),
  collectSystemMetrics: z.boolean().default(false),
  eventLogsToMonitor: z.string().optional(),
  configuration: z.record(z.string()).optional(),
});

type AgentConfigFormData = z.infer<typeof agentConfigSchema>;

interface AgentConfigurationProps {
  agent: Agent;
  onUpdate: (agent: Agent) => void;
}

export function AgentConfiguration({ agent, onUpdate }: AgentConfigurationProps) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [config, setConfig] = useState({
    hostname: agent.hostname || '',
    ipAddress: agent.ipAddress || '',
    port: agent.port || 0,
    isEnabled: agent.isEnabled || false,
    collectEventLogs: agent.collectEventLogs || false,
    collectSystemMetrics: agent.collectSystemMetrics || false,
    eventLogsToMonitor: agent.eventLogsToMonitor || '',
    configuration: agent.configuration || {}
  });
  const [isSaving, setIsSaving] = useState(false);

  const {
    register,
    handleSubmit,
    watch,
    formState: { errors, isSubmitting },
  } = useForm<AgentConfigFormData>({
    resolver: zodResolver(agentConfigSchema),
    defaultValues: {
      name: agent.name,
      hostname: agent.hostname,
      ipAddress: agent.ipAddress,
      port: agent.port,
      isEnabled: agent.isEnabled,
      collectEventLogs: agent.collectEventLogs ?? false,
      collectSystemMetrics: agent.collectSystemMetrics ?? false,
      eventLogsToMonitor: agent.eventLogsToMonitor,
      configuration: agent.configuration,
    },
  });

  const isEnabled = watch('isEnabled');
  const collectSystemMetrics = watch('collectSystemMetrics');
  const collectEventLogs = watch('collectEventLogs');

  const onSubmit = async (data: AgentConfigFormData) => {
    try {
      setIsSaving(true);
      await agentService.configureAgent(agent.agentId as string, data);
      await queryClient.invalidateQueries({ queryKey: ['agents'] });
      toast({
        title: 'Success',
        description: 'Agent configuration updated successfully',
      });
      onUpdate({ ...agent, ...data });
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to update agent configuration',
        variant: 'destructive',
      });
    } finally {
      setIsSaving(false);
    }
  };

  const handleSave = async () => {
    try {
      setIsSaving(true);
      if (!agent.id) {
        throw new Error('Agent ID is required');
      }
      await agentService.configureAgent(agent.id as string, config);
      await queryClient.invalidateQueries({ queryKey: ['agents'] });
      toast({
        title: 'Configuration saved',
        description: 'Agent configuration has been updated successfully.',
      });
      onUpdate({ ...agent, ...config });
    } catch (error) {
      toast({
        title: 'Error',
        description: error instanceof Error ? error.message : 'Failed to save configuration',
        variant: 'destructive',
      });
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Agent Configuration</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="hostname">Hostname</Label>
              <Input
                id="hostname"
                value={config.hostname}
                onChange={(e) => setConfig({ ...config, hostname: e.target.value })}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="ipAddress">IP Address</Label>
              <Input
                id="ipAddress"
                value={config.ipAddress}
                onChange={(e) => setConfig({ ...config, ipAddress: e.target.value })}
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="port">Port</Label>
            <Input
              id="port"
              type="number"
              value={config.port}
              onChange={(e) => setConfig({ ...config, port: parseInt(e.target.value) || 0 })}
            />
          </div>

          <div className="flex items-center space-x-2">
            <Switch
              id="isEnabled"
              checked={config.isEnabled}
              onCheckedChange={(checked) => setConfig({ ...config, isEnabled: checked })}
            />
            <Label htmlFor="isEnabled">Enable Agent</Label>
          </div>

          <div className="flex items-center space-x-2">
            <Switch
              id="collectEventLogs"
              checked={config.collectEventLogs}
              onCheckedChange={(checked) => setConfig({ ...config, collectEventLogs: checked })}
            />
            <Label htmlFor="collectEventLogs">Collect Event Logs</Label>
          </div>

          <div className="flex items-center space-x-2">
            <Switch
              id="collectSystemMetrics"
              checked={config.collectSystemMetrics}
              onCheckedChange={(checked) => setConfig({ ...config, collectSystemMetrics: checked })}
            />
            <Label htmlFor="collectSystemMetrics">Collect System Metrics</Label>
          </div>

          <div className="space-y-2">
            <Label htmlFor="eventLogsToMonitor">Event Logs to Monitor</Label>
            <Input
              id="eventLogsToMonitor"
              value={config.eventLogsToMonitor}
              onChange={(e) => setConfig({ ...config, eventLogsToMonitor: e.target.value })}
              placeholder="Comma-separated list of event logs"
            />
          </div>

          <Button
            onClick={handleSave}
            disabled={isSaving}
            className="w-full"
          >
            {isSaving ? 'Saving...' : 'Save Configuration'}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}