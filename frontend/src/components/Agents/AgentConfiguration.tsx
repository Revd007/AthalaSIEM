"use client";

import React from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { Agent } from '../../types/agent';
import { agentService } from '../../services/agent-service';
import { Button } from '../ui/button';
import { Card } from '../ui/card';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Switch } from '../ui/switch';
import { Textarea } from '../ui/textarea';
import { useToast } from '../ui/use-toast';
import { useQueryClient } from '@tanstack/react-query';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";

// Create the schema first
const agentConfigSchema = z.object({
  name: z.string().min(1, 'Name is required'),
  hostname: z.string().min(1, 'Hostname is required'),
  ipAddress: z.string().min(1, 'IP address is required'),
  port: z.number().min(1).max(65535).optional(),
  isEnabled: z.boolean(),
  collectEventLogs: z.boolean(),
  collectSystemMetrics: z.boolean(),
  eventLogsToMonitor: z.string().optional(),
  configuration: z.record(z.string()).optional(),
});

// Infer the type from the schema
type AgentConfigForm = z.infer<typeof agentConfigSchema>;

interface AgentConfigurationProps {
  agent: Agent;
  onClose: () => void;
}

export function AgentConfiguration({ agent, onClose }: AgentConfigurationProps) {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<AgentConfigForm>({
    resolver: zodResolver(agentConfigSchema),
    defaultValues: {
      name: agent.name,
      hostname: agent.hostname,
      ipAddress: agent.ipAddress,
      port: agent.port ?? 514,
      isEnabled: agent.isEnabled ?? agent.enabled ?? true,
      collectEventLogs: agent.collectEventLogs ?? false,
      collectSystemMetrics: agent.collectSystemMetrics ?? false,
      eventLogsToMonitor: Array.isArray(agent.eventLogsToMonitor) 
        ? agent.eventLogsToMonitor.join(', ') 
        : (typeof agent.eventLogsToMonitor === 'string' ? agent.eventLogsToMonitor : ''),
      configuration: agent.configuration ?? {},
    },
  });

  const onSubmit = async (data: AgentConfigForm) => {
    try {
      const agentId = agent.id || agent.agentId;
      if (!agentId) {
        throw new Error('Agent ID is required');
      }
      await agentService.configureAgent(agentId, data);
      await queryClient.invalidateQueries({ queryKey: ['agents'] });
      toast({
        title: 'Success',
        description: 'Agent configuration updated successfully',
      });
      onClose();
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to update agent configuration',
        variant: 'destructive',
      });
    }
  };

  return (
    <Card className="p-6">
      <h3 className="text-lg font-semibold mb-4">Agent Configuration</h3>
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="name">Name</Label>
            <Input
              id="name"
              {...register('name')}
              className={errors.name ? 'border-red-500' : ''}
              aria-describedby="name-description"
            />
            <p id="name-description" className="text-sm text-muted-foreground">
              A unique name for this agent
            </p>
            {errors.name && (
              <p className="text-sm text-red-500">{errors.name.message}</p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="hostname">Hostname</Label>
            <Input
              id="hostname"
              {...register('hostname')}
              className={errors.hostname ? 'border-red-500' : ''}
              aria-describedby="hostname-description"
            />
            <p id="hostname-description" className="text-sm text-muted-foreground">
              The hostname where this agent is installed
            </p>
            {errors.hostname && (
              <p className="text-sm text-red-500">{errors.hostname.message}</p>
            )}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="ipAddress">IP Address</Label>
            <Input
              id="ipAddress"
              {...register('ipAddress')}
              className={errors.ipAddress ? 'border-red-500' : ''}
              aria-describedby="ip-description"
            />
            <p id="ip-description" className="text-sm text-muted-foreground">
              The IP address of the agent
            </p>
            {errors.ipAddress && (
              <p className="text-sm text-red-500">{errors.ipAddress.message}</p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="port">Port</Label>
            <Input
              id="port"
              type="number"
              {...register('port', { valueAsNumber: true })}
              className={errors.port ? 'border-red-500' : ''}
              aria-describedby="port-description"
            />
            <p id="port-description" className="text-sm text-muted-foreground">
              Port number (1-65535)
            </p>
            {errors.port && (
              <p className="text-sm text-red-500">{errors.port.message}</p>
            )}
          </div>
        </div>

        <div className="space-y-2">
          <Label htmlFor="eventLogsToMonitor">Event Logs to Monitor</Label>
          <Textarea
            id="eventLogsToMonitor"
            {...register('eventLogsToMonitor')}
            placeholder="Enter event log names separated by commas"
            aria-describedby="logs-description"
          />
          <p id="logs-description" className="text-sm text-muted-foreground">
            Specify which Windows Event Logs to monitor
          </p>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="flex items-center justify-between rounded-lg border p-4">
            <div className="space-y-0.5">
              <Label htmlFor="isEnabled">Enabled</Label>
              <p className="text-sm text-muted-foreground">
                Enable or disable this agent
              </p>
            </div>
            <Switch
              id="isEnabled"
              {...register('isEnabled')}
            />
          </div>

          <div className="flex items-center justify-between rounded-lg border p-4">
            <div className="space-y-0.5">
              <Label htmlFor="collectSystemMetrics">System Metrics</Label>
              <p className="text-sm text-muted-foreground">
                Collect system performance metrics
              </p>
            </div>
            <Switch
              id="collectSystemMetrics"
              {...register('collectSystemMetrics')}
            />
          </div>
        </div>

        <div className="flex items-center justify-between rounded-lg border p-4">
          <div className="space-y-0.5">
            <Label htmlFor="collectEventLogs">Event Logs</Label>
            <p className="text-sm text-muted-foreground">
              Enable Windows Event Log collection
            </p>
          </div>
          <Switch
            id="collectEventLogs"
            {...register('collectEventLogs')}
          />
        </div>

        <div className="flex justify-end space-x-2">
          <Button 
            type="button" 
            variant="outline" 
            onClick={onClose}
            className="w-24"
          >
            Cancel
          </Button>
          <Button 
            type="submit" 
            disabled={isSubmitting}
            className="w-24"
          >
            {isSubmitting ? 'Saving...' : 'Save'}
          </Button>
        </div>
      </form>
    </Card>
  );
}