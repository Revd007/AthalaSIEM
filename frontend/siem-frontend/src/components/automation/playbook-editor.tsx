import React, { useState } from 'react';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Textarea } from '../ui/textarea';
import { Switch } from '../ui/switch';
import { Card } from '../ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { Trash2 } from 'lucide-react';

export interface PlaybookTemplate {
  id?: string;
  name: string;
  description: string;
  steps: PlaybookStep[];
  triggers: PlaybookTrigger[];
  is_active: boolean;
  created_at?: string;
  updated_at?: string;
}

interface PlaybookStep {
  id: string;
  type: 'action' | 'condition' | 'notification' | 'email' | 'webhook' | 'script';
  config: Record<string, any>;
  next_step?: string;
}

interface PlaybookTrigger {
  type: 'alert' | 'schedule' | 'event';
  conditions: Record<string, any>;
}

interface PlaybookEditorProps {
  template: PlaybookTemplate | null;
  onSave: (template: Partial<PlaybookTemplate>) => Promise<void>;
  onCancel: () => void;
}

export function PlaybookEditor({ template, onSave, onCancel }: PlaybookEditorProps) {
  const [formData, setFormData] = useState<PlaybookTemplate>(
    template || {
      name: '',
      description: '',
      steps: [],
      triggers: [],
      is_active: true
    }
  );

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await onSave(formData);
  };

  const handleAddStep = () => {
    setFormData({
      ...formData,
      steps: [...formData.steps, {
        id: `step-${formData.steps.length + 1}`,
        type: 'action',
        config: {}
      }]
    });
  };

  const handleAddTrigger = () => {
    setFormData({
      ...formData,
      triggers: [...formData.triggers, {
        type: 'alert',
        conditions: {}
      }]
    });
  };

  return (
    <Card className="p-6">
      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label className="block text-sm font-medium text-gray-700">
            Playbook Name
          </label>
          <Input
            value={formData.name}
            onChange={(e) => setFormData({ ...formData, name: e.target.value })}
            placeholder="Enter playbook name"
            required
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">
            Description
          </label>
          <Textarea
            value={formData.description}
            onChange={(e) => setFormData({ ...formData, description: e.target.value })}
            placeholder="Enter playbook description"
            rows={4}
          />
        </div>

        <div className="flex items-center space-x-2">
          <Switch
            checked={formData.is_active}
            onCheckedChange={(checked) => setFormData({ ...formData, is_active: checked })}
          />
          <label className="text-sm font-medium text-gray-700">
            Active
          </label>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">
            Steps
          </label>
          <div className="space-y-4">
            {formData.steps.map((step, index) => (
              <div key={step.id} className="flex items-center space-x-4">
                <Select
                  value={step.type}
                  onValueChange={(value: PlaybookStep['type']) => {
                    const newSteps = [...formData.steps];
                    newSteps[index] = { ...step, type: value };
                    setFormData({ ...formData, steps: newSteps });
                  }}
                >
                  <SelectTrigger className="w-[200px]">
                    <SelectValue placeholder="Select step type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="action">Action</SelectItem>
                    <SelectItem value="condition">Condition</SelectItem>
                    <SelectItem value="notification">Notification</SelectItem>
                    <SelectItem value="email">Email</SelectItem>
                    <SelectItem value="webhook">Webhook</SelectItem>
                    <SelectItem value="script">Script</SelectItem>
                  </SelectContent>
                </Select>

                <Input 
                  value={step.config?.target || ''}
                  onChange={(e) => {
                    const newSteps = [...formData.steps];
                    newSteps[index] = { 
                      ...step, 
                      config: { ...step.config, target: e.target.value }
                    };
                    setFormData({ ...formData, steps: newSteps });
                  }}
                  placeholder="Configuration"
                  className="flex-1"
                />

                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setFormData({
                      ...formData,
                      steps: formData.steps.filter((_, i) => i !== index)
                    });
                  }}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            ))}

            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handleAddStep}
            >
              Add Step
            </Button>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">
            Triggers
          </label>
          <div className="space-y-4">
            {formData.triggers.map((trigger, index) => (
              <div key={index} className="flex items-center space-x-4">
                <Select
                  value={trigger.type}
                  onValueChange={(value: PlaybookTrigger['type']) => {
                    const newTriggers = [...formData.triggers];
                    newTriggers[index] = { ...trigger, type: value };
                    setFormData({ ...formData, triggers: newTriggers });
                  }}
                >
                  <SelectTrigger className="w-[200px]">
                    <SelectValue placeholder="Select trigger type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="alert">Alert Based</SelectItem>
                    <SelectItem value="schedule">Schedule Based</SelectItem>
                    <SelectItem value="event">Event Based</SelectItem>
                  </SelectContent>
                </Select>

                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setFormData({
                      ...formData,
                      triggers: formData.triggers.filter((_, i) => i !== index)
                    });
                  }}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            ))}

            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handleAddTrigger}
            >
              Add Trigger
            </Button>
          </div>
        </div>

        <div className="flex justify-end space-x-2">
          <Button type="button" variant="outline" onClick={onCancel}>
            Cancel
          </Button>
          <Button type="submit">
            {template ? 'Update' : 'Create'} Playbook
          </Button>
        </div>
      </form>
    </Card>
  );
}