import React, { useState, useEffect } from 'react';
import { playbookService } from '../../services/playbook-service';
import { Button } from '../ui/button';
import { DataTable } from '../ui/data-table';
import {PlaybookEditor,  PlaybookTemplate } from './playbook-editor';

export function PlaybookManager() {
  const [templates, setTemplates] = useState<PlaybookTemplate[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState<PlaybookTemplate | null>(null);
  const [isEditing, setIsEditing] = useState(false);

  useEffect(() => {
    loadTemplates();
  }, []);

  const loadTemplates = async () => {
    const data = await playbookService.getTemplates();
    setTemplates(data);
  };

  const handleCreateTemplate = () => {
    setSelectedTemplate(null);
    setIsEditing(true);
  };

  const handleEditTemplate = (template: PlaybookTemplate) => {
    setSelectedTemplate(template);
    setIsEditing(true);
  };

  const handleSaveTemplate = async (template: Partial<PlaybookTemplate>) => {
    await playbookService.createTemplate(template);
    setIsEditing(false);
    loadTemplates();
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Playbook Management</h2>
        <Button onClick={handleCreateTemplate}>Create New Playbook</Button>
      </div>

      {isEditing ? (
        <PlaybookEditor
          template={selectedTemplate}
          onSave={handleSaveTemplate}
          onCancel={() => setIsEditing(false)}
        />
      ) : (
        <DataTable
          data={templates}
          columns={[
            { key: 'name', title: 'Name' },
            { key: 'description', title: 'Description' },
            { key: 'created_at', title: 'Created At' },
            { key: 'is_active', title: 'Status' },
            {
              key: 'actions',
              title: 'Actions',
              render: (template) => (
                <Button onClick={() => handleEditTemplate(template)}>
                  Edit
                </Button>
              ),
            },
          ]}
        />
      )}
    </div>
  );
}