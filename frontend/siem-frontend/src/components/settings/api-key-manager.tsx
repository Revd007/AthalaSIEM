import React, { useState, useEffect } from 'react';
import { apiKeyService } from '../../services/api-key-service';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { DataTable } from '../ui/data-table';
import { Dialog } from '../ui/dialog';

export function APIKeyManager() {
  const [apiKeys, setApiKeys] = useState<APIKey[]>([]);
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [newKeyName, setNewKeyName] = useState('');

  useEffect(() => {
    loadAPIKeys();
  }, []);

  const loadAPIKeys = async () => {
    const keys = await apiKeyService.getAPIKeys();
    setApiKeys(keys);
  };

  const handleCreateKey = async () => {
    await apiKeyService.createAPIKey(newKeyName);
    setIsCreateDialogOpen(false);
    setNewKeyName('');
    loadAPIKeys();
  };

  const handleRevokeKey = async (id: string) => {
    await apiKeyService.revokeAPIKey(id);
    loadAPIKeys();
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">API Keys</h2>
        <Button onClick={() => setIsCreateDialogOpen(true)}>
          Create New API Key
        </Button>
      </div>

      <DataTable
        data={apiKeys}
        columns={[
          { key: 'name', title: 'Name' },
          { key: 'created_at', title: 'Created' },
          { key: 'expires_at', title: 'Expires' },
          { key: 'last_used_at', title: 'Last Used' },
          {
            key: 'actions',
            title: 'Actions',
            render: (apiKey) => (
              <Button
                variant="destructive"
                onClick={() => handleRevokeKey(apiKey.id)}
              >
                Revoke
              </Button>
            ),
          },
        ]}
      />

      <Dialog
        open={isCreateDialogOpen}
        onClose={() => setIsCreateDialogOpen(false)}
      >
        <div className="space-y-4">
          <h3 className="text-lg font-medium">Create New API Key</h3>
          <Input
            value={newKeyName}
            onChange={(e) => setNewKeyName(e.target.value)}
            placeholder="API Key Name"
          />
          <div className="flex justify-end space-x-2">
            <Button
              variant="outline"
              onClick={() => setIsCreateDialogOpen(false)}
            >
              Cancel
            </Button>
            <Button onClick={handleCreateKey}>Create</Button>
          </div>
        </div>
      </Dialog>
    </div>
  );
}