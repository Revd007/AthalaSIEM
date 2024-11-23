import React, { useState } from 'react';
import { MainLayout } from '../components/layout/main-layout';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';
import { Switch } from '../components/ui/switch';
import { Label } from '../components/ui/label';
import { Save } from 'lucide-react';

export default function SettingsPage() {
  const [settings, setSettings] = useState({
    emailNotifications: true,
    alertThreshold: 'medium',
    retentionPeriod: '30',
    apiKey: '••••••••••••••••',
    theme: 'light'
  });

  const handleSave = () => {
    // TODO: Implement settings save functionality
    console.log('Saving settings:', settings);
  };

  return (
    <MainLayout>
      <div className="space-y-6 p-6">
        <h1 className="text-3xl font-bold">Settings</h1>

        <Card>
          <CardHeader>
            <CardTitle>Notifications</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <Label htmlFor="email-notifications">Email Notifications</Label>
              <Switch
                id="email-notifications"
                checked={settings.emailNotifications}
                onCheckedChange={(checked) => 
                  setSettings(prev => ({...prev, emailNotifications: checked}))
                }
              />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Alert Settings</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Minimum Alert Threshold</Label>
              <Select
                value={settings.alertThreshold}
                onValueChange={(value) => 
                  setSettings(prev => ({...prev, alertThreshold: value}))
                }
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="low">Low</SelectItem>
                  <SelectItem value="medium">Medium</SelectItem>
                  <SelectItem value="high">High</SelectItem>
                  <SelectItem value="critical">Critical</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Log Retention Period (days)</Label>
              <Input
                type="number"
                value={settings.retentionPeriod}
                onChange={(e) => 
                  setSettings(prev => ({...prev, retentionPeriod: e.target.value}))
                }
              />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>API Settings</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>API Key</Label>
              <Input
                type="password"
                value={settings.apiKey}
                onChange={(e) => 
                  setSettings(prev => ({...prev, apiKey: e.target.value}))
                }
              />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Appearance</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Theme</Label>
              <Select
                value={settings.theme}
                onValueChange={(value) => 
                  setSettings(prev => ({...prev, theme: value}))
                }
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="light">Light</SelectItem>
                  <SelectItem value="dark">Dark</SelectItem>
                  <SelectItem value="system">System</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>

        <div className="flex justify-end">
          <Button onClick={handleSave} className="w-32">
            <Save className="mr-2 h-4 w-4" />
            Save
          </Button>
        </div>
      </div>
    </MainLayout>
  );
}
