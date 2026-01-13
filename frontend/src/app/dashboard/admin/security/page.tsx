'use client'

import { useState, useEffect } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { 
  Shield, Key, Clock, Save
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { toast } from 'sonner'
import { api } from '@/lib/api'

interface PasswordPolicy {
  minLength: number
  requireUppercase: boolean
  requireLowercase: boolean
  requireDigit: boolean
  requireSpecialChar: boolean
  maxAge: number // days
  preventReuse: number // number of previous passwords
  lockoutThreshold: number // failed attempts
  lockoutDuration: number // minutes
}

interface TwoFactorSettings {
  required: boolean
  allowedMethods: string[]
  gracePeriodDays: number
  rememberDeviceDays: number
}

interface SessionSettings {
  sessionTimeout: number // minutes
  maxConcurrentSessions: number
  requireReauthForSensitive: boolean
}

const defaultPasswordPolicy: PasswordPolicy = {
  minLength: 8,
  requireUppercase: true,
  requireLowercase: true,
  requireDigit: true,
  requireSpecialChar: true,
  maxAge: 90,
  preventReuse: 5,
  lockoutThreshold: 5,
  lockoutDuration: 30,
}

const defaultTwoFactorSettings: TwoFactorSettings = {
  required: false,
  allowedMethods: ['authenticator', 'email'],
  gracePeriodDays: 7,
  rememberDeviceDays: 30,
}

const defaultSessionSettings: SessionSettings = {
  sessionTimeout: 60,
  maxConcurrentSessions: 3,
  requireReauthForSensitive: true,
}

export default function SecuritySettingsPage() {
  const queryClient = useQueryClient()
  const [passwordPolicy, setPasswordPolicy] = useState<PasswordPolicy>(defaultPasswordPolicy)
  const [twoFactorSettings, setTwoFactorSettings] = useState<TwoFactorSettings>(defaultTwoFactorSettings)
  const [sessionSettings, setSessionSettings] = useState<SessionSettings>(defaultSessionSettings)

  // Fetch security settings
  const { data: securitySettings, isLoading } = useQuery({
    queryKey: ['security-settings'],
    queryFn: async () => {
      try {
        const response = await api.get('/api/settings/security')
        return response.data
      } catch {
        // Return defaults if no settings exist
        return {
          passwordPolicy: defaultPasswordPolicy,
          twoFactorSettings: defaultTwoFactorSettings,
          sessionSettings: defaultSessionSettings,
        }
      }
    },
  })

  useEffect(() => {
    if (securitySettings) {
      setPasswordPolicy(securitySettings.passwordPolicy || defaultPasswordPolicy)
      setTwoFactorSettings(securitySettings.twoFactorSettings || defaultTwoFactorSettings)
      setSessionSettings(securitySettings.sessionSettings || defaultSessionSettings)
    }
  }, [securitySettings])

  // Save password policy mutation
  const savePasswordPolicyMutation = useMutation({
    mutationFn: async (policy: PasswordPolicy) => {
      const response = await api.put('/api/settings/security/password-policy', policy)
      return response.data
    },
    onSuccess: () => {
      toast.success('Password policy saved successfully')
      queryClient.invalidateQueries({ queryKey: ['security-settings'] })
    },
    onError: (error: Error & { response?: { data?: { message?: string } } }) => {
      toast.error(error.response?.data?.message || 'Failed to save password policy')
    },
  })

  // Save 2FA settings mutation
  const saveTwoFactorMutation = useMutation({
    mutationFn: async (settings: TwoFactorSettings) => {
      const response = await api.put('/api/settings/security/two-factor', settings)
      return response.data
    },
    onSuccess: () => {
      toast.success('Two-factor settings saved successfully')
      queryClient.invalidateQueries({ queryKey: ['security-settings'] })
    },
    onError: (error: Error & { response?: { data?: { message?: string } } }) => {
      toast.error(error.response?.data?.message || 'Failed to save 2FA settings')
    },
  })

  // Save session settings mutation
  const saveSessionMutation = useMutation({
    mutationFn: async (settings: SessionSettings) => {
      const response = await api.put('/api/settings/security/session', settings)
      return response.data
    },
    onSuccess: () => {
      toast.success('Session settings saved successfully')
      queryClient.invalidateQueries({ queryKey: ['security-settings'] })
    },
    onError: (error: Error & { response?: { data?: { message?: string } } }) => {
      toast.error(error.response?.data?.message || 'Failed to save session settings')
    },
  })

  const getPasswordStrength = () => {
    let score = 0
    if (passwordPolicy.minLength >= 8) score++
    if (passwordPolicy.minLength >= 12) score++
    if (passwordPolicy.requireUppercase) score++
    if (passwordPolicy.requireLowercase) score++
    if (passwordPolicy.requireDigit) score++
    if (passwordPolicy.requireSpecialChar) score++
    if (passwordPolicy.preventReuse >= 5) score++

    if (score >= 6) return { label: 'Strong', color: 'text-green-600' }
    if (score >= 4) return { label: 'Medium', color: 'text-yellow-600' }
    return { label: 'Weak', color: 'text-red-600' }
  }

  const strength = getPasswordStrength()

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  return (
    <div className="p-4 sm:p-6 lg:p-8 space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
            <Shield className="h-6 w-6" />
            Security Settings
          </h1>
          <p className="text-sm text-gray-500 mt-1">Configure security policies and authentication settings</p>
        </div>
      </div>

      <Tabs defaultValue="password" className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="password">
            <Key className="h-4 w-4 mr-2" />
            Password Policy
          </TabsTrigger>
          <TabsTrigger value="2fa">
            <Shield className="h-4 w-4 mr-2" />
            Two-Factor Auth
          </TabsTrigger>
          <TabsTrigger value="session">
            <Clock className="h-4 w-4 mr-2" />
            Session Settings
          </TabsTrigger>
        </TabsList>

        {/* Password Policy Tab */}
        <TabsContent value="password" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>Password Requirements</span>
                <Badge className={strength.color}>{strength.label}</Badge>
              </CardTitle>
              <CardDescription>Configure password complexity requirements</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <Label>Minimum Length</Label>
                  <Input
                    type="number"
                    min={6}
                    max={32}
                    value={passwordPolicy.minLength}
                    onChange={(e) => setPasswordPolicy({ ...passwordPolicy, minLength: parseInt(e.target.value) })}
                  />
                  <p className="text-xs text-gray-500 mt-1">Minimum number of characters required</p>
                </div>
                <div>
                  <Label>Password Max Age (days)</Label>
                  <Input
                    type="number"
                    min={0}
                    max={365}
                    value={passwordPolicy.maxAge}
                    onChange={(e) => setPasswordPolicy({ ...passwordPolicy, maxAge: parseInt(e.target.value) })}
                  />
                  <p className="text-xs text-gray-500 mt-1">0 = never expires</p>
                </div>
              </div>

              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Require Uppercase Letter</Label>
                    <p className="text-xs text-gray-500">At least one uppercase (A-Z)</p>
                  </div>
                  <Switch
                    checked={passwordPolicy.requireUppercase}
                    onCheckedChange={(checked) => setPasswordPolicy({ ...passwordPolicy, requireUppercase: checked })}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Require Lowercase Letter</Label>
                    <p className="text-xs text-gray-500">At least one lowercase (a-z)</p>
                  </div>
                  <Switch
                    checked={passwordPolicy.requireLowercase}
                    onCheckedChange={(checked) => setPasswordPolicy({ ...passwordPolicy, requireLowercase: checked })}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Require Digit</Label>
                    <p className="text-xs text-gray-500">At least one number (0-9)</p>
                  </div>
                  <Switch
                    checked={passwordPolicy.requireDigit}
                    onCheckedChange={(checked) => setPasswordPolicy({ ...passwordPolicy, requireDigit: checked })}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Require Special Character</Label>
                    <p className="text-xs text-gray-500">At least one symbol (!@#$%^&*)</p>
                  </div>
                  <Switch
                    checked={passwordPolicy.requireSpecialChar}
                    onCheckedChange={(checked) => setPasswordPolicy({ ...passwordPolicy, requireSpecialChar: checked })}
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4 border-t">
                <div>
                  <Label>Prevent Password Reuse</Label>
                  <Input
                    type="number"
                    min={0}
                    max={24}
                    value={passwordPolicy.preventReuse}
                    onChange={(e) => setPasswordPolicy({ ...passwordPolicy, preventReuse: parseInt(e.target.value) })}
                  />
                  <p className="text-xs text-gray-500 mt-1">Number of previous passwords to remember</p>
                </div>
                <div>
                  <Label>Account Lockout Threshold</Label>
                  <Input
                    type="number"
                    min={1}
                    max={20}
                    value={passwordPolicy.lockoutThreshold}
                    onChange={(e) => setPasswordPolicy({ ...passwordPolicy, lockoutThreshold: parseInt(e.target.value) })}
                  />
                  <p className="text-xs text-gray-500 mt-1">Failed attempts before lockout</p>
                </div>
                <div>
                  <Label>Lockout Duration (minutes)</Label>
                  <Input
                    type="number"
                    min={1}
                    max={1440}
                    value={passwordPolicy.lockoutDuration}
                    onChange={(e) => setPasswordPolicy({ ...passwordPolicy, lockoutDuration: parseInt(e.target.value) })}
                  />
                  <p className="text-xs text-gray-500 mt-1">How long accounts remain locked</p>
                </div>
              </div>

              <div className="flex justify-end">
                <Button 
                  onClick={() => savePasswordPolicyMutation.mutate(passwordPolicy)}
                  disabled={savePasswordPolicyMutation.isPending}
                >
                  <Save className="h-4 w-4 mr-2" />
                  {savePasswordPolicyMutation.isPending ? 'Saving...' : 'Save Password Policy'}
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Two-Factor Auth Tab */}
        <TabsContent value="2fa" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Two-Factor Authentication Settings</CardTitle>
              <CardDescription>Configure organization-wide 2FA requirements</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="flex items-center gap-3">
                  <Shield className="h-6 w-6 text-blue-600" />
                  <div>
                    <p className="font-medium">Require 2FA for All Users</p>
                    <p className="text-sm text-gray-500">Force all users to enable two-factor authentication</p>
                  </div>
                </div>
                <Switch
                  checked={twoFactorSettings.required}
                  onCheckedChange={(checked) => setTwoFactorSettings({ ...twoFactorSettings, required: checked })}
                />
              </div>

              <div className="space-y-4">
                <Label>Allowed Authentication Methods</Label>
                <div className="flex flex-wrap gap-2">
                  {['authenticator', 'email', 'sms', 'hardware'].map((method) => (
                    <Button
                      key={method}
                      type="button"
                      size="sm"
                      variant={twoFactorSettings.allowedMethods.includes(method) ? 'default' : 'outline'}
                      onClick={() => {
                        if (twoFactorSettings.allowedMethods.includes(method)) {
                          setTwoFactorSettings({
                            ...twoFactorSettings,
                            allowedMethods: twoFactorSettings.allowedMethods.filter(m => m !== method)
                          })
                        } else {
                          setTwoFactorSettings({
                            ...twoFactorSettings,
                            allowedMethods: [...twoFactorSettings.allowedMethods, method]
                          })
                        }
                      }}
                    >
                      {method.charAt(0).toUpperCase() + method.slice(1)}
                    </Button>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <Label>Grace Period (days)</Label>
                  <Input
                    type="number"
                    min={0}
                    max={30}
                    value={twoFactorSettings.gracePeriodDays}
                    onChange={(e) => setTwoFactorSettings({ ...twoFactorSettings, gracePeriodDays: parseInt(e.target.value) })}
                  />
                  <p className="text-xs text-gray-500 mt-1">Time to setup 2FA after requirement is enabled</p>
                </div>
                <div>
                  <Label>Remember Device (days)</Label>
                  <Input
                    type="number"
                    min={0}
                    max={90}
                    value={twoFactorSettings.rememberDeviceDays}
                    onChange={(e) => setTwoFactorSettings({ ...twoFactorSettings, rememberDeviceDays: parseInt(e.target.value) })}
                  />
                  <p className="text-xs text-gray-500 mt-1">How long to trust a verified device</p>
                </div>
              </div>

              <div className="flex justify-end">
                <Button 
                  onClick={() => saveTwoFactorMutation.mutate(twoFactorSettings)}
                  disabled={saveTwoFactorMutation.isPending}
                >
                  <Save className="h-4 w-4 mr-2" />
                  {saveTwoFactorMutation.isPending ? 'Saving...' : 'Save 2FA Settings'}
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Session Settings Tab */}
        <TabsContent value="session" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Session Management</CardTitle>
              <CardDescription>Configure session security and timeout settings</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <Label>Session Timeout (minutes)</Label>
                  <Input
                    type="number"
                    min={5}
                    max={1440}
                    value={sessionSettings.sessionTimeout}
                    onChange={(e) => setSessionSettings({ ...sessionSettings, sessionTimeout: parseInt(e.target.value) })}
                  />
                  <p className="text-xs text-gray-500 mt-1">Auto-logout after inactivity</p>
                </div>
                <div>
                  <Label>Max Concurrent Sessions</Label>
                  <Input
                    type="number"
                    min={1}
                    max={10}
                    value={sessionSettings.maxConcurrentSessions}
                    onChange={(e) => setSessionSettings({ ...sessionSettings, maxConcurrentSessions: parseInt(e.target.value) })}
                  />
                  <p className="text-xs text-gray-500 mt-1">Maximum simultaneous logins per user</p>
                </div>
              </div>

              <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div>
                  <p className="font-medium">Require Re-authentication for Sensitive Actions</p>
                  <p className="text-sm text-gray-500">Users must re-enter password for critical operations</p>
                </div>
                <Switch
                  checked={sessionSettings.requireReauthForSensitive}
                  onCheckedChange={(checked) => setSessionSettings({ ...sessionSettings, requireReauthForSensitive: checked })}
                />
              </div>

              <div className="flex justify-end">
                <Button 
                  onClick={() => saveSessionMutation.mutate(sessionSettings)}
                  disabled={saveSessionMutation.isPending}
                >
                  <Save className="h-4 w-4 mr-2" />
                  {saveSessionMutation.isPending ? 'Saving...' : 'Save Session Settings'}
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
