'use client'

import React, { useState, useEffect } from 'react'
import { User, Settings, Shield, Key, Bell, Globe, Palette, Lock, Eye, EyeOff, Camera, Save, AlertCircle, Users, CheckCircle2, Clock, FileText, Plus, ChevronDown, ArrowUpDown, Search, MoreVertical, Edit2, Trash2, Ban, Globe2, Calendar, X } from 'lucide-react'
import { toast } from 'sonner'
import { api } from '@/lib/api'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Checkbox } from '@/components/ui/checkbox'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'

interface UserProfile {
  id: string
  username: string
  email: string
  firstName?: string
  lastName?: string
  fullName?: string
  role: string
  isActive: boolean
  createdAt: string
  lastLoginAt?: string
  twoFactorEnabled: boolean
  avatar?: string
}

interface ProfileFormData {
  username: string
  email: string
  firstName: string
  lastName: string
  fullName: string
}

interface SecurityFormData {
  currentPassword: string
  newPassword: string
  confirmPassword: string
}

interface NotificationSettings {
  emailAlerts: boolean
  pushNotifications: boolean
  securityAlerts: boolean
  reportNotifications: boolean
  maintenanceNotifications: boolean
}

interface PreferenceSettings {
  theme: 'light' | 'dark' | 'system'
  language: string
  timezone: string
  dateFormat: string
  timeFormat: '12h' | '24h'
}

interface UserListItem {
  id: string
  username: string
  email: string
  firstName: string
  lastName: string
  isActive: boolean
  roles: string[]
  twoFactorEnabled?: boolean
}

interface UserHardeningSettings {
  maxConcurrentSessions: number
  sessionTimeoutMinutes: number
  requireReauthForSensitive: boolean
  restrictLoginByIP: boolean
  allowedIPAddresses?: string[]
  restrictLoginByTime: boolean
  allowedTimeWindows?: Array<{ start: string; end: string; daysOfWeek?: string[] }>
  maxFailedLoginAttempts: number
  lockoutDurationMinutes: number
  enablePasswordExpiration: boolean
  passwordExpirationDays: number
  preventPasswordReuse: boolean
  passwordHistoryCount: number
  requireStrongPassword: boolean
  minPasswordLength: number
  requireUppercase: boolean
  requireLowercase: boolean
  requireDigit: boolean
  requireSpecialChar: boolean
  logAllLoginAttempts: boolean
  emailSecurityNotifications: boolean
}

const UserProfile: React.FC = () => {
  const [activeTab, setActiveTab] = useState('profile')
  const [user, setUser] = useState<UserProfile | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isSaving, setIsSaving] = useState(false)
  const [showPassword, setShowPassword] = useState(false)
  const [showNewPassword, setShowNewPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)

  // Form states
  const [profileForm, setProfileForm] = useState<ProfileFormData>({
    username: '',
    email: '',
    firstName: '',
    lastName: '',
    fullName: ''
  })

  const [securityForm, setSecurityForm] = useState<SecurityFormData>({
    currentPassword: '',
    newPassword: '',
    confirmPassword: ''
  })

  const [notifications, setNotifications] = useState<NotificationSettings>({
    emailAlerts: true,
    pushNotifications: true,
    securityAlerts: true,
    reportNotifications: false,
    maintenanceNotifications: true
  })

  const [preferences, setPreferences] = useState<PreferenceSettings>({
    theme: 'system',
    language: 'en',
    timezone: 'UTC',
    dateFormat: 'MM/dd/yyyy',
    timeFormat: '24h'
  })

  // User management states
  const [selectedUsers, setSelectedUsers] = useState<string[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [categoryFilter, setCategoryFilter] = useState('all')
  const [userDialogOpen, setUserDialogOpen] = useState(false)
  const [selectedUserForEdit, setSelectedUserForEdit] = useState<UserListItem | null>(null)
  const queryClient = useQueryClient()

  // User hardening settings
  const [hardeningSettings, setHardeningSettings] = useState<UserHardeningSettings>({
    maxConcurrentSessions: 3,
    sessionTimeoutMinutes: 60,
    requireReauthForSensitive: true,
    restrictLoginByIP: false,
    allowedIPAddresses: [],
    restrictLoginByTime: false,
    allowedTimeWindows: [],
    maxFailedLoginAttempts: 5,
    lockoutDurationMinutes: 30,
    enablePasswordExpiration: false,
    passwordExpirationDays: 90,
    preventPasswordReuse: true,
    passwordHistoryCount: 5,
    requireStrongPassword: true,
    minPasswordLength: 8,
    requireUppercase: true,
    requireLowercase: true,
    requireDigit: true,
    requireSpecialChar: true,
    logAllLoginAttempts: true,
    emailSecurityNotifications: true
  })
  const [newIPAddress, setNewIPAddress] = useState('')
  const [newTimeWindow, setNewTimeWindow] = useState({ start: '09:00', end: '17:00', daysOfWeek: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] })

  // Load user profile on component mount
  useEffect(() => {
    loadUserProfile()
    if (activeTab === 'security') {
      loadHardeningSettings()
    }
  }, [activeTab])

  const loadUserProfile = async () => {
    try {
      setIsLoading(true)
      const response = await api.get('/api/users/me')
      const userData = response.data as any
      setUser(userData)
      
      // Populate form data (handle both camelCase and PascalCase)
      setProfileForm({
        username: userData.username || userData.Username || '',
        email: userData.email || userData.Email || '',
        firstName: userData.firstName || userData.FirstName || '',
        lastName: userData.lastName || userData.LastName || '',
        fullName: (userData.firstName || userData.FirstName || '') + ' ' + (userData.lastName || userData.LastName || '')
      })

      // Load user preferences and settings
      await loadUserSettings()
    } catch (error) {
      console.error('Error loading user profile:', error)
      toast.error('Failed to load user profile')
    } finally {
      setIsLoading(false)
    }
  }

  const loadUserSettings = async () => {
    try {
      // Load notification settings (optional - may not exist)
      try {
        const notificationsResponse = await api.get<NotificationSettings>('/api/users/me/notifications')
        if (notificationsResponse.data) {
          setNotifications(notificationsResponse.data)
        }
      } catch {
        // Notifications endpoint not available, use defaults
      }

      // Load preferences (optional - may not exist)
      try {
        const preferencesResponse = await api.get<PreferenceSettings>('/api/users/me/preferences')
        if (preferencesResponse.data) {
          setPreferences(preferencesResponse.data)
        }
      } catch {
        // Preferences endpoint not available, use defaults
      }
    } catch (error) {
      console.error('Error loading user settings:', error)
    }
  }

  const handleProfileSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsSaving(true)

    try {
      // Only send fields that the backend expects
      await api.put('/api/users/me', {
        username: profileForm.username,
        email: profileForm.email,
        firstName: profileForm.firstName,
        lastName: profileForm.lastName
      })
      await loadUserProfile()
      toast.success('Profile updated successfully')
    } catch (error) {
      console.error('Error updating profile:', error)
      toast.error(error instanceof Error ? error.message : 'Failed to update profile')
    } finally {
      setIsSaving(false)
    }
  }

  const handleSecuritySubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (securityForm.newPassword !== securityForm.confirmPassword) {
      toast.error('New passwords do not match')
      return
    }

    if (securityForm.newPassword.length < 8) {
      toast.error('New password must be at least 8 characters long')
      return
    }

    setIsSaving(true)

    try {
      await api.put('/api/users/me/password', {
        currentPassword: securityForm.currentPassword,
        newPassword: securityForm.newPassword
      })

      setSecurityForm({
        currentPassword: '',
        newPassword: '',
        confirmPassword: ''
      })
      toast.success('Password changed successfully')
    } catch (error) {
      console.error('Error changing password:', error)
      toast.error(error instanceof Error ? error.message : 'Failed to change password')
    } finally {
      setIsSaving(false)
    }
  }

  const handleNotificationUpdate = async (setting: keyof NotificationSettings, value: boolean) => {
    const updatedNotifications = { ...notifications, [setting]: value }
    setNotifications(updatedNotifications)

    try {
      await api.put('/api/users/me/notifications', updatedNotifications)
      toast.success('Notification settings updated')
    } catch (error) {
      console.error('Error updating notifications:', error)
      // Endpoint may not exist yet, just show warning instead of error
      toast.warning('Notification settings saved locally (backend endpoint not available yet)')
      // Don't revert - keep the local change
    }
  }

  const handlePreferenceUpdate = async (setting: keyof PreferenceSettings, value: string) => {
    const updatedPreferences = { ...preferences, [setting]: value }
    setPreferences(updatedPreferences)

    try {
      await api.put('/api/users/me/preferences', updatedPreferences)
      toast.success('Preferences updated')
    } catch (error) {
      console.error('Error updating preferences:', error)
      // Endpoint may not exist yet, just show warning instead of error
      toast.warning('Preferences saved locally (backend endpoint not available yet)')
      // Don't revert - keep the local change
    }
  }

  const toggleTwoFactor = async () => {
    try {
      const response = await api.post<{ enabled?: boolean }>('/api/users/me/2fa/toggle', {})
      await loadUserProfile()
      const newStatus = response.data?.enabled ?? !user?.twoFactorEnabled
      toast.success(`Two-factor authentication ${newStatus ? 'enabled' : 'disabled'}`)
    } catch (error) {
      console.error('Error toggling 2FA:', error)
      toast.error(error instanceof Error ? error.message : 'Failed to toggle two-factor authentication')
    }
  }

  const loadHardeningSettings = async () => {
    try {
      const response = await api.get<UserHardeningSettings>('/api/users/me/hardening')
      if (response.data) {
        setHardeningSettings(response.data)
      }
    } catch (error) {
      console.error('Error loading hardening settings:', error)
      // Use defaults if not available
    }
  }

  const saveHardeningSettings = async () => {
    try {
      setIsSaving(true)
      await api.put('/api/users/me/hardening', hardeningSettings)
      toast.success('Security hardening settings saved successfully')
    } catch (error) {
      console.error('Error saving hardening settings:', error)
      toast.error(error instanceof Error ? error.message : 'Failed to save hardening settings')
    } finally {
      setIsSaving(false)
    }
  }

  const addIPAddress = () => {
    if (newIPAddress.trim() && !hardeningSettings.allowedIPAddresses?.includes(newIPAddress.trim())) {
      setHardeningSettings({
        ...hardeningSettings,
        allowedIPAddresses: [...(hardeningSettings.allowedIPAddresses || []), newIPAddress.trim()]
      })
      setNewIPAddress('')
    }
  }

  const removeIPAddress = (ip: string) => {
    setHardeningSettings({
      ...hardeningSettings,
      allowedIPAddresses: hardeningSettings.allowedIPAddresses?.filter(addr => addr !== ip) || []
    })
  }

  const addTimeWindow = () => {
    setHardeningSettings({
      ...hardeningSettings,
      allowedTimeWindows: [...(hardeningSettings.allowedTimeWindows || []), { ...newTimeWindow }]
    })
    setNewTimeWindow({ start: '09:00', end: '17:00', daysOfWeek: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] })
  }

  const removeTimeWindow = (index: number) => {
    setHardeningSettings({
      ...hardeningSettings,
      allowedTimeWindows: hardeningSettings.allowedTimeWindows?.filter((_, i) => i !== index) || []
    })
  }

  const isAdmin = user?.role === 'Admin' || (user as any)?.roles?.includes('Admin')

  // Fetch users for management tab
  const { data: usersList = [], isLoading: usersLoading } = useQuery({
    queryKey: ['admin-users'],
    queryFn: async () => {
      const response = await api.get<UserListItem[]>('/api/users')
      return response.data
    },
    enabled: isAdmin && activeTab === 'users'
  })

  const tabs = [
    { id: 'profile', label: 'Profile', icon: User },
    { id: 'security', label: 'Security', icon: Shield },
    { id: 'notifications', label: 'Notifications', icon: Bell },
    { id: 'preferences', label: 'Preferences', icon: Settings },
    ...(isAdmin ? [{ id: 'users', label: 'Manage Users', icon: Users }] : [])
  ]

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center space-x-6">
          <div className="relative">
            <div className="h-20 w-20 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
              {user?.avatar ? (
                <img 
                  src={user.avatar} 
                  alt="Profile" 
                  className="h-20 w-20 rounded-full object-cover"
                />
              ) : (
                <span className="text-2xl font-semibold text-white">
                  {(user?.firstName?.[0] || user?.username?.[0] || 'U').toUpperCase()}
                </span>
              )}
            </div>
            <button className="absolute bottom-0 right-0 bg-white rounded-full p-1 shadow-lg border border-gray-200 hover:bg-gray-50">
              <Camera className="h-4 w-4 text-gray-600" />
            </button>
          </div>
          <div className="flex-1">
            <h1 className="text-2xl font-bold text-gray-900">
              {user?.fullName || user?.firstName ? `${user?.firstName} ${user?.lastName}`.trim() : user?.username}
            </h1>
            <p className="text-gray-600">{user?.email}</p>
            <div className="flex items-center space-x-4 mt-2">
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                {user?.role}
              </span>
              {user?.twoFactorEnabled && (
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                  <Shield className="h-3 w-3 mr-1" />
                  2FA Enabled
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8 px-6">
            {tabs.map((tab) => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`py-4 px-1 border-b-2 font-medium text-sm flex items-center space-x-2 ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  <span>{tab.label}</span>
                </button>
              )
            })}
          </nav>
        </div>

        {/* Tab Content */}
        <div className="p-6">
          {/* Profile Tab */}
          {activeTab === 'profile' && (
            <form onSubmit={handleProfileSubmit} className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Username
                  </label>
                  <input
                    type="text"
                    value={profileForm.username}
                    onChange={(e) => setProfileForm({ ...profileForm, username: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Email
                  </label>
                  <input
                    type="email"
                    value={profileForm.email}
                    onChange={(e) => setProfileForm({ ...profileForm, email: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    First Name
                  </label>
                  <input
                    type="text"
                    value={profileForm.firstName}
                    onChange={(e) => setProfileForm({ ...profileForm, firstName: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Last Name
                  </label>
                  <input
                    type="text"
                    value={profileForm.lastName}
                    onChange={(e) => setProfileForm({ ...profileForm, lastName: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>
              
              <div className="flex justify-end">
                <button
                  type="submit"
                  disabled={isSaving}
                  className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
                >
                  <Save className="h-4 w-4 mr-2" />
                  {isSaving ? 'Saving...' : 'Save Changes'}
                </button>
              </div>
            </form>
          )}

          {/* Security Tab */}
          {activeTab === 'security' && (
            <div className="space-y-8">
              {/* Password Change */}
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-4">Change Password</h3>
                <form onSubmit={handleSecuritySubmit} className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Current Password
                    </label>
                    <div className="relative">
                      <input
                        type={showPassword ? 'text' : 'password'}
                        value={securityForm.currentPassword}
                        onChange={(e) => setSecurityForm({ ...securityForm, currentPassword: e.target.value })}
                        className="w-full px-3 py-2 pr-10 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                        required
                      />
                      <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute inset-y-0 right-0 pr-3 flex items-center"
                      >
                        {showPassword ? <EyeOff className="h-4 w-4 text-gray-400" /> : <Eye className="h-4 w-4 text-gray-400" />}
                      </button>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        New Password
                      </label>
                      <div className="relative">
                        <input
                          type={showNewPassword ? 'text' : 'password'}
                          value={securityForm.newPassword}
                          onChange={(e) => setSecurityForm({ ...securityForm, newPassword: e.target.value })}
                          className="w-full px-3 py-2 pr-10 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                          required
                          minLength={8}
                        />
                        <button
                          type="button"
                          onClick={() => setShowNewPassword(!showNewPassword)}
                          className="absolute inset-y-0 right-0 pr-3 flex items-center"
                        >
                          {showNewPassword ? <EyeOff className="h-4 w-4 text-gray-400" /> : <Eye className="h-4 w-4 text-gray-400" />}
                        </button>
                      </div>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Confirm New Password
                      </label>
                      <div className="relative">
                        <input
                          type={showConfirmPassword ? 'text' : 'password'}
                          value={securityForm.confirmPassword}
                          onChange={(e) => setSecurityForm({ ...securityForm, confirmPassword: e.target.value })}
                          className="w-full px-3 py-2 pr-10 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                          required
                          minLength={8}
                        />
                        <button
                          type="button"
                          onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                          className="absolute inset-y-0 right-0 pr-3 flex items-center"
                        >
                          {showConfirmPassword ? <EyeOff className="h-4 w-4 text-gray-400" /> : <Eye className="h-4 w-4 text-gray-400" />}
                        </button>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex justify-end">
                    <button
                      type="submit"
                      disabled={isSaving}
                      className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
                    >
                      <Lock className="h-4 w-4 mr-2" />
                      {isSaving ? 'Changing...' : 'Change Password'}
                    </button>
                  </div>
                </form>
              </div>

              {/* Two-Factor Authentication */}
              <div className="border-t border-gray-200 pt-8">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Two-Factor Authentication</h3>
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <Shield className="h-6 w-6 text-green-600" />
                    <div>
                      <p className="font-medium text-gray-900">
                        Two-Factor Authentication {user?.twoFactorEnabled ? 'Enabled' : 'Disabled'}
                      </p>
                      <p className="text-sm text-gray-600">
                        Add an extra layer of security to your account
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={toggleTwoFactor}
                    className={`inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md ${
                      user?.twoFactorEnabled
                        ? 'text-red-700 bg-red-100 hover:bg-red-200'
                        : 'text-green-700 bg-green-100 hover:bg-green-200'
                    }`}
                  >
                    {user?.twoFactorEnabled ? 'Disable' : 'Enable'} 2FA
                  </button>
                </div>
              </div>

              {/* User Hardening Settings */}
              <div className="border-t border-gray-200 pt-8">
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h3 className="text-lg font-medium text-gray-900">Account Hardening</h3>
                    <p className="text-sm text-gray-600 mt-1">Configure security settings for your account</p>
                  </div>
                  <Button onClick={saveHardeningSettings} disabled={isSaving}>
                    <Save className="h-4 w-4 mr-2" />
                    {isSaving ? 'Saving...' : 'Save Settings'}
                  </Button>
                </div>

                <div className="space-y-6">
                  {/* Session Management */}
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h4 className="font-medium text-gray-900 mb-4 flex items-center gap-2">
                      <Clock className="h-5 w-5" />
                      Session Management
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <Label>Max Concurrent Sessions</Label>
                        <Input
                          type="number"
                          min={1}
                          max={10}
                          value={hardeningSettings.maxConcurrentSessions}
                          onChange={(e) => setHardeningSettings({ ...hardeningSettings, maxConcurrentSessions: parseInt(e.target.value) || 1 })}
                        />
                      </div>
                      <div>
                        <Label>Session Timeout (minutes)</Label>
                        <Input
                          type="number"
                          min={5}
                          max={1440}
                          value={hardeningSettings.sessionTimeoutMinutes}
                          onChange={(e) => setHardeningSettings({ ...hardeningSettings, sessionTimeoutMinutes: parseInt(e.target.value) || 60 })}
                        />
                      </div>
                    </div>
                    <div className="mt-4 flex items-center justify-between">
                      <div>
                        <Label>Require Re-authentication for Sensitive Actions</Label>
                        <p className="text-xs text-gray-500">Re-enter password for critical operations</p>
                      </div>
                      <Switch
                        checked={hardeningSettings.requireReauthForSensitive}
                        onCheckedChange={(checked) => setHardeningSettings({ ...hardeningSettings, requireReauthForSensitive: checked })}
                      />
                    </div>
                  </div>

                  {/* Access Restrictions */}
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h4 className="font-medium text-gray-900 mb-4 flex items-center gap-2">
                      <Ban className="h-5 w-5" />
                      Access Restrictions
                    </h4>
                    
                    {/* IP Restriction */}
                    <div className="mb-4">
                      <div className="flex items-center justify-between mb-2">
                        <div>
                          <Label>Restrict Login by IP Address</Label>
                          <p className="text-xs text-gray-500">Only allow login from specific IP addresses</p>
                        </div>
                        <Switch
                          checked={hardeningSettings.restrictLoginByIP}
                          onCheckedChange={(checked) => setHardeningSettings({ ...hardeningSettings, restrictLoginByIP: checked })}
                        />
                      </div>
                      {hardeningSettings.restrictLoginByIP && (
                        <div className="mt-3 space-y-2">
                          <div className="flex gap-2">
                            <Input
                              placeholder="192.168.1.1 or 192.168.1.0/24"
                              value={newIPAddress}
                              onChange={(e) => setNewIPAddress(e.target.value)}
                              onKeyPress={(e) => e.key === 'Enter' && addIPAddress()}
                            />
                            <Button type="button" onClick={addIPAddress} size="sm">
                              <Plus className="h-4 w-4" />
                            </Button>
                          </div>
                          <div className="flex flex-wrap gap-2">
                            {hardeningSettings.allowedIPAddresses?.map((ip) => (
                              <Badge key={ip} variant="secondary" className="flex items-center gap-1">
                                {ip}
                                <button onClick={() => removeIPAddress(ip)} className="ml-1">
                                  <X className="h-3 w-3" />
                                </button>
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Time Restriction */}
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <div>
                          <Label>Restrict Login by Time</Label>
                          <p className="text-xs text-gray-500">Only allow login during specific time windows</p>
                        </div>
                        <Switch
                          checked={hardeningSettings.restrictLoginByTime}
                          onCheckedChange={(checked) => setHardeningSettings({ ...hardeningSettings, restrictLoginByTime: checked })}
                        />
                      </div>
                      {hardeningSettings.restrictLoginByTime && (
                        <div className="mt-3 space-y-3">
                          <div className="flex gap-2">
                            <Input
                              type="time"
                              value={newTimeWindow.start}
                              onChange={(e) => setNewTimeWindow({ ...newTimeWindow, start: e.target.value })}
                            />
                            <Input
                              type="time"
                              value={newTimeWindow.end}
                              onChange={(e) => setNewTimeWindow({ ...newTimeWindow, end: e.target.value })}
                            />
                            <Button type="button" onClick={addTimeWindow} size="sm">
                              <Plus className="h-4 w-4" />
                            </Button>
                          </div>
                          <div className="space-y-2">
                            {hardeningSettings.allowedTimeWindows?.map((window, index) => (
                              <div key={index} className="flex items-center gap-2 p-2 bg-white rounded">
                                <span className="text-sm">{window.start} - {window.end}</span>
                                <button onClick={() => removeTimeWindow(index)} className="ml-auto">
                                  <X className="h-4 w-4 text-red-600" />
                                </button>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Login Security */}
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h4 className="font-medium text-gray-900 mb-4 flex items-center gap-2">
                      <Shield className="h-5 w-5" />
                      Login Security
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <Label>Max Failed Login Attempts</Label>
                        <Input
                          type="number"
                          min={1}
                          max={20}
                          value={hardeningSettings.maxFailedLoginAttempts}
                          onChange={(e) => setHardeningSettings({ ...hardeningSettings, maxFailedLoginAttempts: parseInt(e.target.value) || 5 })}
                        />
                      </div>
                      <div>
                        <Label>Lockout Duration (minutes)</Label>
                        <Input
                          type="number"
                          min={1}
                          max={1440}
                          value={hardeningSettings.lockoutDurationMinutes}
                          onChange={(e) => setHardeningSettings({ ...hardeningSettings, lockoutDurationMinutes: parseInt(e.target.value) || 30 })}
                        />
                      </div>
                    </div>
                    <div className="mt-4 space-y-2">
                      <div className="flex items-center justify-between">
                        <Label>Log All Login Attempts</Label>
                        <Switch
                          checked={hardeningSettings.logAllLoginAttempts}
                          onCheckedChange={(checked) => setHardeningSettings({ ...hardeningSettings, logAllLoginAttempts: checked })}
                        />
                      </div>
                      <div className="flex items-center justify-between">
                        <Label>Email Security Notifications</Label>
                        <Switch
                          checked={hardeningSettings.emailSecurityNotifications}
                          onCheckedChange={(checked) => setHardeningSettings({ ...hardeningSettings, emailSecurityNotifications: checked })}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Password Policy */}
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h4 className="font-medium text-gray-900 mb-4 flex items-center gap-2">
                      <Key className="h-5 w-5" />
                      Password Policy
                    </h4>
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <Label>Enable Password Expiration</Label>
                          <p className="text-xs text-gray-500">Require password change after set days</p>
                        </div>
                        <Switch
                          checked={hardeningSettings.enablePasswordExpiration}
                          onCheckedChange={(checked) => setHardeningSettings({ ...hardeningSettings, enablePasswordExpiration: checked })}
                        />
                      </div>
                      {hardeningSettings.enablePasswordExpiration && (
                        <div>
                          <Label>Password Expiration (days)</Label>
                          <Input
                            type="number"
                            min={1}
                            max={365}
                            value={hardeningSettings.passwordExpirationDays}
                            onChange={(e) => setHardeningSettings({ ...hardeningSettings, passwordExpirationDays: parseInt(e.target.value) || 90 })}
                          />
                        </div>
                      )}
                      <div className="flex items-center justify-between">
                        <div>
                          <Label>Prevent Password Reuse</Label>
                          <p className="text-xs text-gray-500">Remember previous passwords</p>
                        </div>
                        <Switch
                          checked={hardeningSettings.preventPasswordReuse}
                          onCheckedChange={(checked) => setHardeningSettings({ ...hardeningSettings, preventPasswordReuse: checked })}
                        />
                      </div>
                      {hardeningSettings.preventPasswordReuse && (
                        <div>
                          <Label>Password History Count</Label>
                          <Input
                            type="number"
                            min={0}
                            max={24}
                            value={hardeningSettings.passwordHistoryCount}
                            onChange={(e) => setHardeningSettings({ ...hardeningSettings, passwordHistoryCount: parseInt(e.target.value) || 5 })}
                          />
                        </div>
                      )}
                      <div className="flex items-center justify-between">
                        <div>
                          <Label>Require Strong Password</Label>
                          <p className="text-xs text-gray-500">Enforce complexity requirements</p>
                        </div>
                        <Switch
                          checked={hardeningSettings.requireStrongPassword}
                          onCheckedChange={(checked) => setHardeningSettings({ ...hardeningSettings, requireStrongPassword: checked })}
                        />
                      </div>
                      {hardeningSettings.requireStrongPassword && (
                        <div className="space-y-3 pl-4 border-l-2 border-gray-300">
                          <div>
                            <Label>Minimum Password Length</Label>
                            <Input
                              type="number"
                              min={6}
                              max={32}
                              value={hardeningSettings.minPasswordLength}
                              onChange={(e) => setHardeningSettings({ ...hardeningSettings, minPasswordLength: parseInt(e.target.value) || 8 })}
                            />
                          </div>
                          <div className="space-y-2">
                            <div className="flex items-center justify-between">
                              <Label className="text-sm">Require Uppercase</Label>
                              <Switch
                                checked={hardeningSettings.requireUppercase}
                                onCheckedChange={(checked) => setHardeningSettings({ ...hardeningSettings, requireUppercase: checked })}
                              />
                            </div>
                            <div className="flex items-center justify-between">
                              <Label className="text-sm">Require Lowercase</Label>
                              <Switch
                                checked={hardeningSettings.requireLowercase}
                                onCheckedChange={(checked) => setHardeningSettings({ ...hardeningSettings, requireLowercase: checked })}
                              />
                            </div>
                            <div className="flex items-center justify-between">
                              <Label className="text-sm">Require Digit</Label>
                              <Switch
                                checked={hardeningSettings.requireDigit}
                                onCheckedChange={(checked) => setHardeningSettings({ ...hardeningSettings, requireDigit: checked })}
                              />
                            </div>
                            <div className="flex items-center justify-between">
                              <Label className="text-sm">Require Special Character</Label>
                              <Switch
                                checked={hardeningSettings.requireSpecialChar}
                                onCheckedChange={(checked) => setHardeningSettings({ ...hardeningSettings, requireSpecialChar: checked })}
                              />
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Notifications Tab */}
          {activeTab === 'notifications' && (
            <div className="space-y-6">
              <h3 className="text-lg font-medium text-gray-900">Notification Preferences</h3>
              
              <div className="space-y-4">
                {Object.entries(notifications).map(([key, value]) => (
                  <div key={key} className="flex items-center justify-between py-3">
                    <div>
                      <h4 className="text-sm font-medium text-gray-900 capitalize">
                        {key.replace(/([A-Z])/g, ' $1').trim()}
                      </h4>
                      <p className="text-sm text-gray-600">
                        {getNotificationDescription(key as keyof NotificationSettings)}
                      </p>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        checked={value}
                        onChange={(e) => handleNotificationUpdate(key as keyof NotificationSettings, e.target.checked)}
                        className="sr-only peer"
                      />
                      <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                    </label>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Preferences Tab */}
          {activeTab === 'preferences' && (
            <div className="space-y-6">
              <h3 className="text-lg font-medium text-gray-900">System Preferences</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Theme
                  </label>
                  <select
                    value={preferences.theme}
                    onChange={(e) => handlePreferenceUpdate('theme', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="light">Light</option>
                    <option value="dark">Dark</option>
                    <option value="system">System</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Language
                  </label>
                  <select
                    value={preferences.language}
                    onChange={(e) => handlePreferenceUpdate('language', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="en">English</option>
                    <option value="id">Indonesian</option>
                    <option value="es">Spanish</option>
                    <option value="fr">French</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Timezone
                  </label>
                  <select
                    value={preferences.timezone}
                    onChange={(e) => handlePreferenceUpdate('timezone', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="UTC">UTC</option>
                    <option value="Asia/Jakarta">Asia/Jakarta</option>
                    <option value="America/New_York">America/New_York</option>
                    <option value="Europe/London">Europe/London</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Date Format
                  </label>
                  <select
                    value={preferences.dateFormat}
                    onChange={(e) => handlePreferenceUpdate('dateFormat', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="MM/dd/yyyy">MM/dd/yyyy</option>
                    <option value="dd/MM/yyyy">dd/MM/yyyy</option>
                    <option value="yyyy-MM-dd">yyyy-MM-dd</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Time Format
                  </label>
                  <select
                    value={preferences.timeFormat}
                    onChange={(e) => handlePreferenceUpdate('timeFormat', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="12h">12 Hour</option>
                    <option value="24h">24 Hour</option>
                  </select>
                </div>
              </div>
            </div>
          )}

          {/* User Management Tab */}
          {activeTab === 'users' && isAdmin && (
            <div className="space-y-6">
              {/* Header with title and help */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <h2 className="text-xl font-semibold text-gray-900">Manage Users</h2>
                  <button className="text-gray-400 hover:text-gray-600">
                    <AlertCircle className="h-4 w-4" />
                  </button>
                </div>
                <button className="text-sm text-blue-600 hover:text-blue-800 flex items-center gap-1">
                  <User className="h-4 w-4" />
                  User audit
                </button>
              </div>

              {/* Controls */}
              <div className="flex items-center justify-between gap-4">
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <Label>Select Category</Label>
                    <Select value={categoryFilter} onValueChange={setCategoryFilter}>
                      <SelectTrigger className="w-[180px]">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Users</SelectItem>
                        <SelectItem value="active">Active</SelectItem>
                        <SelectItem value="inactive">Inactive</SelectItem>
                        <SelectItem value="admin">Admins</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <Button variant="outline" size="sm">
                    <FileText className="h-4 w-4 mr-2" />
                    Manage Roles
                  </Button>
                  <Button size="sm" className="bg-green-600 hover:bg-green-700">
                    <Plus className="h-4 w-4 mr-2" />
                    Add User
                  </Button>
                </div>
              </div>

              {/* Search and Manage Toolbar */}
              <div className="flex items-center justify-between gap-4">
                <div className="flex items-center gap-2 flex-1">
                  <div className="relative flex-1 max-w-md">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                    <Input
                      placeholder="Search users..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-10"
                    />
                  </div>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="outline" size="sm">
                        Manage
                        <ChevronDown className="h-4 w-4 ml-2" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent>
                      <DropdownMenuItem>Bulk Actions</DropdownMenuItem>
                      <DropdownMenuItem>Export</DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
                <div className="flex items-center gap-4 text-sm text-gray-600">
                  <span>1 - {usersList.length} of {usersList.length}</span>
                  <Select defaultValue="10">
                    <SelectTrigger className="w-[70px]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="10">10</SelectItem>
                      <SelectItem value="25">25</SelectItem>
                      <SelectItem value="50">50</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {/* Users Table */}
              <div className="border rounded-lg overflow-hidden">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-12">
                        <Checkbox
                          checked={selectedUsers.length === usersList.length && usersList.length > 0}
                          onCheckedChange={(checked) => {
                            if (checked) {
                              setSelectedUsers(usersList.map(u => u.id))
                            } else {
                              setSelectedUsers([])
                            }
                          }}
                        />
                      </TableHead>
                      <TableHead className="w-24">Actions</TableHead>
                      <TableHead className="cursor-pointer">
                        <div className="flex items-center gap-2">
                          User Name
                          <ArrowUpDown className="h-4 w-4" />
                        </div>
                      </TableHead>
                      <TableHead>Delegated Roles</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {usersLoading ? (
                      <TableRow>
                        <TableCell colSpan={4} className="text-center py-8">
                          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600 mx-auto"></div>
                        </TableCell>
                      </TableRow>
                    ) : usersList.length === 0 ? (
                      <TableRow>
                        <TableCell colSpan={4} className="text-center py-8 text-gray-500">
                          No users found
                        </TableCell>
                      </TableRow>
                    ) : (
                      usersList
                        .filter(u => {
                          const matchesSearch = u.username.toLowerCase().includes(searchQuery.toLowerCase()) ||
                            u.email.toLowerCase().includes(searchQuery.toLowerCase())
                          const matchesCategory = categoryFilter === 'all' ||
                            (categoryFilter === 'active' && u.isActive) ||
                            (categoryFilter === 'inactive' && !u.isActive) ||
                            (categoryFilter === 'admin' && u.roles.includes('Admin'))
                          return matchesSearch && matchesCategory
                        })
                        .map((userItem) => (
                          <TableRow key={userItem.id}>
                            <TableCell>
                              <Checkbox
                                checked={selectedUsers.includes(userItem.id)}
                                onCheckedChange={(checked) => {
                                  if (checked) {
                                    setSelectedUsers([...selectedUsers, userItem.id])
                                  } else {
                                    setSelectedUsers(selectedUsers.filter(id => id !== userItem.id))
                                  }
                                }}
                              />
                            </TableCell>
                            <TableCell>
                              <div className="flex items-center gap-2">
                                <CheckCircle2 className="h-5 w-5 text-green-600" />
                                <button className="text-gray-400 hover:text-gray-600">
                                  <User className="h-5 w-5" />
                                </button>
                              </div>
                            </TableCell>
                            <TableCell className="font-medium">{userItem.username}</TableCell>
                            <TableCell>
                              <div className="flex items-center gap-2">
                                <div className="flex gap-1">
                                  {userItem.roles.map((role) => (
                                    <Badge key={role} variant="secondary" className="text-xs">
                                      {role}
                                    </Badge>
                                  ))}
                                </div>
                                <button className="text-blue-600 hover:text-blue-800 text-sm">
                                  Details
                                </button>
                              </div>
                            </TableCell>
                          </TableRow>
                        ))
                    )}
                  </TableBody>
                </Table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

const getNotificationDescription = (key: keyof NotificationSettings): string => {
  const descriptions: Record<keyof NotificationSettings, string> = {
    emailAlerts: 'Receive security alerts via email',
    pushNotifications: 'Receive push notifications in browser',
    securityAlerts: 'Get notified about security events',
    reportNotifications: 'Receive notifications when reports are generated',
    maintenanceNotifications: 'Get notified about system maintenance'
  }
  return descriptions[key]
}

export default UserProfile 