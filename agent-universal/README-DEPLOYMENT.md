# Athala SIEM Universal Agent - Deployment Guide

## Simple Installation (Recommended)

### Option 1: PowerShell Deployment Script
```powershell
# Run PowerShell as Administrator
.\deploy-agent.ps1 -InstallPath "C:\Program Files\Athala SIEM Universal Agent" -BackendUrl "http://YOUR-BACKEND:9595"
```

### Option 2: Manual Installation (Like Splunk/Wazuh)
```powershell
# 1. Copy files to installation directory
mkdir "C:\Program Files\Athala SIEM Universal Agent"
copy dist\publish\* "C:\Program Files\Athala SIEM Universal Agent\"

# 2. Install as Windows Service
sc create "AthalaSIEMUniversalAgent" binPath= "C:\Program Files\Athala SIEM Universal Agent\athala-agent.exe" start= auto displayName= "Athala SIEM Universal Agent"

# 3. Start service
sc start AthalaSIEMUniversalAgent

# 4. Check status
sc query AthalaSIEMUniversalAgent
```

### Option 3: Portable/Console Mode
```powershell
# Just run directly without service installation
dist\publish\athala-agent.exe --console
```

## Configuration

Edit `appsettings.json`:
```json
{
  "BackendUrl": "http://YOUR-BACKEND:9595",
  "AgentName": "AthalaSIEMUniversalAgent",
  "LogLevel": {
    "Default": "Information"
  }
}
```

## Service Management

```powershell
# Start service
sc start AthalaSIEMUniversalAgent

# Stop service
sc stop AthalaSIEMUniversalAgent

# Restart service
sc stop AthalaSIEMUniversalAgent && sc start AthalaSIEMUniversalAgent

# Check status
sc query AthalaSIEMUniversalAgent

# Uninstall service
sc delete AthalaSIEMUniversalAgent
```

## Size Comparison

- **Original Agent**: 200+ MB
- **Universal Agent**: 0.82 MB
- **Deployment**: Simple ZIP + PowerShell (like Splunk Universal Forwarder)

## Troubleshooting

### Test Connection
```powershell
athala-agent.exe --test-connection
```

### Console Mode (Debug)
```powershell
athala-agent.exe --console
```

### Check Dependencies
```powershell
athala-agent.exe --help
``` 