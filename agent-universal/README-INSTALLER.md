# 🛡️ AthalaSIEM Universal Agent - Fixed Installer Guide

## 🎯 **WHAT WAS FIXED**

**Previous Issues:**
- ❌ `hostpolicy.dll` not found error
- ❌ Service failed to start (Error 1920)  
- ❌ .NET runtime dependencies missing
- ❌ Hardcoded Manager IP (192.168.1.100)

**✅ NOW FIXED:**
- ✅ **Complete .NET runtime inclusion** - All hostpolicy.dll, coreclr.dll, etc.
- ✅ **Proper service installation** - Service starts correctly
- ✅ **Self-contained deployment** - No external .NET required
- ✅ **User-input Manager IP** - GUI asks for YOUR backend IP
- ✅ **Enterprise deployment ready** - PowerShell + MSI options

---

## 🔧 **BUILD THE INSTALLER**

### **Prerequisites**
```powershell
# 1. Install WiX Toolset v3.11+
# Download from: https://wixtoolset.org/releases/

# 2. Verify installation
candle.exe -? 
light.exe -?
```

### **Build Command**
```powershell
# Run PowerShell as Administrator in agent-universal/ directory
cd agent-universal
.\build-installer.ps1 -Configuration Release -Platform x64
```

**Output will be in `dist/deployment/`:**
- ✅ `AthalaSIEM-UniversalAgent-1.0.0-x64.msi` (MSI Installer)
- ✅ `deploy-agent.ps1` (PowerShell deployment script)
- ✅ `portable/` (Portable files)
- ✅ `README.md` (Installation instructions)

---

## 📦 **INSTALLATION OPTIONS**

### **Option 1: MSI Installer (Recommended)**

#### **GUI Installation**
```powershell
# Double-click the MSI file
.\AthalaSIEM-UniversalAgent-1.0.0-x64.msi

# The installer will ask for:
# - Manager IP: YOUR_BACKEND_SERVER_IP (REQUIRED!)
# - Manager Port: 9595 (default)
# - Agent Name: Unique name for this agent
```

#### **Silent Installation**
```powershell
# Silent installation with YOUR backend IP
msiexec /i "AthalaSIEM-UniversalAgent-1.0.0-x64.msi" ^
    MANAGERIP="YOUR_BACKEND_SERVER_IP" ^
    MANAGERPORT="9595" ^
    NAME="Production-Agent-01" ^
    /quiet /norestart

# Example with real IP:
msiexec /i "AthalaSIEM-UniversalAgent-1.0.0-x64.msi" ^
    MANAGERIP="10.0.1.50" ^
    MANAGERPORT="9595" ^
    NAME="WebServer-01" ^
    /quiet /norestart
```

### **Option 2: PowerShell Deployment**

#### **Single Machine**
```powershell
# Run PowerShell as Administrator
.\deploy-agent.ps1 -BackendUrl "http://YOUR_BACKEND_IP:9595" -StartService

# Example with real backend:
.\deploy-agent.ps1 -BackendUrl "http://10.0.1.50:9595" -StartService -TestConnection
```

#### **Mass Deployment** 
```powershell
# Deploy to multiple servers (SCCM/Group Policy compatible)
$servers = @("server1", "server2", "server3")
$backendUrl = "http://10.0.1.50:9595"  # ← YOUR BACKEND IP

foreach ($server in $servers) {
    Invoke-Command -ComputerName $server -ScriptBlock {
        param($url, $serverName)
        .\deploy-agent.ps1 -BackendUrl $url -AgentName $serverName -StartService -SilentInstall
    } -ArgumentList $backendUrl, $server
}
```

---

## ⚠️ **CRITICAL REQUIREMENTS**

### **1. Administrator Privileges Required**
```powershell
# ❌ WILL FAIL WITHOUT ADMIN RIGHTS
# The agent MUST run as Administrator to access Security Event Logs
# Without Security logs, this is NOT a functional SIEM agent!

# ✅ RIGHT-CLICK PowerShell → "Run as Administrator"
```

### **2. Manager IP is REQUIRED**
```
❌ NO MORE hardcoded defaults like 192.168.1.100
✅ You MUST specify your actual backend server IP
✅ GUI installer will ask for Manager IP during setup
✅ Silent install requires MANAGERIP parameter
```

### **3. Network Requirements**
```powershell
# Backend must be accessible on port 9595
Test-NetConnection YOUR_BACKEND_IP -Port 9595

# Test agent connection after installation
"C:\Program Files\Athala Security\SIEM Universal Agent\athala-agent.exe" --test-connection
```

---

## 🔍 **VERIFICATION COMMANDS**

### **Service Management**
```powershell
# Check service status
Get-Service AthalaSIEMUniversalAgent

# Start/Stop service
Start-Service AthalaSIEMUniversalAgent
Stop-Service AthalaSIEMUniversalAgent

# Restart service
Restart-Service AthalaSIEMUniversalAgent
```

### **Agent Commands**
```powershell
# Test connection to backend
"C:\Program Files\Athala Security\SIEM Universal Agent\athala-agent.exe" --test-connection

# Show current configuration
"C:\Program Files\Athala Security\SIEM Universal Agent\athala-agent.exe" --config

# Run in console mode for debugging
"C:\Program Files\Athala Security\SIEM Universal Agent\athala-agent.exe" --console

# Show help
"C:\Program Files\Athala Security\SIEM Universal Agent\athala-agent.exe" --help
```

---

## 🚨 **TROUBLESHOOTING**

### **Common Issues**

#### **"hostpolicy.dll not found"**
```
✅ FIXED in new installer!
The MSI now includes ALL .NET runtime files properly.
```

#### **"Service failed to start" (Error 1920)**
```
✅ FIXED in new installer!
Service installation is now properly configured.
```

#### **"Manager IP not configured"**
```
Solution:
1. Edit appsettings.json manually:
   "SiemManager": {
     "ManagerIP": "YOUR_BACKEND_SERVER_IP",
     "ManagerPort": 9595
   }

2. Or reinstall with correct MANAGERIP parameter
```

#### **"Cannot access Security Event Log"**
```
Solution: Run as Administrator
Right-click PowerShell → "Run as Administrator"
```

### **Diagnostic Commands**
```powershell
# Full health check
Get-Service AthalaSIEMUniversalAgent
Get-Process athala-agent -ErrorAction SilentlyContinue
Get-WinEvent -LogName Application -Source "AthalaSIEM*" -MaxEvents 10

# Check configuration
"C:\Program Files\Athala Security\SIEM Universal Agent\athala-agent.exe" --config

# Test connection
"C:\Program Files\Athala Security\SIEM Universal Agent\athala-agent.exe" --test-connection
```

---

## 🏢 **ENTERPRISE FEATURES NOW WORKING**

✅ **Windows Event Log Collection** - Security, System, Application (ALL events)  
✅ **File Integrity Monitoring (FIM)** - Real-time + periodic scan with SHA256  
✅ **Registry Monitoring** - Critical keys monitoring with threat detection  
✅ **Local Analysis Engine** - Rules, correlators, enrichers for attack detection  
✅ **Batch Processing** - 100 logs/30 seconds (configurable)  
✅ **Health Monitoring** - Heartbeat every 1 minute with auto-recovery  
✅ **Enterprise Deployment** - MSI + PowerShell + Group Policy support  

---

## 🎉 **READY FOR PRODUCTION!**

The installer is now **PRODUCTION-READY** and follows enterprise SIEM patterns like:
- ✅ **Splunk Universal Forwarder** - MSI installer, silent deployment
- ✅ **Wazuh Agent** - Service management, mass deployment
- ✅ **ManageEngine EventLog Analyzer** - Security filtering, correlation

**🚀 Deploy with confidence!** The missing .NET runtime files and service issues are now resolved. 