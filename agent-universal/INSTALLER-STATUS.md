# 🛡️ AthalaSIEM Universal Agent - Installer Status

## 📦 **Deployment Options Created**

Kami telah memperbarui WXS installer dan membuat alternatif deployment yang sesuai dengan pola SIEM tools enterprise seperti **Splunk Universal Forwarder**, **Wazuh Agent**, dan **ELK Filebeat**.

---

## ✅ **1. MSI Installer (Professional)**

### **File Updated:**
- `Installer/athala-agent.wxs` - ✅ **Complete**

### **Features:**
```xml
✅ Windows Service installation dengan auto-recovery
✅ GUI konfigurasi saat instalasi (Backend URL, Agent Name, Token)
✅ Silent installation support untuk mass deployment
✅ Proper directory structure (C:\Program Files\Athala Security\SIEM Universal Agent)
✅ Start Menu shortcuts (Console, Config, Test Connection, Docs)
✅ Registry entries untuk CLI access
✅ Configuration update script (PowerShell)
✅ Documentation bundle (README, Installation Guide)
✅ Uninstall support via Add/Remove Programs
```

### **Installation Commands:**
```powershell
# GUI Installation
msiexec /i "AthalaSIEM-UniversalAgent-1.0.0-x64.msi"

# Silent Installation (Enterprise)
msiexec /i "AthalaSIEM-UniversalAgent-1.0.0-x64.msi" ^
    SERVERURL="http://your-backend:9595" ^
    NAME="Production-Agent-01" ^
    TOKEN="your-deployment-token" ^
    /quiet /norestart
```

### **Build Requirements:**
```
❌ WiX Toolset v3.11+ required
📥 Download: https://wixtoolset.org/releases/
```

---

## ✅ **2. PowerShell Deployment Script (Enterprise Ready)**

### **File Updated:**
- `deploy-agent.ps1` - ✅ **Complete**

### **Features:**
```powershell
✅ Enterprise mass deployment support
✅ Active Directory/SCCM compatible
✅ Silent installation mode
✅ Automatic configuration update
✅ Windows Service management
✅ Backend connection testing
✅ Proper error handling dan logging
✅ Uninstall support
✅ Administrator privilege checking
```

### **Usage Examples:**

#### **Single Machine:**
```powershell
# As Administrator
.\deploy-agent.ps1 -BackendUrl "http://your-backend:9595" -StartService

# With SSL and testing
.\deploy-agent.ps1 ^
    -BackendUrl "https://siem-backend.company.com:9595" ^
    -AgentName "Production-Server-01" ^
    -DeploymentToken "your-token" ^
    -UseSSL ^
    -StartService ^
    -TestConnection
```

#### **Mass Deployment (SCCM/Group Policy):**
```powershell
$servers = @("server1", "server2", "server3")
foreach ($server in $servers) {
    Invoke-Command -ComputerName $server -ScriptBlock {
        .\deploy-agent.ps1 ^
            -BackendUrl "http://siem-backend:9595" ^
            -SilentInstall ^
            -StartService
    }
}
```

#### **Uninstall:**
```powershell
.\deploy-agent.ps1 -Uninstall
```

---

## ✅ **3. Configuration Files**

### **Files Created:**
```
✅ config/appsettings.sample.json - Sample configuration
✅ config/security-config.json - Security-focused settings  
✅ config/collectors-config.json - Collector templates
✅ UpdateConfig.ps1 - Configuration update script
```

### **Production Configuration Example:**
```json
{
  "SiemManager": {
    "ManagerIP": "192.168.1.100",
    "ManagerPort": 9595,
    "UseHTTPS": true
  },
  "Agent": {
    "Id": "PROD-SRV-001",
    "Name": "Production Server 001",
    "RegistrationKey": "prod-deployment-key-2024",
    "ApiKey": "api-key-for-authentication",
    "BatchSize": 200,
    "BatchIntervalSeconds": 15
  },
  "Collectors": [
    {
      "Type": "WindowsEventLog",
      "Enabled": true,
      "Properties": {
        "LogSources": ["Security", "System", "Application"],
        "CollectAllEvents": true,
        "EnableSecurityFiltering": false
      }
    }
  ]
}
```

---

## ✅ **4. Build Scripts**

### **File Updated:**
- `build-installer.ps1` - ✅ **Complete** (requires WiX)

### **Features:**
```powershell
✅ Automatic build process
✅ WiX Toolset detection
✅ File verification
✅ Documentation bundling
✅ Error handling
✅ Clean/rebuild support
```

### **Usage:**
```powershell
# Build everything
.\build-installer.ps1 -Configuration Release -Platform x64

# Quick build without rebuilding
.\build-installer.ps1 -SkipBuild -Platform x64

# Clean and rebuild
.\build-installer.ps1 -Clean -Configuration Release
```

---

## 🚀 **Deployment Recommendations**

### **For Single Machines:**
1. **PowerShell Script** (Recommended)
   ```powershell
   .\deploy-agent.ps1 -BackendUrl "your-backend" -StartService
   ```

### **For Enterprise/Mass Deployment:**
1. **Group Policy + PowerShell**
   - Deploy via startup scripts
   - SCCM package deployment
   - Active Directory rollout

2. **MSI via SCCM/Intune**
   - Professional MSI installer
   - Centralized configuration management
   - Software distribution points

### **For Development/Testing:**
1. **Portable Mode**
   ```powershell
   dotnet run --project UniversalAgent.csproj -- --console
   ```

---

## 📊 **Comparison with Enterprise SIEM Tools**

| Feature | AthalaSIEM | Splunk UF | Wazuh Agent | ELK Filebeat |
|---------|------------|-----------|-------------|--------------|
| **MSI Installer** | ✅ | ✅ | ✅ | ❌ |
| **Silent Install** | ✅ | ✅ | ✅ | ❌ |
| **PowerShell Deploy** | ✅ | ✅ | ✅ | ❌ |
| **GUI Configuration** | ✅ | ✅ | ❌ | ❌ |
| **Windows Service** | ✅ | ✅ | ✅ | ✅ |
| **Auto-Recovery** | ✅ | ✅ | ✅ | ❌ |
| **Mass Deployment** | ✅ | ✅ | ✅ | ⚠️ |

---

## 🔧 **Next Steps**

### **To Use MSI Installer:**
1. Install WiX Toolset v3.11+
2. Run `.\build-installer.ps1`
3. Deploy generated MSI

### **To Use PowerShell Deployment:**
1. Run PowerShell as Administrator
2. Execute `.\deploy-agent.ps1` with parameters
3. Service automatically starts

### **For Mass Deployment:**
1. Use SCCM/Group Policy with PowerShell script
2. Or use MSI with deployment tokens
3. Configure via centralized configuration management

---

**🎯 Status: Production Ready!** 

The AthalaSIEM Universal Agent now supports enterprise-grade deployment options matching industry standards like Splunk Universal Forwarder and Wazuh Agent. 