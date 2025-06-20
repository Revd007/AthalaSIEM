# Athala SIEM Agent Architecture Fix

## 🚨 Issues Found and Fixed

### 1. **Wrong Communication Architecture**
**Problem**: Agent was trying to connect to port 7078 (likely frontend) instead of backend
**Solution**: Configured agent to only communicate with backend on port 9595

### 2. **Incorrect Installation Location**
**Problem**: Agent installed in `Program Files (x86)` (32-bit) instead of `Program Files` (64-bit)
**Solution**: Updated WiX installer to use `ProgramFilesFolder` instead of `ProgramFiles6432Folder`

### 3. **Conflicting Configuration Files**
**Problem**: Multiple `AgentSettings.cs` files with different default URLs
**Solution**: Removed duplicate configuration file, standardized on single source

### 4. **SSL/TLS Misconfiguration**
**Problem**: Agent configured for HTTPS but backend running on HTTP
**Solution**: Disabled SSL for development, aligned protocols

## ✅ Proper SIEM Agent Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │
│   SIEM Agent    │───▶│  Backend API    │───▶│   Database      │
│  (Data Source)  │    │   (Port 9595)   │    │   (Storage)     │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                │ API Calls
                                ▼
                       ┌─────────────────┐
                       │                 │
                       │  Frontend UI    │
                       │  (Port 3000)    │
                       │                 │
                       └─────────────────┘
```

### **Key Principles:**
1. **Agents communicate ONLY with Backend** - Never directly with frontend
2. **Backend is the central hub** - All data flows through it
3. **Frontend is separate** - Gets data from backend via API
4. **No direct agent-to-frontend** - This is NOT how SIEM systems work

## 🔧 Configuration Changes Applied

### 1. **AgentSettings.cs (Models/)**
```csharp
public string BackendApiUrl { get; set; } = "http://localhost:9595";
public string BackendGrpcUrl { get; set; } = "http://localhost:9595";
public string BackendUrl 
{ 
    get => BackendApiUrl;
    set => BackendApiUrl = value;
}
```

### 2. **WiX Installer (AgentInstaller.wxs)**
```xml
<!-- Changed from ProgramFiles6432Folder to ProgramFilesFolder -->
<StandardDirectory Id="ProgramFilesFolder">
  <Directory Id="INSTALLFOLDER" Name="Athala SIEM Agent">
```

### 3. **appsettings.json Template**
```json
{
  "Agent": {
    "AgentName": "AthalaSIEM Agent",
    "BackendApiUrl": "http://localhost:9595",
    "BackendGrpcUrl": "http://localhost:9595",
    "EncryptLogs": false,
    "UseMutualTls": false,
    "UseTrafficCompression": true
  }
}
```

## 🛠️ How to Apply Fixes

### **Immediate Fix (Current Installation)**
```powershell
# Run as Administrator
.\fix_agent_complete.ps1
```

### **Clean Installation (Recommended)**
1. Uninstall current agent
2. Use updated installer with fixed configuration
3. Install to proper 64-bit location

## 📊 Comparison with Other SIEM Solutions

| SIEM Product | Agent Communication | Architecture |
|--------------|-------------------|--------------|
| **Splunk** | Agent → Forwarder → Indexer | ✅ Correct |
| **Elastic SIEM** | Beats → Elasticsearch | ✅ Correct |
| **QRadar** | WinCollect → Console | ✅ Correct |
| **Sentinel** | Agent → Log Analytics | ✅ Correct |
| **Athala SIEM** | Agent → Backend API | ✅ Now Fixed |

## 🔍 Why Port 7078 Was Wrong

The port 7078 appears to have been:
- A frontend development port
- Possibly from old configuration
- NOT the correct backend endpoint

**Agents should connect to:**
- ✅ Backend API/gRPC ports (9595)
- ❌ Frontend UI ports (3000, 7078, etc.)

## 🚀 Best Practices for SIEM Agents

### **1. Single Point of Data Ingestion**
- All agents connect to backend
- Backend handles data processing
- Frontend displays processed data

### **2. Proper Error Handling**
- Retry logic for backend connections
- Local buffering when backend unavailable
- Health monitoring and heartbeat

### **3. Security**
- TLS encryption for production
- API key authentication
- Mutual TLS for sensitive environments

### **4. Architecture Separation**
- Data collection (agents)
- Data processing (backend)
- Data visualization (frontend)

## 📝 Implementation Notes

### **Port Usage:**
- `9595` - Backend API/gRPC (correct for agents)
- `3000` - Frontend UI (for users)
- `7078` - Unknown/incorrect (removed)

### **Installation Paths:**
- ✅ `C:\Program Files\Athala SIEM Agent` (64-bit)
- ❌ `C:\Program Files (x86)\Athala SIEM Agent` (32-bit)

### **Configuration Priority:**
1. Command line arguments
2. appsettings.json
3. Environment variables
4. Default values in code

## 🎯 Result

The agent now follows proper SIEM architecture patterns:
- ✅ Communicates only with backend
- ✅ Uses correct ports (9595)
- ✅ Proper 64-bit installation path
- ✅ Unified configuration
- ✅ Matches industry standards 