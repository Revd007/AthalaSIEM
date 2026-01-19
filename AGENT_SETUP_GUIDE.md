# 🚀 AthalaSIEM Agent Setup Guide

## Important: Events Come From The Agent!

**The backend API cannot generate events on its own.** Events must come from the **AthalaSIEM Agent** running on your machines. If the agent isn't running, you won't see any events in the frontend.

---

##  Step-by-Step Setup

### Step 1: Verify Agent Installation

1. **Check if agent is installed:**
   ```powershell
   # Run as Administrator
   Get-Service -Name "AthalaSIEMAgent"
   ```

2. **If service doesn't exist:**
   - Run the agent installer (`AthalaSIEMAgent.msi`) **as Administrator**
   - Follow the installation wizard
   - The service should be created automatically

### Step 2: Start the Agent Service

**If the service exists but isn't running:**

```powershell
# Run as Administrator
Start-Service -Name "AthalaSIEMAgent"
```

**Or use Services GUI:**
1. Press `Win + R`, type `services.msc`, press Enter
2. Find "Athala SIEM Agent"
3. Right-click → **Start**

### Step 3: Verify Agent Registration

The agent must register with the backend to appear in the frontend.

**Check registration:**
```powershell
# Check agent identity file
Get-Content "$env:ProgramData\AthalaSIEM\agent_identity.json"
```

**If not registered:**
- The agent will attempt to register automatically on startup
- Check agent logs for registration errors
- Ensure backend is running and accessible

### Step 4: Check Agent Logs

**View agent logs:**
```powershell
# Check latest log file
$logPath = "C:\Program Files\Athala SIEM Agent\logs"
Get-ChildItem $logPath -Filter "*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content -Tail 50
```

**Look for:**
-  `"Agent registered successfully"`
-  `"Collecting logs from Windows Event Log"`
-  `"Successfully sent X logs"`
- Any error messages

### Step 5: Verify Backend Connection

**Check agent configuration:**
```powershell
# View backend URL in config
$config = Get-Content "C:\Program Files\Athala SIEM Agent\appsettings.json" | ConvertFrom-Json
$config.Agent.BackendApiUrl
```

**Ensure backend is running:**
- Backend should be on: `http://localhost:9595` (or your configured URL)
- Test: Open `http://localhost:9595/swagger` in browser

### Step 6: Check Frontend

1. **Open frontend:** `http://localhost:7654`
2. **Go to Agents page:** Check if your agent appears
3. **Check agent status:** Should show "Online" if running

---

## 🔧 Quick Diagnostic Script

Run the diagnostic script to check everything at once:

```powershell
# Run as Administrator
.\backend\Scripts\check-agent-status.ps1
```

This will check:
-  Service status
-  Installation location
-  Agent registration
-  Backend connectivity
-  Log files

---

## 🐛 Common Issues & Solutions

### Issue 1: Agent Service Not Running

**Symptoms:**
- No events in frontend
- Agent doesn't appear in frontend

**Solution:**
```powershell
Start-Service -Name "AthalaSIEMAgent"
```

### Issue 2: Agent Not Registered

**Symptoms:**
- Agent service is running
- Agent doesn't appear in frontend
- No agent identity file

**Solution:**
1. Check agent logs for registration errors
2. Verify backend is running
3. Check backend URL in `appsettings.json`
4. Restart agent service

### Issue 3: No Events Being Collected

**Symptoms:**
- Agent is running and registered
- Agent appears in frontend
- But no events/logs visible

**Solution:**
1. Check `appsettings.json` - ensure collectors are enabled:
   ```json
   "Collectors": [
     {
       "Type": "WindowsEventLog",
       "Enabled": true,
       "Properties": {
         "EventLogs": "Application,System,Security"
       }
     }
   ]
   ```

2. Check Windows Event Log permissions:
   - Agent service needs access to Event Logs
   - Service should run as `LocalSystem` (default)

3. Check agent logs for collection errors

### Issue 4: Backend Not Receiving Events

**Symptoms:**
- Agent is collecting logs (see in agent logs)
- But events don't appear in backend/frontend

**Solution:**
1. Check backend logs for errors
2. Verify gRPC port (50051) is accessible
3. Check agent can reach backend:
   ```powershell
   Test-NetConnection -ComputerName localhost -Port 9595
   Test-NetConnection -ComputerName localhost -Port 50051
   ```

---

## 📊 Verification Checklist

Use this checklist to verify everything is working:

- [ ] Agent service is installed
- [ ] Agent service is **Running**
- [ ] Agent identity file exists (`$env:ProgramData\AthalaSIEM\agent_identity.json`)
- [ ] Agent appears in frontend at `/agents`
- [ ] Agent status shows "Online"
- [ ] Agent logs show "Collecting logs"
- [ ] Agent logs show "Successfully sent logs"
- [ ] Backend is running
- [ ] Backend logs show incoming log batches
- [ ] Events appear in frontend

---

## 🎯 Quick Start Commands

```powershell
# 1. Check service status
Get-Service -Name "AthalaSIEMAgent"

# 2. Start service (if stopped)
Start-Service -Name "AthalaSIEMAgent"

# 3. Check agent logs
Get-Content "C:\Program Files\Athala SIEM Agent\logs\agent-*.log" -Tail 20

# 4. Restart service
Restart-Service -Name "AthalaSIEMAgent"

# 5. Check if agent is registered
Test-Path "$env:ProgramData\AthalaSIEM\agent_identity.json"
```

---

## 📞 Still Having Issues?

1. **Check agent logs** - Look for errors
2. **Check backend logs** - Look for incoming requests
3. **Run diagnostic script** - `check-agent-status.ps1`
4. **Verify network connectivity** - Agent must reach backend
5. **Check Windows Event Viewer** - Look for service errors

---

## 🔄 Event Flow

Understanding how events flow:

```
Windows Event Log
       ↓
Agent Collector (WindowsEventLogCollector)
       ↓
Agent Normalizer
       ↓
Agent Buffer
       ↓
gRPC Forwarder → Backend (Port 50051)
       ↓
Backend Log Service
       ↓
Database (log_entries table)
       ↓
Frontend API (/api/logs)
       ↓
Frontend UI
```

**If any step fails, events won't appear in the frontend!**
