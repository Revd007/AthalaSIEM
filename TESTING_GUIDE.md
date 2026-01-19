# AthalaSIEM Testing Guide

Complete guide for testing Agent, Backend, and Frontend components.

## Prerequisites

- .NET 8 SDK installed
- Node.js 18+ and npm installed
- PostgreSQL database running
- Windows OS (for agent testing)

## Quick Start Testing

### 1. Backend Setup & Testing

#### Step 1: Configure Backend

Edit `backend/appsettings.json`:

```json
{
  "ConnectionStrings": {
    "DefaultConnection": "Host=localhost;Port=5432;Database=siem-db;Username=youruser;Password=yourpassword;"
  },
  "Kestrel": {
    "Endpoints": {
      "Http": {
        "Url": "http://0.0.0.0:9595"
      }
    }
  },
  "Jwt": {
    "Key": "your-256-bit-secret-key-change-in-production",
    "Issuer": "AthalaSIEM",
    "Audience": "AthalaSIEMUsers",
    "ExpireMinutes": 60
  }
}
```

#### Step 2: Run Database Migrations

```powershell
cd backend
dotnet ef database update
```

#### Step 3: Start Backend

```powershell
cd backend
dotnet run
```

**Expected Output:**
```
🚀 Backend server listening on port: 9595
💡 Override via environment: ATHALA_Kestrel__Endpoints__Http__Url=http://0.0.0.0:YOUR_PORT
```

**Verify Backend:**
- HTTP API: http://localhost:9595/swagger
- Health Check: http://localhost:9595/health
- gRPC: http://localhost:9595 (same port, HTTP/2)

#### Step 4: Test Backend Endpoints

```powershell
# Test health endpoint
curl http://localhost:9595/health

# Test agent registration (get deployment token first)
curl -X POST http://localhost:9595/api/agents/register `
  -H "Content-Type: application/json" `
  -d '{\"agentName\":\"TestAgent\",\"deploymentToken\":\"athala-siem-agent-registration-2025\"}'
```

---

### 2. Agent Setup & Testing

#### Step 1: Configure Agent

Edit `agent-universal/appsettings.json`:

```json
{
  "SiemManager": {
    "ManagerIP": "localhost",
    "ManagerPort": 9595,
    "UseHTTPS": false
  },
  "Agent": {
    "Id": "TEST-AGENT-001",
    "Name": "Test Agent",
    "DeploymentToken": "athala-siem-agent-registration-2025",
    "BatchSize": 100,
    "BatchIntervalSeconds": 30,
    "HeartbeatIntervalSeconds": 30
  },
  "Collectors": [
    {
      "Type": "WindowsEventLog",
      "Enabled": true,
      "Properties": {
        "LogSources": ["Security", "System", "Application"]
      }
    },
    {
      "Type": "FileIntegrity",
      "Enabled": true,
      "Properties": {
        "FallbackPaths": ["C:\\TestLogs"]
      }
    }
  ]
}
```

#### Step 2: Build Agent

```powershell
cd agent-universal
dotnet build -c Release
```

#### Step 3: Test Agent in Console Mode

```powershell
cd agent-universal
dotnet run -- --test-connection
```

**Expected Output:**
```
🔗 Testing connection to backend...
✅ Connection successful!
```

#### Step 4: Run Agent (Console Mode for Testing)

```powershell
cd agent-universal
dotnet run
```

**Expected Output:**
```
Initializing gRPC communication to: http://localhost:9595
gRPC communication service initialized successfully
Agent registered successfully
gRPC streams initialized
```

#### Step 5: Verify Agent Registration

Check backend logs or query:

```powershell
# Check agents via API
curl http://localhost:9595/api/agents
```

#### Step 6: Test gRPC Communication

The agent should automatically:
1. Register via HTTP (control plane)
2. Switch to gRPC streaming (data plane)
3. Send heartbeats every 30 seconds
4. Send log batches every 30 seconds

**Monitor Agent Logs:**
- Look for: "Successfully sent {LogCount} logs via gRPC"
- Look for: "Heartbeat sent successfully"

---

### 3. Frontend Setup & Testing

#### Step 1: Install Dependencies

```powershell
cd frontend
npm install
```

#### Step 2: Configure Frontend

Create or edit `frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:9595
NEXT_PUBLIC_WS_URL=ws://localhost:9595
```

#### Step 3: Start Frontend

```powershell
cd frontend
npm run dev
```

**Expected Output:**
```
- ready started server on 0.0.0.0:3000
- Local: http://localhost:3000
```

#### Step 4: Access Frontend

Open browser: http://localhost:3000

**Test Pages:**
- Dashboard: http://localhost:3000/dashboard
- Agents: http://localhost:3000/dashboard/agents
- Logs: http://localhost:3000/dashboard/events
- Normalization: http://localhost:3000/dashboard/normalization
- Correlation: http://localhost:3000/dashboard/correlation

---

## End-to-End Testing Scenarios

### Scenario 1: Agent Registration Flow

1. **Start Backend**
   ```powershell
   cd backend
   dotnet run
   ```

2. **Start Agent**
   ```powershell
   cd agent-universal
   dotnet run
   ```

3. **Verify in Frontend**
   - Navigate to: http://localhost:3000/dashboard/agents
   - Should see agent listed with status "Online"

4. **Check Backend Logs**
   - Should see: "Agent registered: TEST-AGENT-001"
   - Should see: "gRPC stream established"

### Scenario 2: Log Collection & Normalization

1. **Trigger Log Collection**
   - Agent automatically collects Windows Event Logs
   - Or manually trigger: Create a test file in monitored FIM path

2. **Verify Logs in Backend**
   ```powershell
   curl http://localhost:9595/api/logs?limit=10
   ```

3. **Check Normalization**
   - Navigate to: http://localhost:3000/dashboard/normalization
   - Should see normalized logs with ECS fields

4. **Verify in Database**
   ```sql
   SELECT COUNT(*) FROM "LogEntries";
   SELECT COUNT(*) FROM "NormalizedLogs";
   ```

### Scenario 3: Correlation & Alerts

1. **Generate Test Events**
   - Agent collects multiple failed login attempts
   - Or manually insert test logs via API

2. **Check Correlation**
   - Navigate to: http://localhost:3000/dashboard/correlation
   - Should see correlation rules and alerts

3. **Verify Alerts**
   ```powershell
   curl http://localhost:9595/api/alerts?recent=true
   ```

### Scenario 4: gRPC Streaming Test

1. **Monitor Network Traffic**
   ```powershell
   # Use Wireshark or netstat to monitor connections
   netstat -an | findstr 9595
   ```

2. **Check Agent Logs**
   - Should see: "Log stream established, accepted: X, rejected: Y"
   - Should see: "Heartbeat stream active"

3. **Check Backend Logs**
   - Should see: "StreamLogs called"
   - Should see: "StreamHeartbeat called"

---

## Testing Commands Reference

### Backend Testing

```powershell
# Run backend
cd backend
dotnet run

# Run with specific port
$env:ATHALA_Kestrel__Endpoints__Http__Url="http://0.0.0.0:9596"
dotnet run

# Run migrations
dotnet ef database update

# Check database connection
dotnet ef dbcontext info
```

### Agent Testing

```powershell
# Test connection only
cd agent-universal
dotnet run -- --test-connection

# Run in console mode
dotnet run

# Build for deployment
dotnet publish -c Release -r win-x64 --self-contained -o ./publish

# Deploy agent (Windows)
.\deploy-agent.ps1 -BackendUrl "http://localhost:9595" -AgentName "TestAgent" -StartService
```

### Frontend Testing

```powershell
# Development mode
cd frontend
npm run dev

# Production build
npm run build
npm start

# Run tests
npm test
```

---

## Troubleshooting

### Backend Issues

**Problem: Port already in use**
```powershell
# Find process using port
netstat -ano | findstr :9595
# Kill process
taskkill /PID <PID> /F
```

**Problem: Database connection failed**
- Verify PostgreSQL is running
- Check connection string in `appsettings.json`
- Test connection: `psql -h localhost -U youruser -d siem-db`

**Problem: gRPC not working**
- Verify HTTP/2 is enabled: Check `Program.cs` for `HttpProtocols.Http1AndHttp2`
- Test with: `curl -v --http2 http://localhost:9595/health`

### Agent Issues

**Problem: Agent cannot connect to backend**
- Verify backend is running: `curl http://localhost:9595/health`
- Check `SiemManager:ManagerIP` and `SiemManager:ManagerPort` in `appsettings.json`
- Check firewall rules

**Problem: gRPC connection fails**
- Verify backend supports HTTP/2
- Check agent logs for gRPC errors
- Fallback to HTTP: Agent should automatically fallback

**Problem: No logs being collected**
- Check collector configuration in `appsettings.json`
- Verify collectors are enabled
- Check agent logs for collector errors

### Frontend Issues

**Problem: Cannot connect to backend**
- Verify `NEXT_PUBLIC_API_URL` in `.env.local`
- Check CORS settings in backend `Program.cs`
- Verify backend is running

**Problem: No data displayed**
- Check browser console for errors
- Verify API endpoints are accessible
- Check network tab in browser DevTools

---

## Performance Testing

### Load Test Agent

```powershell
# Run agent with high log volume
# Modify appsettings.json:
# "BatchSize": 1000
# "BatchIntervalSeconds": 10
```

### Monitor System Resources

```powershell
# Windows Performance Monitor
perfmon

# Check agent process
Get-Process | Where-Object {$_.ProcessName -like "*athala*"}
```

---

## Next Steps

1. **Production Deployment**: See `README_ENTERPRISE_DEPLOYMENT.md`
2. **Security Hardening**: Configure TLS/HTTPS
3. **Scaling**: Set up multiple agents and load balancing
4. **Monitoring**: Set up logging and metrics collection

---

## Quick Test Checklist

- [ ] Backend starts on port 9595
- [ ] Backend Swagger UI accessible
- [ ] Database migrations applied
- [ ] Agent connects to backend
- [ ] Agent registers successfully
- [ ] gRPC streaming established
- [ ] Logs are collected and sent
- [ ] Frontend displays agent status
- [ ] Logs visible in frontend
- [ ] Normalization working
- [ ] Correlation generating alerts

---

## Support

For issues or questions:
1. Check logs in `backend/logs` and `agent-universal/logs`
2. Review error messages in console
3. Check database connectivity
4. Verify configuration files
