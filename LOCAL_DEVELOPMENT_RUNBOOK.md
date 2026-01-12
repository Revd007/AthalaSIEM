# AthalaSIEM Local Development Runbook

## Quick Start (5 Minutes to Running System)

### 1. Start Backend (2 minutes)

```bash
cd backend

# Install dependencies (first time only)
dotnet restore

# Update database connection string in appsettings.json
# ConnectionStrings:DefaultConnection = "Host=localhost;Database=athalasiem;Username=postgres;Password=yourpassword"

# Run migrations
dotnet ef database update

# Start backend
dotnet run
```

Backend will start on:
- HTTP API: `http://localhost:9595`
- gRPC: `http://localhost:9595` (same port, different protocol)
- Swagger: `http://localhost:9595/swagger`

### 2. Start Frontend (1 minute)

```bash
cd frontend

# Install dependencies (first time only)
npm install

# Start frontend
npm run dev
```

Frontend will start on `http://localhost:3000`

### 3. Build and Run Agent (2 minutes)

#### Option A: Run from Source

```bash
cd agent

# Install dependencies
dotnet restore

# Update appsettings.json with backend URL
# BackendGrpcUrl = "http://localhost:9595"

# Run agent
dotnet run
```

#### Option B: Install as Windows Service

```powershell
cd agent

# Build release
dotnet publish -c Release -o ./publish

# Install service (requires admin)
cd publish
.\AthalaSIEM.Agent.exe install

# Start service
.\AthalaSIEM.Agent.exe start
```

---

## Configuration Files

### Backend (`backend/appsettings.json`)

```json
{
  "ConnectionStrings": {
    "DefaultConnection": "Host=localhost;Database=athalasiem;Username=postgres;Password=yourpassword"
  },
  "GrpcServer": {
    "Url": "http://0.0.0.0:9595"
  },
  "Cors": {
    "AllowedOrigins": ["http://localhost:3000", "http://localhost:7654"]
  },
  "Jwt": {
    "SecretKey": "your-secret-key-here",
    "Issuer": "AthalaSIEM",
    "Audience": "AthalaSIEM-Users"
  }
}
```

### Agent (`agent/appsettings.json`)

```json
{
  "AgentSettings": {
    "BackendGrpcUrl": "http://localhost:9595",
    "BackendApiUrl": "http://localhost:9595",
    "HeartbeatIntervalMinutes": 1,
    "LogBatchSize": 100,
    "LogSendingIntervalSeconds": 30
  },
  "Collectors": {
    "WindowsEventLog": {
      "Enabled": true,
      "Logs": ["Security", "System", "Application"]
    }
  }
}
```

---

## Database Setup

### Create Database

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE athalasiem;

# Exit
\q
```

### Run Migrations

```bash
cd backend
dotnet ef database update
```

### Seed Initial Data

The backend automatically seeds:
- Admin user (username: `admin`, password: `Admin123!`)
- Default roles (Admin, Operator, Viewer)

---

## Testing the System

### 1. Register an Agent

```bash
# Generate deployment token (via API)
curl -X POST http://localhost:9595/api/agents/generate-token \
  -H "Authorization: Bearer <jwt-token>" \
  -H "Content-Type: application/json" \
  -d '{"expiresInHours": 24}'

# Or register directly
curl -X POST http://localhost:9595/api/agents/register \
  -H "Content-Type: application/json" \
  -d '{
    "hostname": "test-agent",
    "ipAddress": "127.0.0.1",
    "operatingSystem": "Windows",
    "version": "1.0.0"
  }'
```

### 2. Check Agent Status

```bash
# List all agents
curl http://localhost:9595/api/agents

# Get specific agent
curl http://localhost:9595/api/agents/<agent-id>
```

### 3. View Logs

```bash
# Get logs
curl "http://localhost:9595/api/logs?agentId=<agent-id>&limit=100"

# Search logs
curl "http://localhost:9595/api/logs/search?query=error&limit=50"
```

### 4. View Alerts

```bash
# Get recent alerts
curl "http://localhost:9595/api/alerts?recent=true"

# Get alerts by severity
curl "http://localhost:9595/api/alerts?severity=High"
```

---

## Development Workflow

### Backend Development

1. **Make changes** to backend code
2. **Restart backend** (`dotnet run` or `dotnet watch run`)
3. **Test via Swagger** at `http://localhost:9595/swagger`
4. **Check logs** in console output

### Agent Development

1. **Make changes** to agent code
2. **Rebuild agent** (`dotnet build`)
3. **Restart agent** (stop and start service, or `dotnet run`)
4. **Check agent logs** in `agent/logs/` directory

### Frontend Development

1. **Make changes** to frontend code
2. **Hot reload** automatically (Next.js dev server)
3. **Check browser console** for errors
4. **Test API calls** via Network tab

---

## Common Issues and Solutions

### Issue: Backend won't start

**Error**: "Connection string not found"
- **Solution**: Check `appsettings.json` has `ConnectionStrings:DefaultConnection`

**Error**: "Port already in use"
- **Solution**: Change port in `appsettings.json` or kill process using port 9595

### Issue: Agent can't connect to backend

**Error**: "Connection refused"
- **Solution**: 
  - Verify backend is running
  - Check `BackendGrpcUrl` in agent config
  - Check firewall rules

**Error**: "Invalid API key"
- **Solution**: Re-register agent or check API key in agent config

### Issue: Logs not appearing

**Error**: Logs sent but not visible
- **Solution**:
  - Check normalization worker is running
  - Check database for logs in `log_entries` table
  - Check backend logs for errors

### Issue: Frontend not loading data

**Error**: CORS errors
- **Solution**: Add frontend URL to `Cors:AllowedOrigins` in backend config

**Error**: 401 Unauthorized
- **Solution**: Login via frontend to get JWT token

---

## Debugging Tips

### Backend Logging

Backend uses Serilog. Check console output or log files:
- Console: Real-time logs
- File: `backend/logs/` directory (if configured)

### Agent Logging

Agent logs to:
- Console: If running with `dotnet run`
- File: `agent/logs/` directory
- Windows Event Log: If installed as service

### Database Debugging

```bash
# Connect to database
psql -U postgres -d athalasiem

# Check agents
SELECT id, hostname, status, last_connected FROM agents;

# Check logs
SELECT COUNT(*) FROM log_entries;
SELECT COUNT(*) FROM normalized_logs;

# Check alerts
SELECT id, title, severity, status, created_at FROM alerts ORDER BY created_at DESC LIMIT 10;
```

### gRPC Debugging

Use gRPC tools:
```bash
# Install grpcurl
# Windows: choco install grpcurl
# Linux: apt-get install grpcurl

# List services
grpcurl -plaintext localhost:9595 list

# Call RegisterAgent
grpcurl -plaintext -d '{"hostname":"test","ip_address":"127.0.0.1","operating_system":"Windows","agent_version":"1.0","agent_type":"Windows"}' \
  localhost:9595 athala.siem.SiemService/RegisterAgent
```

---

## Performance Tuning

### Backend

- **Database Connection Pool**: Increase in `appsettings.json`
- **gRPC Message Size**: Already set to 100MB
- **Worker Threads**: Adjust in `Program.cs`

### Agent

- **Log Batch Size**: Increase `LogBatchSize` for higher throughput
- **Heartbeat Interval**: Decrease for faster status updates
- **Buffer Size**: Increase `MaxLogBufferSize` for offline resilience

---

## Production Checklist

Before deploying to production:

- [ ] Change JWT secret key
- [ ] Use HTTPS/TLS for gRPC
- [ ] Configure proper CORS origins
- [ ] Set up database backups
- [ ] Configure log rotation
- [ ] Set up monitoring/alerting
- [ ] Review security settings
- [ ] Test failover scenarios
- [ ] Document deployment process
- [ ] Set up CI/CD pipeline

---

## Next Steps

1. **Add more detection rules** - Create rules for common attack patterns
2. **Enhance correlation** - Implement attack chain detection
3. **Add threat intelligence** - Integrate external TI feeds
4. **Improve UI** - Add more visualizations and dashboards
5. **Scale testing** - Test with 100+ agents
6. **Performance optimization** - Optimize database queries and workers
