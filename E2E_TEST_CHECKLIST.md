# AthalaSIEM End-to-End Test Checklist

## Prerequisites
- [ ] PostgreSQL database running and accessible
- [ ] Backend API running on `http://localhost:9595`
- [ ] Frontend running on `http://localhost:3000` (or configured port)
- [ ] Agent installer available or agent source code compiled

## Scenario 1: Agent Registration

### Steps:
1. [ ] Generate deployment token in backend (via API or UI)
2. [ ] Install agent with token (or run agent with registration)
3. [ ] Verify agent appears in `/api/agents` endpoint
4. [ ] Verify agent appears in frontend agent list
5. [ ] Check agent status is "Online" or "Pending"

### Expected Results:
- Agent ID is generated
- API key is returned
- Agent record exists in database
- Frontend displays agent with correct hostname/IP

### Verification Commands:
```bash
# Check agent via API
curl http://localhost:9595/api/agents

# Check database
psql -d athalasiem -c "SELECT id, hostname, status FROM agents;"
```

---

## Scenario 2: Heartbeat Monitoring

### Steps:
1. [ ] Wait 30-60 seconds after agent registration
2. [ ] Check agent heartbeat endpoint logs
3. [ ] Verify agent status updated in database
4. [ ] Verify frontend shows "Last Heartbeat" timestamp
5. [ ] Verify agent status is "Online"

### Expected Results:
- Heartbeat received every 30 seconds (configurable)
- Agent `LastConnected` timestamp updated
- Frontend updates in real-time (polling every 10s)
- Agent health metrics visible

### Verification:
```bash
# Check backend logs for heartbeat
# Look for: "Heartbeat received from agent {AgentId}"

# Check database
psql -d athalasiem -c "SELECT id, hostname, last_connected, status FROM agents;"
```

---

## Scenario 3: Log Ingestion

### Steps:
1. [ ] Agent collects Windows Event Log (Security, System, Application)
2. [ ] Agent sends batched logs via gRPC
3. [ ] Verify logs stored in `log_entries` table
4. [ ] Verify logs normalized in `normalized_logs` table
5. [ ] Verify logs visible in frontend

### Expected Results:
- Logs received via `ForwardLogs` gRPC call
- Logs stored with correct agent_id
- Normalization worker processes logs
- ECS fields populated in normalized_logs
- Frontend displays logs in log viewer

### Verification:
```bash
# Check logs in database
psql -d athalasiem -c "SELECT COUNT(*) FROM log_entries WHERE agent_id = '<agent_id>';"
psql -d athalasiem -c "SELECT COUNT(*) FROM normalized_logs WHERE agent_id = '<agent_id>';"

# Check via API
curl http://localhost:9595/api/logs?agentId=<agent_id>
```

---

## Scenario 4: Basic Detection

### Steps:
1. [ ] Create detection rule (e.g., "Multiple Failed Logins")
2. [ ] Trigger rule condition (generate matching logs)
3. [ ] Verify detection engine processes logs
4. [ ] Verify alert created in `alerts` table
5. [ ] Verify alert visible in frontend

### Expected Results:
- Detection rule loaded and active
- Logs matched against rules
- Alert created with correct severity
- Alert deduplication working
- Frontend shows alert in alerts table

### Verification:
```bash
# Check alerts
psql -d athalasiem -c "SELECT id, title, severity, status FROM alerts ORDER BY created_at DESC LIMIT 10;"

# Check via API
curl http://localhost:9595/api/alerts?recent=true
```

---

## Scenario 5: Agent Offline Detection

### Steps:
1. [ ] Stop agent service/process
2. [ ] Wait 2-3 minutes (offline threshold)
3. [ ] Verify backend marks agent as offline
4. [ ] Verify frontend updates agent status
5. [ ] Verify alert created for offline agent (if configured)

### Expected Results:
- Agent status changes to "Offline"
- Frontend shows agent as offline (red indicator)
- Last heartbeat timestamp stops updating
- Optional: Alert created for offline agent

### Verification:
```bash
# Check agent status
psql -d athalasiem -c "SELECT id, hostname, status, last_connected FROM agents WHERE id = '<agent_id>';"

# Check for offline alerts
psql -d athalasiem -c "SELECT * FROM alerts WHERE title LIKE '%offline%' OR title LIKE '%agent%';"
```

---

## Scenario 6: End-to-End Flow (Complete Pipeline)

### Steps:
1. [ ] Agent registers successfully
2. [ ] Agent sends heartbeat
3. [ ] Agent collects and sends logs
4. [ ] Logs normalized and stored
5. [ ] Detection rules run on normalized logs
6. [ ] Alert generated from detection
7. [ ] All data visible in frontend

### Expected Results:
- Complete pipeline works: Ingest → Normalize → Detect → Alert
- All components integrated
- Frontend displays all data correctly
- No errors in backend logs

---

## Performance Checks

### Log Throughput:
- [ ] Agent can send 1000+ logs/minute
- [ ] Backend processes logs without backlog
- [ ] Normalization keeps up with ingestion
- [ ] Detection engine processes in real-time

### System Resources:
- [ ] Agent CPU usage < 5%
- [ ] Agent memory usage < 200MB
- [ ] Backend handles concurrent agents
- [ ] Database queries performant

---

## Error Handling

### Network Failures:
- [ ] Agent retries on connection failure
- [ ] Agent buffers logs when offline
- [ ] Agent resumes when connection restored

### Invalid Data:
- [ ] Backend rejects invalid API keys
- [ ] Backend handles malformed logs gracefully
- [ ] Errors logged but don't crash system

---

## Security Checks

### Authentication:
- [ ] Agent API key validation works
- [ ] Invalid API keys rejected
- [ ] gRPC calls authenticated

### Data Integrity:
- [ ] Logs not tampered with
- [ ] Agent identity verified
- [ ] Audit trail maintained

---

## Frontend Integration

### Agent List Page:
- [ ] Shows all registered agents
- [ ] Displays online/offline status
- [ ] Shows last heartbeat time
- [ ] Shows log volume per agent

### Log Viewer:
- [ ] Displays logs from agents
- [ ] Filterable by agent, time, severity
- [ ] Real-time updates (polling)

### Alerts Page:
- [ ] Shows recent alerts
- [ ] Filterable by severity, status
- [ ] Alert details viewable
- [ ] Alert acknowledgment works

### Dashboard:
- [ ] System health indicators
- [ ] Agent count and status
- [ ] Log volume metrics
- [ ] Alert statistics

---

## Success Criteria

✅ **All scenarios pass**
✅ **No critical errors in logs**
✅ **Frontend displays all data correctly**
✅ **System handles 5+ agents simultaneously**
✅ **Complete pipeline works end-to-end**

---

## Quick Test Script

```bash
#!/bin/bash
# Quick E2E test script

echo "Testing Agent Registration..."
AGENT_ID=$(curl -s -X POST http://localhost:9595/api/agents/register \
  -H "Content-Type: application/json" \
  -d '{"hostname":"test-agent","ipAddress":"127.0.0.1","operatingSystem":"Windows"}' \
  | jq -r '.agentId')

echo "Agent ID: $AGENT_ID"

echo "Waiting for heartbeat..."
sleep 35

echo "Checking agent status..."
curl -s http://localhost:9595/api/agents/$AGENT_ID | jq

echo "Checking logs..."
curl -s "http://localhost:9595/api/logs?agentId=$AGENT_ID" | jq '.items | length'

echo "Checking alerts..."
curl -s "http://localhost:9595/api/alerts?recent=true" | jq '.items | length'
```

---

## Troubleshooting

### Agent Not Registering:
- Check backend is running
- Check gRPC port (50051) is accessible
- Check deployment token is valid
- Check agent logs for errors

### Logs Not Appearing:
- Check agent is collecting logs
- Check gRPC connection
- Check backend logs for errors
- Check normalization worker is running

### Alerts Not Generating:
- Check detection rules are enabled
- Check rule conditions match logs
- Check detection worker is running
- Check alert repository

### Frontend Not Updating:
- Check API endpoints are accessible
- Check CORS configuration
- Check frontend polling interval
- Check browser console for errors
