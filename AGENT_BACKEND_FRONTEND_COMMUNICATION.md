# Agent ↔ Backend ↔ Frontend Communication Architecture

## Overview

This document describes the complete communication flow between the three components of AthalaSIEM:
- **Agent** (Universal Agent - C# .NET 8)
- **Backend** (ASP.NET Core API - Port 9595)
- **Frontend** (Next.js React - Port 3000)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ATHALASIEM ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│              │         │              │         │              │
│    AGENT     │◄───────►│   BACKEND    │◄───────►│  FRONTEND   │
│              │         │              │         │              │
│  (Collector) │         │  (API Server)│         │  (Dashboard) │
│              │         │              │         │              │
└──────────────┘         └──────────────┘         └──────────────┘
     │                          │                          │
     │                          │                          │
     ▼                          ▼                          ▼
┌──────────┐              ┌──────────┐              ┌──────────┐
│  Logs    │              │ Database │              │  Browser │
│ Sources  │              │ (Postgres)│              │   (User) │
└──────────┘              └──────────┘              └──────────┘
```

---

## 1. Agent → Backend Communication

### 1.1 Protocols

The agent uses **two communication protocols**:

#### A. **REST API (HTTP/HTTPS)** - Primary
- **Base URL**: `http://localhost:9595/api`
- **Authentication**: API Key in `X-API-Key` header
- **Content-Type**: `application/json`
- **Service**: `BackendCommunicationService.cs`

#### B. **gRPC (HTTP/2)** - High-Performance (Optional)
- **Base URL**: `http://localhost:9595` (same port, different protocol)
- **Protocol**: Protocol Buffers (binary)
- **Service**: `GrpcCommunicationService.cs`
- **Status**: Proto defined, client implementation in progress

### 1.2 Agent Registration Flow

```
┌─────────┐                                    ┌─────────┐
│  Agent  │                                    │ Backend │
└────┬────┘                                    └────┬────┘
     │                                               │
     │ 1. POST /api/agentdeployment/register        │
     │    {                                          │
     │      "hostname": "REVIAN_Win32NT_DEV",       │
     │      "ipAddress": "192.168.1.100",           │
     │      "operatingSystem": "Windows 10",        │
     │      "agentVersion": "1.0.0",                │
     │      "agentType": "Windows",                 │
     │      "registrationKey": "xxx"                │
     │    }                                          │
     ├──────────────────────────────────────────────►│
     │                                               │
     │ 2. Response:                                  │
     │    {                                          │
     │      "agentId": "guid-xxx",                   │
     │      "apiKey": "api-key-xxx",                 │
     │      "success": true                          │
     │    }                                          │
     │◄──────────────────────────────────────────────┤
     │                                               │
     │ 3. Store agentId & apiKey locally            │
     │                                               │
```

**Implementation Location:**
- Agent: `agent-universal/Services/BackendCommunicationService.cs::RegisterAgentAsync()`
- Backend: `backend/Controllers/AgentsController.cs::RegisterAgent()`

### 1.3 Heartbeat Flow

```
┌─────────┐                                    ┌─────────┐
│  Agent  │                                    │ Backend │
└────┬────┘                                    └────┬────┘
     │                                               │
     │ Every 1 minute (configurable)                │
     │                                               │
     │ POST /api/agents/{agentId}/heartbeat         │
     │    {                                          │
     │      "timestamp": "2026-01-14T10:00:00Z",    │
     │      "status": "Healthy",                    │
     │      "cpuUsage": 25.5,                        │
     │      "memoryUsage": 60.2,                    │
     │      "diskUsage": 45.0,                      │
     │      "uptimeHours": 72.5,                    │
     │      "activeCollectors": 3,                  │
     │      "logsCollected": 1500,                  │
     │      "logsForwarded": 1500                   │
     │    }                                          │
     ├──────────────────────────────────────────────►│
     │                                               │
     │ Response: 200 OK                              │
     │◄──────────────────────────────────────────────┤
```

**Implementation Location:**
- Agent: `agent-universal/Services/BackendCommunicationService.cs::SendHeartbeat()`
- Backend: `backend/Controllers/AgentsController.cs::RecordHeartbeat()`

### 1.4 Log Forwarding Flow

```
┌─────────┐                                    ┌─────────┐
│  Agent  │                                    │ Backend │
└────┬────┘                                    └────┬────┘
     │                                               │
     │ Collector → Parser → Normalizer → Buffer      │
     │                                               │
     │ Every 30 seconds (configurable)             │
     │ Batch size: 100 logs (configurable)          │
     │                                               │
     │ POST /api/logs/batch                          │
     │    {                                          │
     │      "agentId": "guid-xxx",                   │
     │      "logs": [                                │
     │        {                                      │
     │          "id": "log-1",                       │
     │          "timestamp": "2026-01-14T10:00:00Z",│
     │          "source": "WindowsEventLog",        │
     │          "sourceType": "Security",            │
     │          "logLevel": "Information",           │
     │          "message": "User logged in",          │
     │          "metadata": {                        │
     │            "@timestamp": "...",                │
     │            "agent.id": "...",                  │
     │            "host.name": "...",                │
     │            "event.category": "authentication" │
     │          }                                    │
     │        },                                     │
     │        ...                                    │
     │      ]                                        │
     │    }                                          │
     ├──────────────────────────────────────────────►│
     │                                               │
     │ Response:                                     │
     │    {                                          │
     │      "success": true,                         │
     │      "acceptedCount": 100,                    │
     │      "rejectedCount": 0                       │
     │    }                                          │
     │◄──────────────────────────────────────────────┤
```

**Implementation Location:**
- Agent: `agent-universal/Services/BackendCommunicationService.cs::ProcessLogBatch()`
- Backend: `backend/Controllers/LogsController.cs::BatchCreate()`

**Key Features:**
- **Batching**: Logs are queued and sent in batches (default: 100 logs, every 30 seconds)
- **Queue Management**: Max queue size 50,000 logs (configurable)
- **Backpressure**: If queue is full, oldest logs are removed (25% by default)
- **Archival**: Failed logs are archived to disk for later retry
- **Retry Logic**: Automatic retry on network failures

### 1.5 Configuration Fetch Flow

```
┌─────────┐                                    ┌─────────┐
│  Agent  │                                    │ Backend │
└────┬────┘                                    └────┬────┘
     │                                               │
     │ Every 30 minutes (configurable)             │
     │                                               │
     │ GET /api/agents/{agentId}/configuration       │
     │    Headers: X-API-Key: {apiKey}               │
     ├──────────────────────────────────────────────►│
     │                                               │
     │ Response:                                     │
     │    {                                          │
     │      "agentId": "guid-xxx",                   │
     │      "collectors": [...],                     │
     │      "eventLogsToMonitor": [...],             │
     │      "fimConfiguration": {...},               │
     │      "version": "v2"                          │
     │    }                                          │
     │◄──────────────────────────────────────────────┤
     │                                               │
     │ If version changed → Update local config     │
     │                                               │
```

**Implementation Location:**
- Agent: `agent-universal/Services/BackendCommunicationService.cs::FetchBackendConfiguration()`
- Backend: `backend/Controllers/AgentsController.cs::GetAgentConfiguration()`

---

## 2. Backend → Frontend Communication

### 2.1 Protocol

- **REST API (HTTP/HTTPS)**
- **Base URL**: `http://localhost:9595/api`
- **Authentication**: JWT Bearer Token
- **Content-Type**: `application/json`

### 2.2 Key Endpoints

#### Agent Management
```
GET    /api/agents              - List all agents
GET    /api/agents/{id}         - Get agent details
PUT    /api/agents/{id}          - Update agent
DELETE /api/agents/{id}          - Delete agent (Admin only)
GET    /api/agents/{id}/health   - Get agent health metrics
```

#### Logs
```
GET    /api/logs                - List logs (with pagination, filters)
GET    /api/logs/{id}           - Get log details
POST   /api/logs/batch          - Batch create logs (Agent only)
```

#### Alerts
```
GET    /api/alerts              - List alerts
GET    /api/alerts/{id}         - Get alert details
PUT    /api/alerts/{id}/acknowledge - Acknowledge alert
```

#### Dashboard
```
GET    /api/dashboard/stats     - Get dashboard statistics
GET    /api/dashboard/health    - Get system health overview
```

**Implementation Location:**
- Frontend: `frontend/src/lib/api.ts` - API helper with token refresh
- Backend: Various controllers in `backend/Controllers/`

---

## 3. Agent Internal Architecture (Pipeline)

Based on your specification, the agent follows a **pipeline-based architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│                    AGENT PIPELINE STAGES                      │
└─────────────────────────────────────────────────────────────┘

┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│          │    │          │    │          │    │          │    │          │
│Collector │───►│  Parser  │───►│Normalizer│───►│  Buffer  │───►│ Exporter │
│          │    │          │    │          │    │          │    │          │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
     │                │                │                │                │
     │                │                │                │                │
     ▼                ▼                ▼                ▼                ▼
RawEvent      ParsedEvent      NormalizedEvent    Queue         Backend/File
```

### 3.1 Pipeline Stage Responsibilities

#### **Collector** (`agent-universal/Collectors/`)
- **Responsibility**: Acquire raw telemetry
- **Output**: `RawEvent` (raw log data)
- **Must NOT**: Parse, Normalize, Detect
- **Examples**:
  - `WindowsEventLogCollector.cs` - Windows Event Log
  - `JournalctlCollector.cs` - Linux systemd journal
  - `SyslogServerCollector.cs` - Network syslog

#### **Parser** (`agent-universal/Core/`)
- **Responsibility**: Decode and structure raw logs
- **Output**: `ParsedEvent` (structured data)
- **Must NOT**: Normalize schema, Apply detection

#### **Normalizer** (`agent-universal/Core/`)
- **Responsibility**: Map to Athala ECS-lite schema
- **Output**: `NormalizedEvent` (standardized format)
- **Must Preserve**: `raw_event` field

#### **Buffer** (`agent-universal/Services/BackendCommunicationService.cs`)
- **Responsibility**: Reliability and backpressure
- **Strategies**: Memory queue, Disk fallback
- **Features**:
  - Max queue size: 50,000 logs
  - Automatic archival on failure
  - Retry logic

#### **Exporter** (`agent-universal/Services/BackendCommunicationService.cs`)
- **Responsibility**: Deliver events
- **Modes**: HTTP (REST), gRPC, File, Console

---

## 4. Normalization Schema (Athala ECS-lite)

All events are normalized to this schema before sending to backend:

```json
{
  "@timestamp": "2026-01-14T10:00:00.000Z",
  "agent": {
    "id": "agent-guid",
    "name": "REVIAN_Win32NT_DEV"
  },
  "host": {
    "name": "REVIAN-WIN10",
    "os": {
      "name": "Windows",
      "version": "10.0.19042"
    }
  },
  "event": {
    "category": "authentication",
    "action": "user_login",
    "outcome": "success"
  },
  "log": {
    "level": "Information"
  },
  "user": {
    "name": "Administrator",
    "id": "S-1-5-21-..."
  },
  "process": {
    "name": "winlogon.exe",
    "pid": 1234,
    "command_line": "C:\\Windows\\System32\\winlogon.exe"
  },
  "network": {
    "source": {
      "ip": "192.168.1.100",
      "port": 3389
    },
    "destination": {
      "ip": "192.168.1.50",
      "port": 3389
    },
    "protocol": "RDP"
  },
  "athala": {
    "raw_event": "{...original raw log...}",
    "collector": "WindowsEventLogCollector",
    "source_type": "Security",
    "pipeline_stage": "normalized"
  }
}
```

---

## 5. Configuration Files

### Agent Configuration (`agent-universal/appsettings.json`)

```json
{
  "AgentSettings": {
    "BackendUrl": "http://localhost:9595",
    "AgentId": "",
    "ApiKey": "",
    "RegistrationKey": "your-registration-key"
  },
  "Communication": {
    "MaxQueueSize": 50000,
    "BatchSize": 100,
    "BatchIntervalSeconds": 30,
    "HeartbeatIntervalMinutes": 1,
    "ConfigUpdateIntervalMinutes": 30,
    "RetentionDays": 90,
    "ArchiveDirectory": "LogArchive"
  },
  "GrpcCommunication": {
    "ServerUrl": "http://localhost:9595",
    "BatchSize": 100,
    "BatchIntervalSeconds": 30,
    "MaxQueueSize": 10000
  }
}
```

### Backend Configuration (`backend/appsettings.json`)

```json
{
  "ConnectionStrings": {
    "DefaultConnection": "Host=localhost;Database=athalasiem;..."
  },
  "JwtSettings": {
    "SecretKey": "...",
    "Issuer": "AthalaSIEM",
    "Audience": "AthalaSIEM",
    "ExpirationMinutes": 60
  },
  "AgentRegistration": {
    "RegistrationKey": "your-registration-key"
  }
}
```

---

## 6. Error Handling & Resilience

### Agent Resilience Features

1. **Queue Management**
   - Max queue size prevents memory exhaustion
   - Automatic oldest log removal when queue is full
   - Configurable removal percentage

2. **Archival**
   - Failed logs are saved to disk (`LogArchive/`)
   - Automatic retry on next batch cycle
   - Retention policy (default: 90 days)

3. **Connection Retry**
   - Automatic retry on network failures
   - Exponential backoff
   - Health check before sending

4. **Graceful Degradation**
   - Agent continues collecting even if backend is down
   - Logs are queued and sent when connection restored
   - Local file export as fallback

### Backend Error Handling

1. **Validation**
   - Input validation on all endpoints
   - API key validation for agent requests
   - JWT token validation for frontend requests

2. **Database Transactions**
   - Atomic operations for batch inserts
   - Rollback on errors
   - Foreign key constraint handling

---

## 7. Security

### Agent → Backend
- **API Key Authentication**: `X-API-Key` header
- **Registration Key**: Required for initial registration
- **TLS/SSL**: Supported (configure in `appsettings.json`)

### Frontend → Backend
- **JWT Bearer Token**: Standard OAuth 2.0 flow
- **Token Refresh**: Automatic refresh on expiration
- **Role-Based Access Control**: Admin/User roles

---

## 8. Testing Strategy

### Agent Testing (Per Your Specification)

1. **Unit Tests**
   - Parser tests
   - Normalizer tests
   - Buffer tests

2. **Pipeline Tests**
   - Collector → File exporter (no backend required)
   - End-to-end pipeline validation

3. **Replay Tests**
   - Recorded logs replay
   - Golden file comparison

4. **Failure Tests**
   - Backend unreachable
   - Network timeout
   - Disk full
   - Burst traffic

**Golden Rule**: Agent MUST run without backend (test mode with file export)

---

## 9. Deployment Modes

### Agent Deployment

#### Windows
- **Mode**: Windows Service
- **Installation**: MSI installer
- **Service Name**: `AthalaSIEM Universal Agent`

#### Linux
- **Mode**: systemd daemon
- **Service File**: `athalasiem-agent.service`
- **Installation**: `systemctl enable athalasiem-agent`

#### Container
- **Mode**: Docker container
- **Image**: `athalasiem/agent:latest`
- **Deployment**: Docker Compose or Kubernetes

---

## 10. Monitoring & Observability

### Agent Metrics
- Logs collected per second
- Logs forwarded per second
- Queue size
- Connection status
- Collector status
- System resource usage (CPU, Memory, Disk)

### Backend Metrics
- Agents connected
- Logs received per second
- Alerts generated
- API response times
- Database query performance

### Frontend Metrics
- Page load times
- API call success rates
- User activity

---

## 11. Next Steps for Agent Implementation

Based on your specification, here's what needs to be implemented:

###  Already Implemented
- [x] REST API communication (`BackendCommunicationService`)
- [x] Agent registration
- [x] Heartbeat mechanism
- [x] Log batching and queue management
- [x] Configuration fetching
- [x] Windows Event Log collector
- [x] Basic pipeline structure

### 🔄 In Progress
- [ ] gRPC client implementation (proto defined, client needs completion)
- [ ] Complete normalization to Athala ECS-lite schema
- [ ] Additional collectors (Journalctl, Syslog, Docker)

###  To Do
- [ ] Parser implementation for all log types
- [ ] Normalizer with full ECS-lite mapping
- [ ] Test mode with file export
- [ ] Replay runner for recorded logs
- [ ] Failure scenario testing
- [ ] Docker deployment files
- [ ] systemd service file
- [ ] Windows Service wrapper

---

## 12. Quick Reference

### Agent Startup Sequence

1. Load configuration from `appsettings.json`
2. Initialize collectors
3. Register with backend (`POST /api/agentdeployment/register`)
4. Store `agentId` and `apiKey`
5. Start heartbeat timer (every 1 minute)
6. Start batch timer (every 30 seconds)
7. Start config update timer (every 30 minutes)
8. Begin collecting logs

### Backend Startup Sequence

1. Load configuration
2. Initialize database connection
3. Start gRPC server (if enabled)
4. Start HTTP REST API server (port 9595)
5. Initialize services (AgentService, LogService, etc.)
6. Ready to accept connections

### Frontend Startup Sequence

1. Load React application
2. Check for stored JWT token
3. If token exists, validate with backend
4. If valid, redirect to dashboard
5. If invalid/expired, redirect to login
6. Fetch initial data (agents, logs, alerts)

---

## Summary

The communication flow is:
1. **Agent** collects logs → normalizes → batches → sends to **Backend** via REST/gRPC
2. **Backend** stores logs → processes → generates alerts → serves to **Frontend** via REST API
3. **Frontend** displays data → user interacts → sends commands to **Backend** → **Backend** updates **Agent** config

All communication is **asynchronous**, **batched**, and **resilient** to failures.
