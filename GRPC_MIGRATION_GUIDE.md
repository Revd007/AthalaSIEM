# AthalaSIEM gRPC Migration Guide

## Overview

This document describes the migration of AthalaSIEM agent-backend communication from HTTP to gRPC as the primary data plane, while maintaining HTTP as a control plane fallback.

## Architecture

### Communication Model

```
┌─────────────┐                    ┌─────────────┐
│   Agent     │                    │   Backend   │
└──────┬──────┘                    └──────┬──────┘
       │                                   │
       │  Control Plane (HTTP)             │
       │  - Registration                   │
       │  - Configuration                  │
       │  - Auto-deployment                │
       ├───────────────────────────────────>│
       │                                   │
       │  Data Plane (gRPC Primary)       │
       │  - Log Streaming                 │
       │  - Heartbeat Streaming           │
       │  - System Metrics                │
       ├═══════════════════════════════════>│
       │                                   │
       │  Fallback (HTTP)                 │
       │  - If gRPC unavailable           │
       └───────────────────────────────────>│
```

### Protocol Selection

1. **Control Plane**: Always HTTP
   - Agent registration
   - Configuration retrieval
   - Auto-deployment token fetch

2. **Data Plane**: gRPC (primary), HTTP (fallback)
   - Log forwarding (streaming)
   - Heartbeat (bidirectional streaming)
   - System metrics (streaming)

## Implementation Status

###  Completed

1. **Proto Definitions**
   - Updated `backend/Protos/siem.proto` with streaming RPCs
   - Updated `agent-universal/Protos/siem.proto` (synced)
   - Added `grpc_endpoint` to `RegisterAgentResponse`

2. **Backend gRPC Service**
   - Implemented `StreamLogs` - client streaming
   - Implemented `StreamHeartbeat` - bidirectional streaming
   - Implemented `StreamSystemMetrics` - client streaming
   - Authentication via gRPC metadata (`x-api-key`, `x-agent-id`)

3. **Agent gRPC Client**
   - Complete `GrpcCommunicationService` implementation
   - Streaming support for logs and heartbeats
   - Automatic fallback detection
   - Connection health monitoring

### 🔄 In Progress

1. **Hybrid Communication Service**
   - gRPC primary, HTTP fallback
   - Automatic protocol switching
   - Configuration-based selection

2. **Bootstrap Flow**
   - HTTP registration → gRPC endpoint discovery
   - Seamless transition to gRPC

###  Pending

1. **mTLS Support**
   - Certificate-based authentication
   - Secure channel configuration

2. **Testing Strategy**
   - Unit tests for streaming
   - Integration tests
   - Performance benchmarks

## Configuration

### Backend (`appsettings.json`)

```json
{
  "GrpcServer": {
    "Url": "http://0.0.0.0:9595"
  },
  "Kestrel": {
    "Endpoints": {
      "Http": {
        "Url": "http://0.0.0.0:9595"
      }
    }
  }
}
```

### Agent (`appsettings.json`)

```json
{
  "SiemManager": {
    "ManagerIP": "192.168.1.17",
    "ManagerPort": 9595,
    "GrpcPort": 9595,
    "UseHTTPS": false
  },
  "Agent": {
    "Id": "auto-generated",
    "ApiKey": "from-registration",
    "BatchSize": 100,
    "BatchIntervalSeconds": 30,
    "HeartbeatIntervalSeconds": 30
  },
  "GrpcCommunication": {
    "MaxQueueSize": 10000,
    "EnableStreaming": true,
    "FallbackToHttp": true
  }
}
```

## Usage

### Agent Initialization

```csharp
// In Program.cs
services.AddSingleton<GrpcCommunicationService>();
services.AddSingleton<BackendCommunicationService>(); // HTTP fallback

// Initialize gRPC first
var grpcService = serviceProvider.GetRequiredService<GrpcCommunicationService>();
if (await grpcService.InitializeAsync())
{
    // Use gRPC for data plane
}
else
{
    // Fallback to HTTP
    var httpService = serviceProvider.GetRequiredService<BackendCommunicationService>();
    await httpService.InitializeAsync();
}
```

### Backend Startup

gRPC is automatically configured in `Program.cs`:

```csharp
app.MapGrpcService<Backend.Services.SiemService>();
```

## Authentication

### API Key (Current)

- **gRPC Metadata**: `x-api-key`, `x-agent-id`
- **HTTP Headers**: `X-API-Key`, `X-Agent-Id`

### mTLS (Future)

- Client certificate validation
- Certificate-based agent identity

## Performance Benefits

1. **Streaming**: Reduced latency for high-volume log forwarding
2. **Binary Protocol**: Smaller payload sizes (Protocol Buffers)
3. **HTTP/2**: Multiplexing, header compression
4. **Connection Reuse**: Persistent connections

## Deployment

### Prerequisites

1. .NET 8.0 runtime
2. gRPC packages installed
3. Proto files generated

### Build

```bash
# Backend
cd backend
dotnet build

# Agent
cd agent-universal
dotnet build
```

### Testing

```bash
# Test gRPC connection
grpcurl -plaintext localhost:9595 list

# Test streaming
# (Use agent with gRPC enabled)
```

## Troubleshooting

### Common Issues

1. **gRPC Connection Failed**
   - Check firewall rules
   - Verify `GrpcServer:Url` configuration
   - Check HTTP/2 support

2. **Streaming Not Working**
   - Verify proto files are generated
   - Check gRPC package versions match
   - Review logs for authentication errors

3. **Fallback to HTTP**
   - Normal behavior if gRPC unavailable
   - Check agent logs for fallback reason

## Next Steps

1. Implement hybrid communication service
2. Add mTLS support
3. Performance testing and optimization
4. Documentation updates
5. Production deployment guide
