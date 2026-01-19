# AthalaSIEM Communication Protocol Summary

## Current Implementation

### Agent → Backend Communication

**Protocol: HTTP (REST API)**
- The agent uses `BackendCommunicationService` which implements HTTP-based communication
- All communication is done via `HttpClient` with JSON payloads
- Endpoints used:
  - `POST /api/deployment/register` - Agent registration
  - `POST /api/agents/{id}/heartbeat` - Heartbeat
  - `POST /api/logs/batch` - Log forwarding
  - `GET /api/configuration/agent/{id}` - Configuration fetch

### gRPC Status

**Backend:**
-  gRPC services are **defined** in `backend/Protos/siem.proto`
-  gRPC server is **configured** in `backend/Program.cs` (line 106-119, 488-489)
-  `SiemService` implementation exists in `backend/Services/SiemService.cs`
-  gRPC endpoint is mapped: `app.MapGrpcService<Backend.Services.SiemService>()`

**Agent:**
- `GrpcCommunicationService` class exists but is **NOT actively used**
- Registered in DI (`Program.cs` line 96) but not injected/used by main services
- gRPC proto file exists but client code is commented out

### Why HTTP Instead of gRPC?

Currently, the agent uses HTTP because:
1. `BackendCommunicationService` is the primary communication service
2. `GrpcCommunicationService` exists but is not integrated into the main pipeline
3. HTTP is simpler for development and debugging
4. All existing endpoints are REST-based

### gRPC Use Cases (If Enabled)

If gRPC were to be used, it would provide:
- **Higher performance** for high-volume log forwarding
- **Streaming support** for real-time log transmission
- **Binary protocol** (Protocol Buffers) for smaller payloads
- **Bidirectional communication** for configuration updates

### Current Issues

1. **Heartbeat Validation Error**: 
   - Error: `"The heartbeat field is required"` and `"The JSON value could not be converted to Backend.Models.AgentStatus"`
   - Cause: JSON serialization mismatch between agent and backend
   - Fix: Ensure camelCase property names and correct enum serialization

2. **FIM Configuration Unauthorized**:
   - Error: `"Failed to fetch FIM configurations: Unauthorized"`
   - Cause: API key not being sent or invalid
   - Fix: Ensure agent registration completes and API key is stored/used

## Recommendation

**For Production:**
- Keep HTTP for now (it's working and simpler)
- Consider migrating to gRPC for high-volume scenarios
- If migrating, complete the `GrpcCommunicationService` implementation

**For Development:**
- Fix the HTTP heartbeat payload format (current priority)
- Ensure API key authentication works correctly
- Test both protocols if needed
