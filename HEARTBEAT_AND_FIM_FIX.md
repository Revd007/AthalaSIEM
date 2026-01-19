# Fix: Heartbeat and FIM Configuration Errors

## Issues Fixed

### 1. Heartbeat Validation Error
**Error:**
```
Heartbeat failed: BadRequest - {"message":"Validation failed","errors":["The heartbeat field is required.","The JSON value could not be converted to Backend.Models.AgentStatus. Path: $.Status | LineNumber: 0 | BytePositionInLine: 111."]}
```

**Root Cause:**
- The agent was sending `Status` as an integer (`3`) instead of the enum string name (`"Online"`)
- ASP.NET Core's model binding expects enum values to be sent as string names, not integers

**Fix Applied:**
- Changed `status = 3` to `Status = "Online"` in `BackendCommunicationService.cs`
- Updated property names to PascalCase (they'll be serialized to camelCase by JsonSerializerOptions)
- The backend's `AgentStatus` enum has `Online = 3`, but the JSON deserializer expects the string name

**File Changed:**
- `agent-universal/Services/BackendCommunicationService.cs` (line ~448)

### 2. FIM Configuration Unauthorized Error
**Error:**
```
Failed to fetch FIM configurations: Unauthorized - Unauthorized
```

**Root Cause:**
- `FIMConfigurationService` was reading the API key from configuration in the constructor
- The API key is set dynamically after agent registration, but the service was using a stale value
- The service stored the API key as a `readonly` field, so it never updated after registration

**Fix Applied:**
- Removed `readonly` fields for `_apiKey`, `_agentId`, and `_backendUrl`
- Created helper methods: `GetApiKey()`, `GetAgentId()`, `GetBackendUrl()` that read from configuration on each call
- Created `CreateAuthenticatedRequest()` method that adds the API key header dynamically
- Updated all HTTP requests to use `CreateAuthenticatedRequest()` instead of relying on default headers

**Files Changed:**
- `agent-universal/Services/FIMConfigurationService.cs`

## Testing

1. **Rebuild the agent:**
   ```powershell
   cd agent-universal
   dotnet build
   ```

2. **Restart the agent service:**
   ```powershell
   # Stop the service
   Stop-Service "AthalaSIEM Universal Agent"
   
   # Reinstall/restart (or use deploy script)
   .\deploy-agent.ps1 -BackendUrl "http://192.168.1.17:9595" -AgentName "Revian" -StartService
   ```

3. **Verify in logs:**
   - Check for successful heartbeat messages (no more "BadRequest" errors)
   - Check for successful FIM configuration fetch (no more "Unauthorized" errors)

## Expected Behavior After Fix

### Heartbeat
- Heartbeat should succeed with `200 OK` response
- Status should be correctly parsed as `AgentStatus.Online`
- Agent status should update in the backend database

### FIM Configuration
- FIM configuration fetch should succeed after agent registration
- API key should be read dynamically from configuration on each request
- No more "Unauthorized" errors when fetching FIM configurations

## Technical Details

### Heartbeat Payload Format
**Before (Incorrect):**
```json
{
  "timestamp": "2026-01-19T14:48:26Z",
  "status": 3,  // ❌ Integer - causes deserialization error
  "cpuUsage": 15.5,
  "memoryUsage": 45.2,
  "diskUsage": 60.0,
  "ipAddress": "192.168.1.100",
  "additionalInfo": "{...}"
}
```

**After (Correct):**
```json
{
  "timestamp": "2026-01-19T14:48:26Z",
  "status": "Online",  // ✅ String enum name - correctly deserialized
  "cpuUsage": 15.5,
  "memoryUsage": 45.2,
  "diskUsage": 60.0,
  "ipAddress": "192.168.1.100",
  "additionalInfo": "{...}"
}
```

### FIM Configuration Service Flow
1. Agent registers with backend → receives API key
2. API key is stored in configuration (`Agent:ApiKey`)
3. `FIMConfigurationService` reads API key dynamically on each request
4. API key is added to request headers via `CreateAuthenticatedRequest()`
5. Backend validates API key and returns FIM configurations

## Notes

- The agent uses HTTP-based communication for heartbeats and FIM configuration (not gRPC)
- The gRPC migration is ongoing, but these HTTP endpoints are still used as fallback
- The API key is stored in configuration after registration and should persist across agent restarts
