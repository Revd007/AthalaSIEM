# UAT (User Acceptance Testing) for AthalaSIEM Universal Agent

## Overview

This UAT framework provides comprehensive testing capabilities for the AthalaSIEM Universal Agent without requiring a live backend server. It's designed to test all major components in an isolated, safe environment.

## Features

- **File Integrity Monitoring (FIM) Testing**: Tests file creation, modification, and deletion detection
- **Event Log Collection Testing**: Tests Windows Event Log collection capabilities
- **Communication Testing**: Tests communication layer and queue functionality
- **Performance Testing**: Tests memory usage, CPU performance, and log processing speed
- **Automated Reporting**: Generates detailed HTML and JSON reports

## UAT Communication Test Behavior

### Understanding Communication Test Results

The Communication Test in UAT environment is designed to test the agent's behavior when **no backend server is running**. This is intentional and expected behavior:

**Expected Results:**
- `Connected=False` - No backend server is running (expected in UAT)
- `QueuedLogs > 0` - Logs are queued locally (queue functionality works)
- `TotalLogsSent=0` - No logs sent because backend is offline (expected)
- `Connection test failed` - Expected because no backend is running

**What the Test Validates:**
1.  **Queue Functionality**: Agent can queue logs even when backend is offline
2.  **Graceful Degradation**: Agent continues working without backend
3.  **Error Handling**: Agent properly handles connection failures
4.  **Resource Management**: Queued logs don't cause memory issues

### Why This Design?

1. **Safety**: UAT runs in isolated environment without external dependencies
2. **Reliability**: Tests agent behavior under real-world conditions (network failures)
3. **Predictability**: Results are consistent regardless of network environment
4. **Security**: No data leaves the local machine during testing

**This is NOT a failure** - it's the expected and correct behavior for UAT environment.

## Quick Start

### Prerequisites

- Windows 10/11 or Windows Server
- .NET 8.0 Runtime
- Administrator privileges (recommended for full functionality)

### Running UAT

Choose one of these methods:

#### Method 1: Batch Script (Recommended)
```batch
run-uat.bat
```

#### Method 2: PowerShell Script
```powershell
.\run-uat.ps1
```

#### Method 3: Direct Command
```bash
dotnet run --configuration Debug -- --run-uat
```

#### Method 4: Manual FIM Testing
```batch
test-fim-quick.bat
```

### Test Configuration

UAT uses `appsettings.uat.json` for safe testing configuration:

```json
{
  "SiemManager": {
    "ManagerIP": "192.0.2.1",
    "ManagerPort": 9595,
    "UseHTTPS": false,
    "_Comment": "UAT Configuration - Non-existent backend IP (192.0.2.1 reserved for testing) ensures offline mode"
  },
  "Agent": {
    "ManagerUrl": "http://192.0.2.1:9595",
    "RequireAdminPrivileges": false,
    "BatchSize": 10,
    "BatchIntervalSeconds": 10
  }
}
```

**Note**: UAT now uses `192.0.2.1` (reserved testing IP) to ensure no accidental backend connections.

### Test Scenarios

#### 1. FIM_Testing
- Tests File Integrity Monitoring
- Creates/modifies files in safe test directories
- Validates file change detection
- Expected: PASS

#### 2. Event_Collection
- Tests Windows Event Log collection
- Collects from Application log only (safe)
- Validates event parsing and filtering
- Expected: PASS

#### 3. Communication_Test
- Tests communication layer functionality
- Validates queue operations without backend
- Tests graceful degradation
- Expected: PASS (even with `Connected=False`)

#### 4. Performance_Test
- Tests memory usage and CPU performance
- Validates log processing speed
- Tests garbage collection behavior
- Expected: PASS

## Test Directory Structure

```
agent-universal/
├── UAT-Test/
│   ├── TestFiles/          # FIM test files
│   └── Documents/          # Additional test files
├── UAT-Reports/            # Generated test reports
├── UAT-Logs/              # UAT-specific logs
└── Temp/
    └── AthalaSIEM-UAT/    # Temporary test files
```

## Safety Features

- **No System Directory Monitoring**: Only test directories are monitored
- **Limited Event Sources**: Only Application log (no Security log)
- **Local Testing Only**: No external network connections
- **Automatic Cleanup**: Test files are cleaned up after testing
- **Non-Admin Mode**: Can run without administrator privileges

## Understanding Test Results

### Successful UAT Results

```json
{
  "OverallStatus": "PASSED",
  "TotalTests": 4,
  "PassedTests": 4,
  "FailedTests": 0
}
```

### Communication Test - Expected Behavior

```json
{
  "TestName": "Communication_Test",
  "Passed": true,
  "Steps": [
    "Communication Health: Connected=False, QueuedLogs=0",
    "Queued 5 test log entries",
    "Connection test failed as expected (no backend running in UAT)",
    "UAT Communication test passed - agent correctly handles offline backend"
  ]
}
```

**This is CORRECT behavior** - the agent is working properly!

## Report Generation

### HTML Report
- Detailed visual report with test steps
- Located in `UAT-Reports/UAT-Report-{timestamp}.html`
- Open in web browser for full details

### JSON Report  
- Machine-readable test results
- Located in `UAT-Reports/UAT-Report-{timestamp}.json`
- Suitable for CI/CD integration

## Troubleshooting

### Recent Fix Applied (July 2025)

#### Issue: UAT Showing Connected=True
- **Problem**: UAT was showing `Connected=True` indicating a backend was running
- **Root Cause**: UAT configuration used `127.0.0.1:9595` which might connect to local services
- **Solution**: Changed to `192.0.2.1:9595` (reserved non-routable IP) to ensure offline mode
- **Result**: UAT now correctly tests offline behavior without accidental connections

### Common Issues

#### "Connected=True" During UAT (Now Fixed)
- **Status**: This should no longer happen with the fix
- **Expected**: `Connected=False` in UAT environment
- **If still occurring**: Check that no backend is running on port 9595
- **Action**: Stop any backend services during UAT

#### "Performance Test failed"
- **Cause**: LogProcessor not initialized
- **Solution**: Fixed in latest version - should pass now

#### "FIM collected 0 log entries"
- **Cause**: No file changes detected
- **Solution**: Verify test files are created in UAT-Test directories

#### UAT Communication Test Warnings
- **Expected Behavior**: 
  - `Connected=False`  (No backend running)
  - `QueuedLogs > 0`  (Logs queued properly)
  - `TotalLogsSent=0`  (No logs sent)
- **If seeing warnings**: Check that backend is not running during UAT

### Performance Benchmarks

Expected UAT performance thresholds:
- **Memory Usage**: < 256MB
- **FIM Detection**: < 5 seconds
- **Log Processing**: > 500 logs/second
- **CPU Usage**: < 50%

## Advanced Usage

### Custom Test Scenarios

You can modify test scenarios by editing `appsettings.uat.json`:

```json
{
  "Collectors": [
    {
      "Type": "FileIntegrity",
      "Properties": {
        "MonitoredPaths": [
          ".\\Your-Custom-Path",
          ".\\Another-Test-Directory"
        ]
      }
    }
  ]
}
```

### Integration with CI/CD

UAT returns appropriate exit codes:
- **0**: All tests passed
- **1**: One or more tests failed
- **2**: UAT framework error

Example Jenkins pipeline:
```groovy
stage('UAT Testing') {
    steps {
        bat 'run-uat.bat'
        publishHTML([
            allowMissing: false,
            alwaysLinkToLastBuild: true,
            keepAll: true,
            reportDir: 'UAT-Reports',
            reportFiles: '*.html',
            reportName: 'UAT Report'
        ])
    }
}
```

## Best Practices

1. **Run UAT before production deployment**
2. **Check HTML reports for detailed analysis**
3. **Monitor memory usage during testing**
4. **Clean up test files after UAT**
5. **Use non-admin mode for basic testing**

## Support

For issues with UAT framework:
1. Check `UAT-Logs/` for detailed error logs
2. Review HTML report for test step details
3. Verify test environment setup
4. Check file permissions for test directories

Remember: **Communication Test showing `Connected=False` is expected behavior in UAT environment!**

## Summary of Communication Test Fix

 **FIXED**: UAT now correctly shows `Connected=False` in isolated environment  
 **IMPROVED**: Better validation of UAT offline behavior  
 **ENHANCED**: Clearer warnings and guidance for unexpected behavior  

**Expected UAT Communication Results**:
- `Connected=False` (No backend connection)
- `QueuedLogs > 0` (Logs properly queued)
- `TotalLogsSent=0` (No logs sent)

If you see `Connected=True` during UAT, this indicates a backend service is running when it shouldn't be. 