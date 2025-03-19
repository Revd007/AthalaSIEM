# AthalaSIEM Agent

## Overview

AthalaSIEM Agent is a security information and event management (SIEM) agent designed to collect, normalize, and forward logs and system metrics to the AthalaSIEM backend server for analysis, storage, and alerting. It operates as a Windows service or Linux daemon with configurable log collectors for various sources.

## Features

- **Multi-platform Support**: Windows and Linux compatible agent
- **Multiple Log Collection Methods**:
  - Windows Event Log (Application, System, Security)
  - Linux Syslog and standard log files
  - Custom log files and sources
- **System Metrics Collection**:
  - CPU usage and utilization
  - Memory usage statistics
  - Disk space and I/O metrics
  - Network interface statistics
- **Secure Communication**:
  - HTTPS API communication (Port 9596)
  - gRPC for efficient streaming (Port 50051)
  - Optional mutual TLS authentication
  - Log encryption and compression
- **Intelligent Processing**:
  - Log normalization and format standardization
  - Event categorization and classification
  - Buffering and batching for efficient transmission
  - Auto-retry mechanisms for unreliable networks
- **Flexible Configuration**:
  - GUI configuration tool
  - Automatic deployment with tokens
  - Centralized configuration management from server

## Architecture Overview

AthalaSIEM Agent follows a modular architecture based on dependency injection and the separation of concerns principle. The architecture is divided into several layers:

1. **Core Layer** (`Program.cs`): Bootstraps the application, configures services, and sets up dependency injection
2. **Collection Layer**: Responsible for gathering logs and metrics from various sources
3. **Processing Layer**: Normalizes and enriches collected data
4. **Communication Layer**: Manages data transmission to the backend
5. **Configuration Layer**: Handles agent settings and configuration
6. **Security Layer**: Implements authentication and encryption

The agent uses a worker service pattern to run background processes and implements the observer pattern for event handling between components.

### High-Level Data Flow

```
                   +---------------+
                   | Configuration |
                   +-------+-------+
                           |
                           v
+----------+      +--------+-------+      +--------------+
| Log      +----->+ Log Processors +----->+ Log Forwarder+-----> Backend Server
| Collectors|      +----------------+      +--------------+       (HTTPS/gRPC)
+----------+              ^
                          |
+----------+      +-------+--------+
| Metric    +----->+ Health Monitor |
| Collectors|      +----------------+
+----------+
```

## Project Structure in Detail

### `/Collectors`

Contains components responsible for collecting logs and metrics from various sources.

- **`ILogCollector.cs`**: Base interface for all log collectors
- **`LogCollectorFactory.cs`**: Factory pattern implementation for creating appropriate collectors
- **`WindowsEventLogCollector.cs`**: Windows Event Log collection implementation
- **`LinuxSyslogCollector.cs`**: Linux Syslog collection implementation
- **`LogNormalizer.cs`**: Standardizes logs from different sources to a common format

**Design Pattern**: Factory pattern for creating collectors, strategy pattern for collector implementations

**Extension Point**: Create custom collectors by implementing the `ILogCollector` interface and registering with `LogCollectorFactory`

### `/Communication`

Handles communication with the backend server.

- **`ILogForwarder.cs`**: Interface for log forwarding implementations
- **`GrpcLogForwarder.cs`**: gRPC implementation for forwarding logs
- **`LogBatchProcessor.cs`**: Batches logs for efficient transmission
- **`LogCompressor.cs`**: Compresses log data before transmission

**Design Pattern**: Strategy pattern for different communication protocols

**Data Flow**: `LogBatchProcessor` → `LogCompressor` → `GrpcLogForwarder` → Backend

### `/Security`

Implements security features for the agent.

- **`IAgentIdentityService.cs`**: Interface for agent identity management
- **`AgentIdentityService.cs`**: Implementation of agent identity management
- **`IEncryptionService.cs`**: Interface for encryption/decryption operations
- **`AesEncryptionService.cs`**: AES implementation of encryption/decryption

**Design Pattern**: Strategy pattern for different encryption methods

### `/Configuration`

Manages agent configuration and settings.

- **`AgentConfigurationForm.cs`**: Windows Forms UI for configuration
- **`AgentConfigurationLauncher.cs`**: Launches the configuration UI

**Data Persistence**: Configuration is stored in `appsettings.json` and can be updated programmatically or via UI

### `/Services`

Contains core agent services.

- **`IAgentHealthMonitor.cs`**: Interface for health monitoring
- **`AgentHealthMonitor.cs`**: Implementation of agent health monitoring
- **`SiemAgentService.cs`**: Main agent service implementation

**Hosting Model**: Implements `IHostedService` for background processing

### `/Models`

Contains data models used throughout the agent.

- **`AgentSettings.cs`**: Configuration settings model
- **`SystemMetrics.cs`**: System performance metrics model
- **`LogModels.cs`**: Log-related data models
- **`RawLogData.cs`**: Raw log data structure
- **`NormalizedLogEntry.cs`**: Normalized log entry structure
- **`AgentHealthModels.cs`**: Health monitoring data models

**Design Pattern**: POCO (Plain Old CLR Objects) for data transfer

### `/Extensions`

Contains extension methods for various classes.

- **`ServiceCollectionExtensions.cs`**: Extensions for service registration

**Design Pattern**: Extension methods pattern for enhancing existing classes

### `/Protos`

Contains Protocol Buffer definitions for gRPC communication.

- **`siem.proto`**: gRPC service and message definitions

## Dependency Injection

The agent uses the built-in .NET dependency injection container. Services are registered in the `Program.cs` file and in `ServiceCollectionExtensions.cs`. The key registrations include:

```csharp
// Core services
services.AddSingleton<IEncryptionService, AesEncryptionService>();
services.AddSingleton<IAgentIdentityService, AgentIdentityService>();
services.AddSingleton<IAgentHealthMonitor, AgentHealthMonitor>();
services.AddSingleton<ILogForwarder, GrpcLogForwarder>();
services.AddSingleton<ILogNormalizer, LogNormalizer>();
services.AddSingleton<ILogCollectorFactory, LogCollectorFactory>();

// HTTP and gRPC clients
services.AddHttpClient("SiemBackend", /* configuration */);
services.AddGrpcClient<SiemService.SiemServiceClient>(/* configuration */);

// Background services
services.AddHostedService<SiemAgentService>();
```

## Agent Settings

The agent is configured using the following settings in `appsettings.json`:

- `AgentName`: Name of the agent (optional, will use hostname if not specified)
- `BackendUrl`: URL of the Athala SIEM backend
- `HeartbeatIntervalMinutes`: Interval in minutes between heartbeats
- `ConfigRefreshIntervalMinutes`: Interval in minutes between configuration refreshes
- `LogBatchSize`: Number of logs to batch before sending to the backend
- `MaxLogBatchIntervalSeconds`: Maximum interval in seconds to wait before sending a batch
- `CollectSystemMetrics`: Whether to collect system metrics
- `SystemMetricsIntervalMinutes`: Interval in minutes between system metrics collections
- `UseCompression`: Whether to compress logs before sending to the backend
- `EncryptLogs`: Whether to encrypt logs before sending to the backend

## Error Handling Strategy

The agent implements a robust error handling strategy:

1. **Exception Handling**: Try-catch blocks around critical operations
2. **Retry Logic**: Polly-based HTTP retry policies
3. **Circuit Breaker**: Prevents repeated calls to failing endpoints
4. **Logging**: Detailed error logging with context information
5. **Graceful Degradation**: Continues partial operation when some components fail

## System Requirements

### Windows
- Windows 10/11 or Windows Server 2016/2019/2022
- .NET 8.0 Runtime
- Administrator privileges for installation and Windows Events access
- Outbound connectivity to AthalaSIEM server (ports 9596 and 50051)

### Linux
- Ubuntu 20.04+, CentOS/RHEL 8+, or other modern distributions
- .NET 8.0 Runtime
- Root privileges for installation and syslog access
- Outbound connectivity to AthalaSIEM server (ports 9596 and 50051)

## Installation

### Windows Installation

1. **Prerequisites**:
   - Windows 8.1/10/11 or Windows Server 2012 R2 or later
   - .NET 8.0 Runtime or later
   - Administrator privileges (required for service installation)

2. **Installation Steps**:
   - **IMPORTANT**: Right-click on the AthalaSIEMAgent.msi installer and select "Run as administrator"
   - Follow the installation wizard prompts
   - Enter your server connection details when prompted
   - Click "Finish" to complete the installation

3. **Troubleshooting**:
   - If you see a "Service could not be installed" error, ensure you're running the installer as administrator
   - Check Windows Event Logs for detailed error information
   - Verify that the .NET Runtime is correctly installed

4. **Manual Service Installation** (if needed):
   ```powershell
   # Open Command Prompt as Administrator
   cd "C:\Program Files (x86)\Athala SIEM Agent"
   sc.exe create AthalaSIEMAgent binPath= "\"C:\Program Files (x86)\Athala SIEM Agent\AthalaSIEM.Agent.exe\"" start= auto DisplayName= "Athala SIEM Agent"
   ```

### Linux Installation

1. **Prerequisites**:
   - Ubuntu 20.04/22.04, CentOS/RHEL 8, or other supported Linux distribution
   - .NET 8.0 Runtime or later
   - Root privileges (for system service installation)

2. **Installation Steps**:
   ```bash
   # Extract the package
   tar -xvzf AthalaSIEMAgent.tar.gz
   cd AthalaSIEMAgent
   
   # Run the installation script with sudo
   sudo ./install.sh
   ```

## Configuration

### Main Configuration File
The agent is configured through the `appsettings.json` file, which contains the following key sections:

```json
{
  "AgentSettings": {
    "AgentName": "AthalaSIEM_Agent",
    "BackendUrl": "https://server:9596",
    "BackendGrpcUrl": "http://server:50051",
    "LogBatchSize": 100,
    "MaxLogBufferSize": 1000,
    "HeartbeatIntervalMinutes": 1,
    "HealthMonitoringIntervalMinutes": 5,
    "EnableSsl": true,
    "UseCompression": true
  },
  "Collectors": [
    {
      "Type": "WindowsEventLog",
      "Enabled": true,
      "IntervalSeconds": 10,
      "Properties": {
        "EventLogs": "Application,System,Security",
        "MaxEvents": "1000"
      }
    }
  ]
}
```

### Configuration UI
Run the configuration UI using:
```
AthalaSIEM.Agent.exe --config
```

### Command-Line Options
- `--register`: Register agent with backend using token
- `--config`: Launch configuration UI
- `--install`: Install as a Windows service
- `--uninstall`: Uninstall Windows service
- `--start`: Start the Windows service
- `--stop`: Stop the Windows service
- `--help`: Show help

## Development

### Prerequisites
- Visual Studio 2022 or later with .NET 8.0 SDK
- Git for source control
- WiX Toolset v4 (for building installers)

### Building
1. Clone the repository
2. Open the solution in Visual Studio
3. Build the solution using Release configuration

### Creating Windows Installer
Run the `build-installer.ps1` script in the Installer/Windows directory:
```powershell
cd .\agent\Installer\Windows\
.\build-installer.ps1
```

### Creating Linux Packages
Run the `build-linux-packages.sh` script:
```bash
chmod +x build-linux-packages.sh
./build-linux-packages.sh
```

## Service Configuration and Working Directory

The AthalaSIEM Agent service is configured to use the installation directory as its working directory. This ensures that configuration files are found in the correct location.

### Configuration Files

The agent looks for `appsettings.json` in the following locations (in order of priority):

1. Registry-defined path (HKLM\SYSTEM\CurrentControlSet\Services\AthalaSIEMAgent\Parameters\ConfigPath)
2. Executable directory
3. Standard installation paths (C:\Program Files, C:\Program Files (x86))
4. Config subdirectory
5. ProgramData (C:\ProgramData\Athala SIEM Agent)

As a last resort, the agent will create a minimal configuration file in ProgramData.

### Log Files

Logs are stored in the following locations (in order of priority):

1. Registry-defined working directory's 'logs' subfolder
2. Configuration file directory's 'logs' subfolder
3. ProgramData (C:\ProgramData\Athala SIEM Agent\logs)
4. Temp directory

### Registry Keys Used by AthalaSIEM Agent

The agent uses the following Windows registry keys for proper service operation:

1. **Service Parameters** - Standard Windows service configuration
   ```
   HKLM\SYSTEM\CurrentControlSet\Services\AthalaSIEMAgent\Parameters
     - WorkingDirectory: Points to the installation folder
     - ConfigPath: Path to the appsettings.json file
   ```

2. **Application Paths** - Helps Windows locate the executable
   ```
   HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\AthalaSIEM.Agent.exe
     - (Default): Full path to the executable
     - Path: Installation directory
   ```

3. **Installation Status** - Records user installation settings
   ```
   HKCU\Software\AthalaSecurity\AthalaSIEMAgent
     - installed: Value set to 1 when installation is complete
   ```

### Manual Service Management

To manage the AthalaSIEM Agent service:

1. Open Services Management Console:
   - Press Win+R, type `services.msc` and press Enter
   - Or use the "Manage Athala SIEM Agent Service" shortcut in the Start Menu

2. Locate "Athala SIEM Agent" in the services list

3. Service management options:
   - To start: Right-click → Start
   - To stop: Right-click → Stop
   - To restart: Right-click → Restart
   - To change startup type: Right-click → Properties → Startup type

4. For advanced configuration via command line (run as Administrator):
   ```
   sc config AthalaSIEMAgent start= auto    # Set to start automatically
   sc config AthalaSIEMAgent start= demand  # Set to start manually
   sc start AthalaSIEMAgent                 # Start the service
   sc stop AthalaSIEMAgent                  # Stop the service
   sc query AthalaSIEMAgent                 # Check service status
   ```

## Troubleshooting

### Log Locations
- Windows: `C:\Program Files\Athala SIEM Agent\Logs\` or `C:\ProgramData\Athala SIEM Agent\logs\`
- Linux: `/var/log/athalasiem/`

### Common Issues
1. **Agent not connecting to server**:
   - Verify network connectivity to server
   - Check that ports 9596 and 50051 are open
   - Verify server URL and ports in configuration
   - Check for valid API key or deployment token

2. **Service not starting**:
   - Check Windows Event Viewer or Linux system logs
   - Verify that the service user has sufficient permissions
   - Check configuration file for errors
   - Ensure the configuration file is in the correct location

3. **No logs being collected**:
   - Verify collector configuration in settings
   - Check that the agent has permissions to read log sources
   - Review agent logs for any collection errors

4. **Configuration Issues**:
   - If the service fails to start due to configuration errors, check:
     - If appsettings.json exists in the installation directory
     - If the file has valid JSON format
     - If all required settings are present
     - Windows Event Viewer for detailed error messages

## License

Copyright © 2025 Athala Security Solutions

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.