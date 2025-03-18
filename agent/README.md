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
### Agent Settings
The agent source code is organized into the following directories:
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

## Event-Based Communication

The agent uses an event-based system for internal communication between components:

1. **Log Collection Events**: `LogCollected` event in `ILogCollector`
2. **Health Monitoring Events**: `HealthStatusChanged` event in `IAgentHealthMonitor`
3. **Configuration Change Events**: `ConfigurationChanged` event

## Thread Management

The agent manages multiple threads for different operations:

1. **Collection Threads**: One per collector (managed by each collector)
2. **Processing Thread**: For log normalization and enrichment
3. **Communication Thread**: For sending data to the backend
4. **Health Monitoring Thread**: For system metrics collection

Thread synchronization is handled through task-based operations and cancellation tokens.

## Configuration Schema

The agent configuration is defined in `appsettings.json` and follows this schema:

```json
{
  "Logging": {
    "LogLevel": { /* Logging configuration */ },
    "File": { /* File logging configuration */ },
    "EventLog": { /* Windows Event Log configuration */ }
  },
  "AgentSettings": {
    "AgentName": "string",
    "BackendApiUrl": "string",
    "BackendGrpcUrl": "string",
    "LogBatchSize": int,
    "MaxLogBufferSize": int,
    "LogSendingIntervalSeconds": int,
    "HeartbeatIntervalMinutes": int,
    "HealthMonitoringIntervalMinutes": int,
    "ConfigRefreshIntervalMinutes": int,
    "MaxLogBatchIntervalSeconds": int,
    "MaxRetries": int,
    "RetryDelaySeconds": int,
    "EncryptLogs": boolean,
    "UseMutualTls": boolean,
    "ClientCertificatePath": "string",
    "ClientCertificatePassword": "string",
    "ServerCaCertificatePath": "string",
    "ValidateServerCertificate": boolean,
    "UseTrafficCompression": boolean,
    "Collectors": [
      {
        "Type": "string",
        "Enabled": boolean,
        "IntervalSeconds": int,
        "Properties": {
          /* Collector-specific properties */
        }
      }
    ],
    "Proxy": {
      "Enabled": boolean,
      "Address": "string",
      "Port": int,
      "Username": "string",
      "Password": "string"
    }
  }
}
```

## Error Handling Strategy

The agent implements a robust error handling strategy:

1. **Exception Handling**: Try-catch blocks around critical operations
2. **Retry Logic**: Polly-based HTTP retry policies
3. **Circuit Breaker**: Prevents repeated calls to failing endpoints
4. **Logging**: Detailed error logging with context information
5. **Graceful Degradation**: Continues partial operation when some components fail

## Extension Points

The agent provides the following extension points for developers:

1. **Custom Collectors**: Implement `ILogCollector` and register with `LogCollectorFactory`
2. **Custom Log Normalizers**: Extend `LogNormalizer` or implement `ILogNormalizer`
3. **Custom Forwarders**: Implement `ILogForwarder` to support additional protocols
4. **Custom Encryption**: Implement `IEncryptionService` for different encryption methods
5. **Additional Metrics**: Extend `SystemMetrics` to collect additional metrics

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
1. Download the latest MSI installer
2. Run the installer with Administrator privileges
3. Follow the setup wizard to configure server connection details
4. Provide the deployment token if available
5. After installation, the service will be registered but not started automatically
6. Start the service via Services console (services.msc) or Command Prompt:
   ```
   sc start AthalaSIEMAgent
   ```

### Linux Installation
1. Install the .NET 8.0 Runtime if not already installed:
   ```
   wget https://dot.net/v1/dotnet-install.sh
   chmod +x dotnet-install.sh
   ./dotnet-install.sh --channel 8.0
   ```

2. Download and install the appropriate package for your distribution:
   ```
   # Debian/Ubuntu
   sudo dpkg -i athalasiem-agent_1.0.0_amd64.deb
   
   # CentOS/RHEL
   sudo rpm -i athalasiem-agent-1.0.0-1.x86_64.rpm
   ```

3. Configure the agent:
   ```
   sudo nano /etc/athalasiem-agent/appsettings.json
   ```

4. Start the service:
   ```
   sudo systemctl start athalasiem-agent
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

## Coding Conventions

The codebase follows these conventions:

1. **Naming**:
   - PascalCase for classes, methods, properties, and public members
   - camelCase for private fields (prefixed with underscore)
   - UPPER_CASE for constants

2. **Documentation**:
   - XML documentation for all public APIs
   - In-line comments for complex logic

3. **Error Handling**:
   - Prefer exceptions for exceptional conditions
   - Use result patterns for expected failure cases
   - Log errors before rethrowing

4. **Async Pattern**:
   - Async/await for asynchronous operations
   - Cancellation token support for cancellable operations
   - ConfigureAwait(false) for library code

## Testing

The agent includes several types of tests:

1. **Unit Tests**: For individual components
2. **Integration Tests**: For component interactions
3. **End-to-End Tests**: For full agent functionality

To run tests:
```
dotnet test agent.sln
```

## Troubleshooting

### Log Locations
- Windows: `C:\Program Files\Athala SIEM Agent\Logs\`
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

3. **No logs being collected**:
   - Verify collector configuration in settings
   - Check that the agent has permissions to read log sources
   - Review agent logs for any collection errors

## Contributing

We welcome contributions to the AthalaSIEM Agent. To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write tests for your changes
5. Submit a pull request

Please follow the existing code style and add appropriate documentation.

## License

Copyright © 2023 Athala SIEM. All rights reserved. 