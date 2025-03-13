# Athala SIEM Agent

The Athala SIEM Agent is a cross-platform security monitoring agent that collects, normalizes, and forwards logs and system metrics to the Athala SIEM backend for analysis and alerting.

## Features

- **Cross-Platform Support**: Runs on both Windows and Linux operating systems
- **Automatic Log Collection**: Collects logs from various sources including Windows Event Logs and Linux Syslog
- **Log Normalization**: Standardizes logs from different sources into a common format
- **Secure Communication**: Uses gRPC with TLS for secure communication with the backend
- **Health Monitoring**: Monitors agent health and system metrics
- **Automatic Registration**: Self-registers with the backend on first run
- **Configurable**: Highly configurable through settings file or backend configuration

## Installation

### Windows

1. Download the latest release from the releases page
2. Run the installer and follow the prompts
3. The agent will be installed as a Windows service and start automatically

### Linux

1. Download the latest release from the releases page
2. Extract the archive to a directory of your choice
3. Run the installation script: `sudo ./install.sh`
4. The agent will be installed as a systemd service and start automatically

## Configuration

The agent is configured through the `appsettings.json` file. The following settings are available:

### Agent Settings

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

### Collectors

The agent supports multiple log collectors, each with its own settings:

#### Windows Event Log Collector

```json
{
  "Type": "WindowsEventLog",
  "Enabled": true,
  "IntervalSeconds": 60,
  "Settings": {
    "EventLogs": "Application,System,Security",
    "CollectionMode": "Polling",
    "MaxEvents": "100",
    "QueryFilter": "*[System[TimeCreated[timediff(@SystemTime) <= 3600000]]]"
  }
}
```

#### Linux Syslog Collector

```json
{
  "Type": "LinuxSyslog",
  "Enabled": true,
  "IntervalSeconds": 60,
  "Settings": {
    "SyslogFiles": "/var/log/syslog,/var/log/messages",
    "CollectionMode": "Polling",
    "MaxLinesPerRead": "1000"
  }
}
```

### Proxy Settings

If the agent needs to connect through a proxy, configure the following settings:

```json
"Proxy": {
  "Enabled": true,
  "Address": "proxy.example.com",
  "Port": 8080,
  "Username": "proxyuser",
  "Password": "proxypassword"
}
```

## Development

### Prerequisites

- .NET 8.0 SDK or later
- Visual Studio 2022 or later (for Windows development)
- VS Code with C# extension (for cross-platform development)

### Building

1. Clone the repository
2. Open the solution in Visual Studio or VS Code
3. Build the solution

### Running in Development Mode

```bash
dotnet run --project agent/agent.csproj
```

### Creating a Release

```bash
dotnet publish -c Release -r win-x64 --self-contained true -p:PublishSingleFile=true
```

Replace `win-x64` with `linux-x64` for Linux builds.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 