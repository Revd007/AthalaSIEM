# AthalaSIEM Universal Agent - MSI Installer

## Overview

This directory contains the enterprise-grade MSI installer for the AthalaSIEM Universal Agent. The installer follows industry standards used by major SIEM tools like Splunk, QRadar, and ArcSight, providing professional deployment capabilities for enterprise environments.

## Features

### 🚀 Enterprise Installation Experience
- **Professional MSI Package**: Built using WiX Toolset for Windows Installer compliance
- **Silent Installation Support**: Full automation for enterprise deployments
- **Group Policy Deployment**: SCCM and GPO deployment ready
- **Upgrade Management**: Automatic handling of version upgrades
- **Uninstall Support**: Clean removal with registry cleanup

### 🏢 Enterprise Deployment Capabilities
- **Command Line Configuration**: Set backend URL, tokens, and settings during installation
- **Registry Integration**: Proper Windows registry configuration
- **Service Management**: Automatic Windows service installation and configuration
- **Directory Structure**: Standard program files layout with proper permissions
- **Start Menu Integration**: Professional shortcuts and documentation access

### 🔧 Configuration Management
- **Backend Auto-Discovery**: Automatic connection to SIEM backend
- **Token Management**: Secure deployment token handling
- **Feature Selection**: Configurable installation components
- **Settings Persistence**: Registry-based configuration storage

## File Structure

```
Installer/
├── EnterpriseMSI.wxs          # Main WiX installer definition
├── AthalaSIEM.wixproj         # WiX project file for Visual Studio/MSBuild
├── build.ps1                  # PowerShell build script
├── LICENSE.rtf                # End User License Agreement
├── README-INSTALLER.md        # This file
├── Banner.bmp                 # Installer banner image (493x58)
├── Dialog.bmp                 # Installer dialog image (493x312)
└── dist/                      # Build output directory
    └── installer/
        ├── AthalaSIEM-UniversalAgent-1.0.0-x64.msi
        └── INSTALLATION-INSTRUCTIONS.md
```

## Building the Installer

### Prerequisites

1. **WiX Toolset**: Download and install from [wixtoolset.org](https://wixtoolset.org/releases/)
   - WiX Toolset v4.x (recommended) or v3.11+
   - Visual Studio integration optional

2. **.NET 8 SDK**: Required for building the agent application
   - Download from [dotnet.microsoft.com](https://dotnet.microsoft.com/download/dotnet/8.0)

3. **Visual Studio** (optional): For IDE-based building
   - Visual Studio 2022 (any edition)
   - WiX Toolset Visual Studio Extension

### Build Methods

#### Method 1: PowerShell Script (Recommended)
```powershell
# Navigate to installer directory
cd agent-universal\Installer

# Build with default settings
.\build.ps1

# Build with custom settings
.\build.ps1 -Configuration Release -Version "1.0.1.0" -OutputPath ".\dist\installer"
```

#### Method 2: Manual WiX Commands
```cmd
# Compile WiX source
candle.exe EnterpriseMSI.wxs -out AthalaSIEM-UniversalAgent.wixobj -dSourceDir=..\bin\Release\net8.0\win-x64\publish -dVersion=1.0.0.0

# Link to create MSI
light.exe AthalaSIEM-UniversalAgent.wixobj -out AthalaSIEM-UniversalAgent-1.0.0-x64.msi -ext WixUIExtension
```

#### Method 3: Visual Studio/MSBuild
```cmd
# Using MSBuild
msbuild AthalaSIEM.wixproj /p:Configuration=Release /p:Platform=x64

# Using Visual Studio
# Open AthalaSIEM.wixproj in Visual Studio and build
```

## Installation Methods

### Interactive Installation

Double-click the MSI file and follow the installation wizard:

1. **Welcome Screen**: Introduction and license agreement
2. **Installation Directory**: Choose installation location (default: `C:\Program Files\Athala Security\AthalaSIEM Universal Agent\`)
3. **Features Selection**: Choose components to install
4. **Configuration**: Enter SIEM backend URL and deployment token
5. **Installation**: Files are copied and service is installed
6. **Completion**: Start menu shortcuts created

### Silent Installation

For enterprise deployments, use command-line parameters:

```cmd
# Basic silent installation
msiexec /i "AthalaSIEM-UniversalAgent-1.0.0-x64.msi" /quiet

# Silent installation with configuration (IP:port format like other SIEM tools)
msiexec /i "AthalaSIEM-UniversalAgent-1.0.0-x64.msi" /quiet ^
  BACKENDURL="192.168.1.100:9595" ^
  GRPCURL="192.168.1.100:50051" ^
  DEPLOYMENTTOKEN="your-deployment-token" ^
  AGENTNAME="WebServer01-Agent" ^
  USEHTTPS="0" ^
  USEGRPC="1"

# Silent installation with logging
msiexec /i "AthalaSIEM-UniversalAgent-1.0.0-x64.msi" /quiet /l*v install.log
```

### Group Policy Deployment

1. Copy MSI to network share: `\\server\share\AthalaSIEM-UniversalAgent-1.0.0-x64.msi`
2. Open Group Policy Management Console
3. Create new GPO: "AthalaSIEM Agent Deployment"
4. Navigate to: Computer Configuration > Policies > Software Settings > Software installation
5. Right-click and select "New > Package"
6. Browse to MSI file and select "Assigned"
7. Configure installation parameters
8. Link GPO to target OUs

### SCCM Deployment

1. **Create Application**:
   - Name: AthalaSIEM Universal Agent
   - Publisher: Athala Security Systems
   - Version: 1.0.0

2. **Configure Detection Method**:
   - Registry: `HKLM\SOFTWARE\AthalaSIEM\UniversalAgent`
   - Value: `Version` equals `1.0.0`

3. **Installation Program**:
   ```cmd
   msiexec /i "AthalaSIEM-UniversalAgent-1.0.0-x64.msi" /quiet BACKENDURL="192.168.1.100:9595" GRPCURL="192.168.1.100:50051"
   ```

4. **Uninstall Program**:
   ```cmd
   msiexec /x {F7B8C9D0-1234-5678-9ABC-DEF012345678} /quiet
   ```

## Configuration Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `BACKENDURL` | SIEM backend address (IP:port format, like other SIEM tools) | `[ComputerName]:9595` | `192.168.1.100:9595` or `siem.company.com:9595` |
| `GRPCURL` | gRPC backend address (IP:port format) | `[BACKENDIP]:50051` | `192.168.1.100:50051` |
| `BACKENDPORT` | Backend HTTP/REST port | `9595` | `9595` |
| `GRPCPORT` | Backend gRPC port | `50051` | `50051` |
| `DEPLOYMENTTOKEN` | Deployment token for agent registration | None | `abc123def456` |
| `AGENTNAME` | Agent name | `[ComputerName]-Agent` | `WebServer01-Agent` |
| `USEHTTPS` | Use HTTPS for backend communication | `0` | `1` (yes) or `0` (no) |
| `USEGRPC` | Use gRPC for data plane (falls back to HTTP) | `1` | `1` (yes) or `0` (no) |

## Installation Components

### Main Feature (Required)
- **AthalaSIEM Universal Agent Core**: Main application and service
- **Registry Configuration**: Windows registry settings
- **Directory Structure**: Standard folder layout
- **Start Menu Shortcuts**: Configuration and status tools

### Optional Features
- **Desktop Shortcut**: Quick access shortcut on desktop
- **Documentation**: Installation guides and help files

### Advanced Monitoring (Always Installed)
- **File Integrity Monitoring (FIM)**: Configurable via backend
- **Registry Monitoring**: Real-time registry change detection
- **Event Correlation**: Advanced security correlation algorithms

## Directory Layout After Installation

```
C:\Program Files\Athala Security\AthalaSIEM Universal Agent\
├── athala-agent.exe                    # Main application
├── athala-agent.dll                    # Application library
├── athala-agent.runtimeconfig.json     # .NET runtime configuration
├── athala-agent.deps.json              # Dependencies
├── appsettings.json                    # Application configuration
├── bin\                                # Additional binaries
├── config\                             # Configuration files
├── logs\                               # Application logs
├── archives\                           # Log archives
├── certificates\                       # SSL certificates
├── temp\                               # Temporary files
└── docs\                               # Documentation
    ├── README.md
    ├── INSTALLATION-GUIDE.md
    ├── ENTERPRISE-CONFIGURATION-SUMMARY.md
    ├── SECURITY-CONFIGURATION-GUIDE.md
    └── LICENSE.rtf
```

## Registry Configuration

### Application Registry Keys
```
HKEY_LOCAL_MACHINE\SOFTWARE\AthalaSIEM\UniversalAgent\
├── InstallPath          # Installation directory
├── Version              # Installed version
├── InstallDate          # Installation date
└── ServiceName          # Windows service name
```

### Configuration Registry Keys
```
HKEY_LOCAL_MACHINE\SOFTWARE\AthalaSIEM\UniversalAgent\Configuration\
├── BackendUrl                    # SIEM backend URL
├── AgentName                     # Unique agent name
├── AutoStartService              # Auto-start setting
├── UseHttps                      # HTTPS preference
├── EnableFIM                     # FIM enable/disable
├── EnableRegistryMonitoring      # Registry monitoring
├── EnableEventCorrelation        # Event correlation
└── ConfigurationSource           # Configuration source (MSI)
```

## Windows Service Configuration

- **Service Name**: `AthalaSIEMUniversalAgent`
- **Display Name**: `AthalaSIEM Universal Agent`
- **Description**: Enterprise SIEM agent providing log collection, file integrity monitoring, registry monitoring, and security correlation for Windows systems
- **Start Type**: Automatic
- **Account**: Local System
- **Dependencies**: Event Log, TCP/IP Protocol Driver

## Start Menu Shortcuts

The installer creates shortcuts in: `Start Menu > Programs > AthalaSIEM Universal Agent`

- **Configure Agent**: Configure agent settings
- **Test SIEM Connection**: Test connectivity to backend
- **Agent Status**: View current agent status
- **Documentation**: Access help files
- **Uninstall**: Remove the application

## Troubleshooting

### Installation Issues

1. **"Administrator rights required"**
   - Run MSI as administrator or use elevated command prompt

2. **"WiX installation failed"**
   - Ensure Windows Installer service is running
   - Check for conflicting installations

3. **"File in use" errors**
   - Stop AthalaSIEM services before installation
   - Reboot and retry installation

### Build Issues

1. **"WiX Toolset not found"**
   - Install WiX Toolset from official website
   - Add WiX bin directory to PATH

2. **"candle.exe not recognized"**
   - Verify WiX installation
   - Check PATH environment variable

3. **".NET SDK not found"**
   - Install .NET 8 SDK
   - Verify with `dotnet --version`

### Runtime Issues

1. **Service won't start**
   - Check Event Viewer for error details
   - Verify configuration in registry
   - Test backend connectivity

2. **Cannot connect to backend**
   - Verify SIEM backend URL and port
   - Check firewall settings
   - Test network connectivity

## Enterprise Features

### Automatic Deployment
- **Token-based Registration**: Secure agent registration
- **Backend Auto-Discovery**: Automatic SIEM backend connection
- **Zero-Touch Installation**: Minimal manual configuration required

### Configuration Management
- **Backend-Driven Config**: All settings configurable via SIEM web interface
- **Dynamic Updates**: Configuration updates without restart
- **Enterprise Search**: Event ID management like Splunk/QRadar

### Security Features
- **Encrypted Communication**: Secure data transmission
- **Certificate Management**: SSL/TLS certificate handling
- **Access Control**: Proper Windows permissions

## Support and Documentation

- **Installation Guide**: Included in docs folder
- **Configuration Guide**: Enterprise configuration summary
- **Security Guide**: Security configuration guidelines
- **Online Documentation**: https://docs.athalasiem.com
- **Support Portal**: https://support.athalasecurity.com
- **Email Support**: support@athalasecurity.com

## Version History

- **v1.0.0**: Initial enterprise MSI release
  - Full WiX-based installer
  - Silent installation support
  - Enterprise deployment features
  - Complete registry integration

---

*AthalaSIEM Universal Agent - Enterprise Security Monitoring*
*Copyright (c) 2024 Athala Security Systems* 