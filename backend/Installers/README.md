# Athala SIEM Agent Installers

This directory contains the installers for the Athala SIEM Agent.

## Windows Installer (MSI)

The Windows installer should be built using the Windows Installer XML (WiX) toolset and placed in this directory as `AthalaAgent.msi`.

### Building the MSI Installer

1. Install WiX Toolset v3.11.2 or later
2. Navigate to the Agent project directory
3. Run the following commands:

```powershell
# Build the agent in release mode
dotnet publish -c Release -r win-x64 --self-contained true

# Create the MSI installer using WiX
# (This requires setting up a WiX project - see Agent/Installer/README.md for details)
```

### Required Files

The following file must be present in this directory:

- `AthalaAgent.msi` - The Windows installer package

### Security Considerations

The MSI installer should:
- Be digitally signed with a valid code signing certificate
- Include proper UAC elevation for installation
- Install the service with appropriate permissions
- Configure Windows Firewall rules automatically
- Set up proper event log access permissions
- Create secure configuration files with restricted access

### Installation Features

The MSI installer should handle:
- Automatic service installation and startup
- Configuration file deployment
- Event log source registration
- Firewall rule creation
- Proper file permissions setup
- Clean uninstallation support

### Troubleshooting

If the installer is not found when downloading from the API:
1. Ensure the MSI file is built and placed in this directory
2. Verify the file name matches exactly: `AthalaAgent.msi`
3. Check file permissions allow the web service to read the file 