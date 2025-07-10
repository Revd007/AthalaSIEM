# 🔒 AthalaSIEM Universal Agent - Security Configuration Guide

## PCI DSS v4.0.1 & ISO 27001:2022 Compliance

This guide provides security configuration requirements for audit compliance.

## ⚠️ CRITICAL SECURITY REQUIREMENTS

### 1. NO HARDCODED CREDENTIALS
- **NEVER** store passwords, API keys, or tokens in configuration files
- **NEVER** commit sensitive data to version control
- Use Windows Credential Manager or Azure Key Vault for secrets

### 2. REQUIRED SECURE CONFIGURATION

#### Environment Variables (Recommended)
```powershell
# SIEM Manager Configuration
$env:ATHALA_SiemManager__ManagerIP = "your-siem-server-ip"
$env:ATHALA_Agent__ManagerUrl = "https://your-siem-server:9595"
$env:ATHALA_Agent__RegistrationKey = "your-secure-registration-key"
$env:ATHALA_Agent__ApiKey = "your-secure-api-key"
```

#### Windows Registry Configuration (Production)
```powershell
# Create registry entries for secure configuration
New-ItemProperty -Path "HKLM:\SOFTWARE\AthalaSIEM\Agent" -Name "ManagerIP" -Value "your-siem-server-ip" -PropertyType String
New-ItemProperty -Path "HKLM:\SOFTWARE\AthalaSIEM\Agent" -Name "RegistrationKey" -Value "your-secure-key" -PropertyType String
New-ItemProperty -Path "HKLM:\SOFTWARE\AthalaSIEM\Agent" -Name "ApiKey" -Value "your-api-key" -PropertyType String
```

### 3. TLS/SSL REQUIREMENTS

#### Enable HTTPS (REQUIRED for Production)
```json
{
  "Agent": {
    "ManagerUrl": "https://your-siem-server:9595",
    "EnableTLS": true
  },
  "Security": {
    "EnableTLS": true,
    "ValidateCertificates": true
  }
}
```

### 4. FILE INTEGRITY MONITORING

#### Secure Path Configuration
```powershell
# Configure via registry for security
$paths = @(
    "C:\Windows\System32\drivers",
    "C:\Windows\System32\config",
    "C:\Program Files\YourCriticalApp"
)
New-ItemProperty -Path "HKLM:\SOFTWARE\AthalaSIEM\Agent\FileIntegrity" -Name "MonitoredPaths" -Value $paths -PropertyType MultiString
```

### 5. AUTHENTICATION & AUTHORIZATION

#### Service Account Configuration (REQUIRED)
```powershell
# Store service credentials in Windows Credential Manager
cmdkey /generic:"AthalaSIEM-ServiceAccount" /user:"DOMAIN\siem-service" /pass
```

#### Administrator Privileges
- Agent MUST run with Administrator privileges for Security Event Log access
- Service account MUST have "Log on as a service" privilege
- Follow principle of least privilege

### 6. LOGGING & AUDITING

#### Secure Logging Configuration
```json
{
  "Logging": {
    "LogLevel": {
      "Default": "Information",
      "AthalaSIEM": "Information"
    },
    "File": {
      "Enabled": true,
      "Path": "C:\ProgramData\AthalaSIEM\Logs\agent-{Date}.log"
    },
    "EventLog": {
      "Enabled": true,
      "SourceName": "AthalaSIEM Universal Agent"
    }
  }
}
```

### 7. NETWORK SECURITY

#### Firewall Configuration
```powershell
# Outbound HTTPS to SIEM Manager
New-NetFirewallRule -DisplayName "AthalaSIEM Agent HTTPS" -Direction Outbound -Protocol TCP -LocalPort Any -RemotePort 443,9595 -Action Allow
```

#### Network Validation
- Validate SIEM manager certificates
- Use mutual TLS authentication when possible
- Implement network segmentation

### 8. DATA PROTECTION

#### Encryption Requirements
```json
{
  "Security": {
    "EnableLogIntegrityHashing": true,
    "HashAlgorithm": "SHA256",
    "EnableEventValidation": true
  },
  "Communication": {
    "EnableBatchCompression": true,
    "EnableTLS": true,
    "VerifyCertificate": true
  }
}
```

## 🛡️ SECURITY CHECKLIST

### Pre-Deployment
- [ ] Remove all hardcoded IP addresses from configuration
- [ ] Remove all default passwords and keys
- [ ] Configure TLS/SSL certificates
- [ ] Set up Windows Credential Manager for service accounts
- [ ] Configure proper file system permissions
- [ ] Set up secure logging location

### Post-Deployment
- [ ] Verify no sensitive data in log files
- [ ] Test TLS connectivity to SIEM manager
- [ ] Validate service account permissions
- [ ] Confirm Security Event Log access
- [ ] Test file integrity monitoring paths
- [ ] Verify audit trail functionality

### Ongoing Maintenance
- [ ] Rotate API keys regularly (quarterly)
- [ ] Monitor for configuration drift
- [ ] Review access logs monthly
- [ ] Update certificates before expiration
- [ ] Conduct security assessments

## 📋 COMPLIANCE MATRIX

| Requirement | PCI DSS | ISO 27001 | Implementation |
|-------------|---------|-----------|----------------|
| Encryption in Transit | 4.1 | A.13.1.1 | TLS 1.3 for all communications |
| Strong Authentication | 8.2 | A.9.2.1 | Windows Authentication + API Keys |
| Access Control | 7.1 | A.9.1.1 | Role-based service accounts |
| Logging & Monitoring | 10.1 | A.12.4.1 | Comprehensive audit trails |
| Secure Configuration | 2.2 | A.12.6.1 | Remove defaults, secure storage |
| Key Management | 3.4 | A.10.1.2 | Windows Credential Manager |

## 🚨 SECURITY WARNINGS

### NEVER DO THIS:
```json
{
  "Agent": {
    "ManagerUrl": "http://192.168.1.100:9595",  // ❌ Hardcoded IP
    "ApiKey": "default-key-123",                 // ❌ Default key
    "ServicePassword": "password123"             // ❌ Plaintext password
  }
}
```

### DO THIS INSTEAD:
```json
{
  "Agent": {
    "ManagerUrl": "",        // ✅ Configure via environment variable
    "ApiKey": "",            // ✅ Configure via secure storage
    "ServicePassword": ""    // ✅ Use Windows Credential Manager
  }
}
```

## 📞 SECURITY SUPPORT

For security-related questions or to report vulnerabilities:
- Follow responsible disclosure practices
- Use encrypted communication channels
- Document all security configurations for audit purposes

---

**Remember**: Security is not optional for SIEM systems. Follow this guide to ensure compliance with PCI DSS v4.0.1 and ISO 27001:2022 standards. 