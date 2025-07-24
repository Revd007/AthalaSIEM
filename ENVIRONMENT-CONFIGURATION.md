# AthalaSIEM Environment Configuration Guide

## 🎯 **Default + Configurable Architecture**

AthalaSIEM follows enterprise SIEM best practices with **default values for development** and **environment overrides for production**.

## 📋 **Configuration Hierarchy**

1. **Hardcoded Defaults** (lowest priority) - Built into code
2. **appsettings.json** (medium priority) - Development defaults  
3. **Environment Variables** (highest priority) - Production overrides

## 🔧 **Backend Configuration**

### **Required Environment Variables (Production)**

```bash
# Database Connection
export ATHALA_ConnectionStrings__DefaultConnection="Host=your-db-host;Port=5432;Database=athala_siem;Username=siem_user;Password=your-secure-password;"

# JWT Security
export ATHALA_Jwt__Key="your-super-secure-jwt-key-minimum-32-characters"
export ATHALA_JwtSettings__Secret="your-super-secure-jwt-secret-minimum-32-characters"

# CORS Origins (Required for frontend access)
export ATHALA_Cors__AllowedOrigins__0="https://your-frontend-domain.com"
export ATHALA_Cors__AllowedOrigins__1="https://siem.your-company.com"
```

### **Optional Environment Variables**

```bash
# Custom Ports
export ATHALA_Kestrel__Endpoints__Http__Url="http://0.0.0.0:8080"
export ATHALA_GrpcServer__Url="http://0.0.0.0:8080"

# Service Discovery
export ATHALA_Enterprise__ServiceDiscovery__EnableDNS=true
export ATHALA_Enterprise__ServiceDiscovery__DNSRecords__0="_siem._tcp.your-domain.com"
```

## 🤖 **Agent Configuration**

### **Optional Environment Variables**

```bash
# SIEM Manager Discovery
export ATHALA_SiemManager__ManagerIP="10.0.1.100"
export ATHALA_SiemManager__ManagerPort=9595

# Agent Registration
export ATHALA_Agent__RegistrationKey="your-deployment-key"
export ATHALA_Agent__DeploymentToken="your-deployment-token"

# Performance Tuning
export ATHALA_Agent__BatchSize=10000
export ATHALA_Agent__MaxQueueSize=2000000
```

## 🌐 **Frontend Configuration**

Create `.env.production` file:

```bash
# Backend API URL
REACT_APP_API_URL=https://your-backend-domain.com
REACT_APP_WS_URL=wss://your-backend-domain.com

# Optional: Custom ports
REACT_APP_API_PORT=9595
```

## 🚀 **Default Ports (Development)**

| Service | Default Port | Environment Override |
|---------|--------------|---------------------|
| Backend API | 9595 | `ATHALA_Kestrel__Endpoints__Http__Url` |
| Frontend | 3000 | `PORT` |
| gRPC | 9595 | `ATHALA_GrpcServer__Url` |
| Database | 5432 | Connection string |

## 📦 **Docker Deployment Example**

```yaml
version: '3.8'
services:
  athala-backend:
    image: athala-siem/backend:latest
    environment:
      - ATHALA_ConnectionStrings__DefaultConnection=Host=postgres;Port=5432;Database=athala_siem;Username=siem_user;Password=secure_password;
      - ATHALA_Jwt__Key=your-production-jwt-key-32-chars-minimum
      - ATHALA_Cors__AllowedOrigins__0=https://siem.company.com
      - ATHALA_Kestrel__Endpoints__Http__Url=http://0.0.0.0:9595
    ports:
      - "9595:9595"
    
  athala-frontend:
    image: athala-siem/frontend:latest
    environment:
      - REACT_APP_API_URL=https://api.siem.company.com
    ports:
      - "3000:3000"
```

## 🔍 **Service Discovery**

### **DNS-based Discovery**
```bash
# Create DNS SRV records
_siem._tcp.company.com. 300 IN SRV 0 5 9595 siem-backend.company.com.
_athala._tcp.company.com. 300 IN SRV 0 5 9595 siem-backend.company.com.
```

### **Environment-based Discovery**
```bash
# Agent automatically discovers backend
export ATHALA_SiemManager__AutoDiscovery=true
export ATHALA_SiemManager__DiscoveryMethods__0="DNS"
export ATHALA_SiemManager__DiscoveryMethods__1="Broadcast"
```

## ✅ **Configuration Validation**

### **Backend Health Check**
```bash
curl http://localhost:9595/health
# Should return: {"status":"Healthy","version":"1.0.0"}
```

### **Agent Registration Test**
```bash
curl -X POST http://localhost:9595/api/agents/register \
  -H "Content-Type: application/json" \
  -d '{"hostname":"test-agent","platform":"Windows"}'
```

## 🛡️ **Security Best Practices**

1. **Never hardcode secrets** in configuration files
2. **Use environment variables** for all sensitive data
3. **Rotate JWT keys** regularly in production
4. **Restrict CORS origins** to known domains only
5. **Use HTTPS** in production environments

## 🔧 **Troubleshooting**

### **Common Issues**

1. **CORS Error**: Check `ATHALA_Cors__AllowedOrigins__*` variables
2. **Database Connection**: Verify `ATHALA_ConnectionStrings__DefaultConnection`
3. **Agent Registration**: Ensure backend is accessible on configured port
4. **Service Discovery**: Check DNS records and network connectivity

### **Debug Commands**

```bash
# Check environment variables
env | grep ATHALA_

# Test backend connectivity
telnet your-backend-host 9595

# Check DNS resolution
nslookup _siem._tcp.your-domain.com
```

## 📚 **Enterprise SIEM Comparison**

| Feature | Wazuh | Splunk | ManageEngine | AthalaSIEM |
|---------|-------|--------|--------------|------------|
| Default Ports | 1514/1515 | 8000/8089 | 8400/8443 | 9595 |
| Auto Discovery | ✅ | ✅ | ✅ | ✅ |
| Environment Config | ✅ | ✅ | ✅ | ✅ |
| Docker Support | ✅ | ✅ | ✅ | ✅ |

---

**💡 Pro Tip**: Start with defaults for development, then use environment variables for production deployment. This matches industry standards used by Wazuh, Splunk, and other enterprise SIEM tools. 