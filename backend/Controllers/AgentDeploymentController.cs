using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Authorization;
using Microsoft.EntityFrameworkCore;
using Backend.Data;
using Backend.Models;
using System.Text.Json;
using System.Security.Cryptography;
using System.Text;
using Microsoft.Extensions.Logging;

namespace Backend.Controllers
{
    /// <summary>
    /// Agent Deployment and Management Controller
    /// Handles multi-platform agent deployment, configuration, and monitoring
    /// </summary>
    [ApiController]
    [Route("api/[controller]")]
    [Authorize]
    public class AgentDeploymentController : ControllerBase
    {
        private readonly ApplicationDbContext _context;
        private readonly ILogger<AgentDeploymentController> _logger;

        public AgentDeploymentController(ApplicationDbContext context, ILogger<AgentDeploymentController> logger)
        {
            _context = context;
            _logger = logger;
        }

        /// <summary>
        /// Get all deployment tokens
        /// </summary>
        [HttpGet("tokens")]
        public async Task<ActionResult<IEnumerable<AgentDeploymentTokenDto>>> GetDeploymentTokens()
        {
            try
            {
                var tokens = await _context.AgentDeploymentTokens
                    .OrderByDescending(t => t.CreatedAt)
                    .Select(t => new AgentDeploymentTokenDto
                    {
                        Id = t.Id,
                        Name = t.Name,
                        Description = t.Description ?? string.Empty,
                        PlatformType = t.PlatformType,
                        ExpiresAt = t.ExpiresAt,
                        IsActive = t.IsActive,
                        UsageCount = t.UsageCount,
                        MaxUsage = t.MaxUsage,
                        CreatedAt = t.CreatedAt,
                        CreatedBy = t.CreatedBy ?? "system",
                        LastUsed = t.LastUsed
                    })
                    .ToListAsync();

                return Ok(tokens);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving deployment tokens");
                return StatusCode(500, "Internal server error");
            }
        }

        /// <summary>
        /// Create new deployment token
        /// </summary>
        [HttpPost("tokens")]
        public async Task<ActionResult<AgentDeploymentTokenDto>> CreateDeploymentToken([FromBody] CreateAgentDeploymentTokenRequest request)
        {
            try
            {
                var token = new AthalaSIEM.Backend.Models.AgentDeploymentToken
                {
                    Name = request.Name,
                    Description = request.Description,
                    PlatformType = request.PlatformType,
                    Token = GenerateSecureToken(),
                    ExpiresAt = request.ExpiresAt,
                    MaxUsage = request.MaxUsage,
                    IsActive = true,
                    Configuration = JsonSerializer.Serialize(request.Configuration),
                    CreatedBy = User.Identity?.Name ?? "system",
                    CreatedAt = DateTime.UtcNow
                };

                _context.AgentDeploymentTokens.Add(token);
                await _context.SaveChangesAsync();

                var response = new AgentDeploymentTokenDto
                {
                    Id = token.Id,
                    Name = token.Name,
                    Description = token.Description,
                    PlatformType = token.PlatformType,
                    Token = token.Token, // Include token only in creation response
                    ExpiresAt = token.ExpiresAt,
                    IsActive = token.IsActive,
                    UsageCount = token.UsageCount,
                    MaxUsage = token.MaxUsage,
                    CreatedAt = token.CreatedAt,
                    CreatedBy = token.CreatedBy ?? "system"
                };

                _logger.LogInformation("Deployment token created: {TokenName} for platform {Platform}", request.Name, request.PlatformType);
                return Ok(response);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating deployment token");
                return StatusCode(500, "Internal server error");
            }
        }

        /// <summary>
        /// Create new deployment token WITHOUT AUTHORIZATION (for testing only)
        /// </summary>
        [HttpPost("tokens/test")]
        [AllowAnonymous]
        public async Task<ActionResult<AgentDeploymentTokenDto>> CreateTestDeploymentToken([FromBody] CreateAgentDeploymentTokenRequest request)
        {
            try
            {
                var token = new AthalaSIEM.Backend.Models.AgentDeploymentToken
                {
                    Name = request.Name,
                    Description = request.Description,
                    PlatformType = request.PlatformType,
                    Token = "athala-siem-agent-registration-2025", // Fixed token for testing
                    ExpiresAt = DateTime.UtcNow.AddDays(30),
                    MaxUsage = request.MaxUsage,
                    IsActive = true,
                    Configuration = JsonSerializer.Serialize(request.Configuration),
                    CreatedBy = "system-test",
                    CreatedAt = DateTime.UtcNow
                };

                // Check if token already exists
                var existingToken = await _context.AgentDeploymentTokens
                    .FirstOrDefaultAsync(t => t.Token == token.Token);
                
                if (existingToken != null)
                {
                    return Ok(new AgentDeploymentTokenDto
                    {
                        Id = existingToken.Id,
                        Name = existingToken.Name,
                        Description = existingToken.Description,
                        PlatformType = existingToken.PlatformType,
                        Token = existingToken.Token,
                        ExpiresAt = existingToken.ExpiresAt,
                        IsActive = existingToken.IsActive,
                        UsageCount = existingToken.UsageCount,
                        MaxUsage = existingToken.MaxUsage,
                        CreatedAt = existingToken.CreatedAt,
                        CreatedBy = existingToken.CreatedBy
                    });
                }

                _context.AgentDeploymentTokens.Add(token);
                await _context.SaveChangesAsync();

                var response = new AgentDeploymentTokenDto
                {
                    Id = token.Id,
                    Name = token.Name,
                    Description = token.Description,
                    PlatformType = token.PlatformType,
                    Token = token.Token,
                    ExpiresAt = token.ExpiresAt,
                    IsActive = token.IsActive,
                    UsageCount = token.UsageCount,
                    MaxUsage = token.MaxUsage,
                    CreatedAt = token.CreatedAt,
                    CreatedBy = token.CreatedBy
                };

                _logger.LogInformation("Test deployment token created: {TokenName} for platform {Platform}", request.Name, request.PlatformType);
                return Ok(response);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating test deployment token");
                return StatusCode(500, new { Error = ex.Message });
            }
        }

        /// <summary>
        /// Get deployment script for specific platform
        /// </summary>
        [HttpGet("scripts/{platform}")]
        public async Task<ActionResult<AgentDeploymentScriptResponse>> GetDeploymentScript(string platform, [FromQuery] string tokenId)
        {
            try
            {
                var token = await _context.AgentDeploymentTokens
                    .FirstOrDefaultAsync(t => t.Id == tokenId && t.IsActive);

                if (token == null)
                {
                    return BadRequest("Invalid or expired deployment token");
                }

                if (token.ExpiresAt.HasValue && token.ExpiresAt.Value < DateTime.UtcNow)
                {
                    return BadRequest("Deployment token has expired");
                }

                if (token.MaxUsage.HasValue && token.UsageCount >= token.MaxUsage.Value)
                {
                    return BadRequest("Deployment token usage limit exceeded");
                }

                var script = GenerateDeploymentScript(platform, token);
                var instructions = GenerateDeploymentInstructions(platform);

                return Ok(new AgentDeploymentScriptResponse
                {
                    Platform = platform,
                    Script = script,
                    Instructions = instructions,
                    ConfigurationTemplate = GetConfigurationTemplate(platform),
                    Prerequisites = GetPrerequisites(platform)
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating deployment script for platform {Platform}", platform);
                return StatusCode(500, "Internal server error");
            }
        }

        /// <summary>
        /// Register new agent using deployment token
        /// </summary>
        [HttpPost("register")]
        [AllowAnonymous]
        public async Task<ActionResult<AgentRegistrationResponse>> RegisterAgent([FromBody] AgentRegistrationRequest request)
        {
            try
            {
                var token = await _context.AgentDeploymentTokens
                    .FirstOrDefaultAsync(t => t.Token == request.DeploymentToken && t.IsActive);

                if (token == null)
                {
                    return BadRequest("Invalid deployment token");
                }

                if (token.ExpiresAt.HasValue && token.ExpiresAt.Value < DateTime.UtcNow)
                {
                    return BadRequest("Deployment token has expired");
                }

                if (token.MaxUsage.HasValue && token.UsageCount >= token.MaxUsage.Value)
                {
                    return BadRequest("Deployment token usage limit exceeded");
                }

                // Check if agent already exists
                var existingAgent = await _context.Agents
                    .FirstOrDefaultAsync(a => a.Hostname == request.Hostname && a.IpAddress == request.IpAddress);

                if (existingAgent != null)
                {
                    return BadRequest($"Agent with hostname {request.Hostname} and IP {request.IpAddress} already exists");
                }

                // Create new agent
                var agent = new AgentModels
                {
                    Name = $"{request.Hostname}_{request.Platform}",
                    Hostname = request.Hostname,
                    IpAddress = request.IpAddress,
                    Platform = request.Platform,
                    OsVersion = request.OsVersion,
                    AgentVersion = request.AgentVersion,
                    ApiKey = GenerateApiKey(),
                    Status = AgentStatus.Online,
                    Type = GetAgentTypeFromPlatform(request.Platform),
                    LastSeen = DateTime.UtcNow,
                    CreatedAt = DateTime.UtcNow,
                    DeploymentTokenId = token.Id
                };

                // Create agent configuration
                var config = new AgentConfigModels
                {
                    AgentId = agent.Id,
                    Configuration = GetDefaultConfiguration(request.Platform, token),
                    LastUpdated = DateTime.UtcNow
                };

                _context.Agents.Add(agent);
                _context.AgentConfigs.Add(config);

                // Update token usage
                token.UsageCount++;
                token.LastUsed = DateTime.UtcNow;

                await _context.SaveChangesAsync();

                _logger.LogInformation("Agent registered successfully: {AgentName} ({Platform}) using token {TokenName}", 
                    agent.Name, request.Platform, token.Name);

                return Ok(new AgentRegistrationResponse
                {
                    AgentId = agent.Id,
                    ApiKey = agent.ApiKey,
                    BackendUrl = GetBackendUrl(),
                    Configuration = config.Configuration,
                    UpdateIntervalSeconds = 300,
                    HeartbeatIntervalSeconds = 60
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error registering agent");
                return StatusCode(500, "Internal server error");
            }
        }

        /// <summary>
        /// Get agent configuration updates
        /// </summary>
        [HttpGet("agents/{agentId}/configuration")]
        public async Task<ActionResult<AgentConfigurationResponse>> GetAgentConfiguration(string agentId)
        {
            try
            {
                var agent = await _context.Agents
                    .Include(a => a.Configuration)
                    .FirstOrDefaultAsync(a => a.Id == agentId);

                if (agent == null)
                {
                    return NotFound("Agent not found");
                }

                var config = agent.Configuration ?? new AgentConfigModels { Configuration = "{}" };

                return Ok(new AgentConfigurationResponse
                {
                    AgentId = agentId,
                    Configuration = config.Configuration ?? "{}",
                    LastUpdated = config.LastUpdated,
                    RequiresRestart = config.RequiresRestart
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving agent configuration for {AgentId}", agentId);
                return StatusCode(500, "Internal server error");
            }
        }

        /// <summary>
        /// Update agent configuration
        /// </summary>
        [HttpPut("agents/{agentId}/configuration")]
        public async Task<ActionResult> UpdateAgentConfiguration(string agentId, [FromBody] UpdateAgentConfigurationRequest request)
        {
            try
            {
                var agent = await _context.Agents
                    .Include(a => a.Configuration)
                    .FirstOrDefaultAsync(a => a.Id == agentId);

                if (agent == null)
                {
                    return NotFound("Agent not found");
                }

                if (agent.Configuration == null)
                {
                    agent.Configuration = new AgentConfigModels
                    {
                        AgentId = agentId
                    };
                    _context.AgentConfigs.Add(agent.Configuration);
                }

                agent.Configuration.Configuration = JsonSerializer.Serialize(request.Configuration);
                agent.Configuration.LastUpdated = DateTime.UtcNow;
                agent.Configuration.RequiresRestart = request.RequiresRestart;

                await _context.SaveChangesAsync();

                _logger.LogInformation("Agent configuration updated for {AgentId}", agentId);
                return Ok();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating agent configuration for {AgentId}", agentId);
                return StatusCode(500, "Internal server error");
            }
        }

        /// <summary>
        /// Get deployment statistics
        /// </summary>
        [HttpGet("statistics")]
        public async Task<ActionResult<DeploymentStatistics>> GetDeploymentStatistics()
        {
            try
            {
                var stats = new DeploymentStatistics
                {
                    TotalTokens = await _context.AgentDeploymentTokens.CountAsync(),
                    ActiveTokens = await _context.AgentDeploymentTokens.CountAsync(t => t.IsActive),
                    TotalDeployments = await _context.Agents.CountAsync(),
                    OnlineAgents = await _context.Agents.CountAsync(a => a.Status == AgentStatus.Online),
                    OfflineAgents = await _context.Agents.CountAsync(a => a.Status == AgentStatus.Offline),
                    PlatformDistribution = await _context.Agents
                        .GroupBy(a => a.Platform)
                        .Select(g => new PlatformCount { Platform = g.Key, Count = g.Count() })
                        .ToListAsync(),
                    RecentDeployments = await _context.Agents
                        .Where(a => a.CreatedAt >= DateTime.UtcNow.AddDays(-7))
                        .GroupBy(a => a.CreatedAt.Date)
                        .Select(g => new DeploymentTrend { Date = g.Key, Count = g.Count() })
                        .OrderBy(d => d.Date)
                        .ToListAsync()
                } ?? throw new InvalidOperationException("Value cannot be null");

                return Ok(stats);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving deployment statistics");
                return StatusCode(500, "Internal server error");
            }
        }

        /// <summary>
        /// Revoke deployment token
        /// </summary>
        [HttpPost("tokens/{tokenId}/revoke")]
        public async Task<ActionResult> RevokeDeploymentToken(string tokenId)
        {
            try
            {
                var token = await _context.AgentDeploymentTokens.FindAsync(tokenId);
                if (token == null)
                {
                    return NotFound("Deployment token not found");
                }

                token.IsActive = false;
                await _context.SaveChangesAsync();

                _logger.LogInformation("Deployment token revoked: {TokenName}", token.Name);
                return Ok();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error revoking deployment token {TokenId}", tokenId);
                return StatusCode(500, "Internal server error");
            }
        }

        /// <summary>
        /// Generate agent installer package
        /// </summary>
        [HttpPost("generate-installer")]
        public async Task<ActionResult<AgentInstallerResponse>> GenerateInstaller([FromBody] GenerateInstallerRequest request)
        {
            try
            {
                var token = await _context.AgentDeploymentTokens
                    .FirstOrDefaultAsync(t => t.Id == request.TokenId && t.IsActive) ?? throw new InvalidOperationException("Value cannot be null");

                if (token == null)
                {
                    return BadRequest("Invalid deployment token");
                }

                var installerData = GenerateInstallerPackage(request.Platform, token, request.CustomConfiguration);

                return Ok(new AgentInstallerResponse
                {
                    Platform = request.Platform,
                    InstallerUrl = installerData.Url,
                    ChecksumSha256 = installerData.Checksum,
                    ExpiresAt = DateTime.UtcNow.AddHours(24),
                    Instructions = GenerateInstallerInstructions(request.Platform)
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating installer for platform {Platform}", request.Platform);
                return StatusCode(500, "Internal server error");
            }
        }

        // Private helper methods

        private string GenerateSecureToken()
        {
            using var rng = RandomNumberGenerator.Create();
            var bytes = new byte[32];
            rng.GetBytes(bytes);
            return Convert.ToBase64String(bytes).Replace("+", "-").Replace("/", "_").Replace("=", "");
        }

        private string GenerateApiKey()
        {
            return Guid.NewGuid().ToString("N");
        }

        private AgentType GetAgentTypeFromPlatform(string platform)
        {
            return platform.ToLowerInvariant() switch
            {
                "windows" => AgentType.WindowsAgent,
                "linux" => AgentType.LinuxAgent,
                "freebsd" => AgentType.UnixAgent,
                "macos" => AgentType.MacAgent,
                "docker" => AgentType.ContainerAgent,
                _ => AgentType.GenericAgent
            };
        }

        private string GenerateDeploymentScript(string platform, AthalaSIEM.Backend.Models.AgentDeploymentToken token)
        {
            return platform.ToLowerInvariant() switch
            {
                "windows" => GenerateWindowsScript(token),
                "linux" => GenerateLinuxScript(token),
                "freebsd" => GenerateFreeBSDScript(token),
                "macos" => GenerateMacOSScript(token),
                "docker" => GenerateDockerScript(token),
                _ => GenerateGenericScript(token)
            };
        }

        private string GenerateWindowsScript(AthalaSIEM.Backend.Models.AgentDeploymentToken token)
        {
            return $@"# AthalaSIEM Agent Deployment Script for Windows
# Generated: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC

Write-Host ""Installing AthalaSIEM Agent..."" -ForegroundColor Green

# Configuration
$BackendUrl = ""{GetBackendUrl()}""
$DeploymentToken = ""{token.Token}""
$AgentDownloadUrl = ""{GetBackendUrl()}/downloads/agent/windows/athala-siem-agent.msi""

# Download and install agent
$TempPath = ""$env:TEMP\athala-siem-agent.msi""
try {{
    Write-Host ""Downloading agent installer..."" -ForegroundColor Yellow
    Invoke-WebRequest -Uri $AgentDownloadUrl -OutFile $TempPath -UseBasicParsing
    
    Write-Host ""Installing agent..."" -ForegroundColor Yellow
    Start-Process msiexec.exe -ArgumentList ""/i $TempPath /quiet BACKEND_URL=$BackendUrl DEPLOYMENT_TOKEN=$DeploymentToken"" -Wait
    
    Write-Host ""Starting AthalaSIEM Agent service..."" -ForegroundColor Yellow
    Start-Service -Name ""AthalaSIEMAgent""
    
    Write-Host ""Agent installed successfully!"" -ForegroundColor Green
}} catch {{
    Write-Host ""Error: $_"" -ForegroundColor Red
    exit 1
}} finally {{
    if (Test-Path $TempPath) {{ Remove-Item $TempPath -Force }}
}}";
        }

        private string GenerateLinuxScript(AthalaSIEM.Backend.Models.AgentDeploymentToken token)
        {
            return $@"#!/bin/bash
# AthalaSIEM Agent Deployment Script for Linux
# Generated: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC

set -e

echo ""Installing AthalaSIEM Agent...""

# Configuration
BACKEND_URL=""{GetBackendUrl()}""
DEPLOYMENT_TOKEN=""{token.Token}""
AGENT_DOWNLOAD_URL=""{GetBackendUrl()}/downloads/agent/linux/athala-siem-agent.deb""

# Detect distribution
if [ -f /etc/debian_version ]; then
    DISTRIB=""debian""
    PACKAGE_EXT=""deb""
elif [ -f /etc/redhat-release ]; then
    DISTRIB=""redhat""
    PACKAGE_EXT=""rpm""
    AGENT_DOWNLOAD_URL=""{GetBackendUrl()}/downloads/agent/linux/athala-siem-agent.rpm""
else
    echo ""Unsupported Linux distribution""
    exit 1
fi

# Download and install
TEMP_FILE=""/tmp/athala-siem-agent.$PACKAGE_EXT""

echo ""Downloading agent installer...""
curl -L -o ""$TEMP_FILE"" ""$AGENT_DOWNLOAD_URL""

echo ""Installing agent...""
if [ ""$DISTRIB"" = ""debian"" ]; then
    dpkg -i ""$TEMP_FILE""
    apt-get install -f -y
else
    rpm -ivh ""$TEMP_FILE""
fi

# Configure agent
echo ""Configuring agent...""
cat > /etc/athala-siem/agent.conf << EOF
backend_url=$BACKEND_URL
deployment_token=$DEPLOYMENT_TOKEN
EOF

# Start service
echo ""Starting AthalaSIEM Agent service...""
systemctl enable athala-siem-agent
systemctl start athala-siem-agent

echo ""Agent installed successfully!""

# Cleanup
rm -f ""$TEMP_FILE""";
        }

        private string GenerateFreeBSDScript(AthalaSIEM.Backend.Models.AgentDeploymentToken token)
        {
            return $@"#!/bin/sh
# AthalaSIEM Agent Deployment Script for FreeBSD
# Generated: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC

set -e

echo ""Installing AthalaSIEM Agent for FreeBSD...""

# Configuration
BACKEND_URL=""{GetBackendUrl()}""
DEPLOYMENT_TOKEN=""{token.Token}""
AGENT_DOWNLOAD_URL=""{GetBackendUrl()}/downloads/agent/freebsd/athala-siem-agent.txz""

# Install dependencies
echo ""Installing dependencies...""
pkg install -y curl

# Download and install
TEMP_FILE=""/tmp/athala-siem-agent.txz""

echo ""Downloading agent installer...""
fetch -o ""$TEMP_FILE"" ""$AGENT_DOWNLOAD_URL""

echo ""Installing agent...""
pkg add ""$TEMP_FILE""

# Configure agent
echo ""Configuring agent...""
mkdir -p /usr/local/etc/athala-siem
cat > /usr/local/etc/athala-siem/agent.conf << EOF
backend_url=$BACKEND_URL
deployment_token=$DEPLOYMENT_TOKEN
EOF

# Enable and start service
echo ""Starting AthalaSIEM Agent service...""
sysrc athala_siem_agent_enable=""YES""
service athala-siem-agent start

echo ""Agent installed successfully!""

# Cleanup
rm -f ""$TEMP_FILE""";
        }

        private string GenerateMacOSScript(AthalaSIEM.Backend.Models.AgentDeploymentToken token)
        {
            return $@"#!/bin/bash
# AthalaSIEM Agent Deployment Script for macOS
# Generated: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC

set -e

echo ""Installing AthalaSIEM Agent for macOS...""

# Configuration
BACKEND_URL=""{GetBackendUrl()}""
DEPLOYMENT_TOKEN=""{token.Token}""
AGENT_DOWNLOAD_URL=""{GetBackendUrl()}/downloads/agent/macos/athala-siem-agent.pkg""

# Download and install
TEMP_FILE=""/tmp/athala-siem-agent.pkg""

echo ""Downloading agent installer...""
curl -L -o ""$TEMP_FILE"" ""$AGENT_DOWNLOAD_URL""

echo ""Installing agent (requires admin privileges)...""
sudo installer -pkg ""$TEMP_FILE"" -target /

# Configure agent
echo ""Configuring agent...""
sudo mkdir -p /etc/athala-siem
sudo cat > /etc/athala-siem/agent.conf << EOF
backend_url=$BACKEND_URL
deployment_token=$DEPLOYMENT_TOKEN
EOF

# Start service
echo ""Starting AthalaSIEM Agent service...""
sudo launchctl load /Library/LaunchDaemons/com.athala.siem.agent.plist
sudo launchctl start com.athala.siem.agent

echo ""Agent installed successfully!""

# Cleanup
rm -f ""$TEMP_FILE""";
        }

        private string GenerateDockerScript(AthalaSIEM.Backend.Models.AgentDeploymentToken token)
        {
            return $@"# AthalaSIEM Agent Docker Deployment
# Generated: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC

version: '3.8'

services:
  athala-siem-agent:
    image: athala/siem-agent:latest
    container_name: athala-siem-agent
    restart: unless-stopped
    environment:
      - BACKEND_URL={GetBackendUrl()}
      - DEPLOYMENT_TOKEN={token.Token}
      - AGENT_NAME={{{{.Node.Hostname}}}}
    volumes:
      - /var/log:/host/var/log:ro
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
    network_mode: host
    privileged: true
    pid: host

# To deploy:
# 1. Save this as docker-compose.yml
# 2. Run: docker-compose up -d
# 3. Check status: docker-compose ps";
        }

        private string GenerateGenericScript(AthalaSIEM.Backend.Models.AgentDeploymentToken token)
        {
            return $@"# AthalaSIEM Agent Generic Deployment Instructions
# Generated: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC

Backend URL: {GetBackendUrl()}
Deployment Token: {token.Token}

Manual Installation Steps:
1. Download the appropriate agent binary for your platform
2. Configure the agent with the backend URL and deployment token
3. Start the agent service
4. Verify connection in the SIEM dashboard";
        }

        private List<string> GenerateDeploymentInstructions(string platform)
        {
            return platform.ToLowerInvariant() switch
            {
                "windows" => new List<string>
                {
                    "Run PowerShell as Administrator",
                    "Execute the deployment script",
                    "Verify service is running: Get-Service AthalaSIEMAgent",
                    "Check logs: Get-EventLog -LogName Application -Source AthalaSIEMAgent"
                },
                "linux" => new List<string>
                {
                    "Run script as root or with sudo",
                    "Verify service status: systemctl status athala-siem-agent",
                    "Check logs: journalctl -u athala-siem-agent",
                    "Configure firewall if needed for outbound connections"
                },
                "freebsd" => new List<string>
                {
                    "Run script as root",
                    "Verify service status: service athala-siem-agent status",
                    "Check logs: tail -f /var/log/athala-siem-agent.log",
                    "Configure firewall rules if necessary"
                },
                "macos" => new List<string>
                {
                    "Run script with sudo privileges",
                    "Verify service: sudo launchctl list | grep athala",
                    "Check logs: tail -f /var/log/athala-siem-agent.log",
                    "Allow network connections in Security preferences"
                },
                "docker" => new List<string>
                {
                    "Ensure Docker and Docker Compose are installed",
                    "Save the compose file and run docker-compose up -d",
                    "Check container status: docker-compose ps",
                    "View logs: docker-compose logs -f athala-siem-agent"
                },
                _ => new List<string>
                {
                    "Follow platform-specific installation guide",
                    "Configure backend URL and deployment token",
                    "Start the agent service",
                    "Verify connectivity"
                }
            };
        }

        private object GetConfigurationTemplate(string platform)
        {
            var baseConfig = new
            {
                backend_url = GetBackendUrl(),
                reporting_interval = 60,
                heartbeat_interval = 30,
                log_level = "INFO",
                collectors = new[]
                {
                    new { type = "syslog", enabled = true },
                    new { type = "file_integrity", enabled = false },
                    new { type = "container", enabled = false },
                    new { type = "cloud_services", enabled = false },
                    new { type = "database", enabled = false }
                }
            };

            return platform.ToLowerInvariant() switch
            {
                "windows" => new
                {
                    baseConfig.backend_url,
                    baseConfig.reporting_interval,
                    baseConfig.heartbeat_interval,
                    baseConfig.log_level,
                    collectors = new[]
                    {
                        new { type = "windows_eventlog", enabled = true },
                        new { type = "file_integrity", enabled = false },
                        new { type = "performance_counters", enabled = false },
                        new { type = "database", enabled = false }
                    }
                },
                "linux" => new
                {
                    baseConfig.backend_url,
                    baseConfig.reporting_interval,
                    baseConfig.heartbeat_interval,
                    baseConfig.log_level,
                    collectors = new[]
                    {
                        new { type = "syslog", enabled = true },
                        new { type = "file_integrity", enabled = false },
                        new { type = "container", enabled = false },
                        new { type = "cloud_services", enabled = false },
                        new { type = "database", enabled = false }
                    }
                },
                "freebsd" => new
                {
                    baseConfig.backend_url,
                    baseConfig.reporting_interval,
                    baseConfig.heartbeat_interval,
                    baseConfig.log_level,
                    collectors = new[]
                    {
                        new { type = "syslog", enabled = true },
                        new { type = "file_integrity", enabled = false }
                    }
                },
                "docker" => new
                {
                    baseConfig.backend_url,
                    baseConfig.reporting_interval,
                    baseConfig.heartbeat_interval,
                    baseConfig.log_level,
                    collectors = new[]
                    {
                        new { type = "container", enabled = true },
                        new { type = "syslog", enabled = true },
                        new { type = "cloud_services", enabled = false }
                    }
                },
                "cloud" => new
                {
                    baseConfig.backend_url,
                    baseConfig.reporting_interval,
                    baseConfig.heartbeat_interval,
                    baseConfig.log_level,
                    collectors = new[]
                    {
                        new { type = "cloud_services", enabled = true },
                        new { type = "container", enabled = false },
                        new { type = "syslog", enabled = true }
                    }
                },
                "industrial" => new
                {
                    baseConfig.backend_url,
                    baseConfig.reporting_interval,
                    baseConfig.heartbeat_interval,
                    baseConfig.log_level,
                    collectors = new[]
                    {
                        new { type = "iot", enabled = true },
                        new { type = "syslog", enabled = true },
                        new { type = "database", enabled = false }
                    }
                },
                "edge" => new
                {
                    baseConfig.backend_url,
                    baseConfig.reporting_interval,
                    baseConfig.heartbeat_interval,
                    baseConfig.log_level,
                    collectors = new[]
                    {
                        new { type = "iot", enabled = true },
                        new { type = "syslog", enabled = true },
                        new { type = "container", enabled = false }
                    }
                },
                _ => baseConfig
            };
        }

        private List<string> GetPrerequisites(string platform)
        {
            return platform.ToLowerInvariant() switch
            {
                "windows" => new List<string>
                {
                    ".NET Framework 4.8 or .NET 6.0 Runtime",
                    "Windows PowerShell 5.1 or PowerShell Core",
                    "Administrator privileges for installation",
                    "Outbound network access to SIEM backend"
                },
                "linux" => new List<string>
                {
                    "curl or wget for downloading",
                    "systemd for service management",
                    "Root privileges for installation",
                    "Outbound network access on configured ports"
                },
                "freebsd" => new List<string>
                {
                    "pkg package manager",
                    "Root privileges for installation",
                    "Network connectivity to backend",
                    "FreeBSD 12.0 or later"
                },
                "macos" => new List<string>
                {
                    "macOS 10.15 (Catalina) or later",
                    "Administrator privileges",
                    "Xcode Command Line Tools",
                    "Network access to backend"
                },
                "docker" => new List<string>
                {
                    "Docker Engine 20.10+",
                    "Docker Compose v2.0+",
                    "Privileged container support",
                    "Host network access"
                },
                _ => new List<string>
                {
                    "Compatible operating system",
                    "Network connectivity",
                    "Appropriate runtime dependencies"
                }
            };
        }

        private string GetDefaultConfiguration(string platform, AthalaSIEM.Backend.Models.AgentDeploymentToken token)
        {
            var config = JsonSerializer.Deserialize<Dictionary<string, object>>(token.Configuration ?? "{}") ?? new Dictionary<string, object>();
            config["platform"] = platform;
            config["deployment_token"] = token.Token ?? string.Empty;
            config["backend_url"] = GetBackendUrl();
            
            return JsonSerializer.Serialize(config);
        }

        private string GetBackendUrl()
        {
            var request = HttpContext.Request;
            return $"{request.Scheme}://{request.Host}";
        }

        private (string Url, string Checksum) GenerateInstallerPackage(string platform, AthalaSIEM.Backend.Models.AgentDeploymentToken token, object? customConfig)
        {
            // In a real implementation, this would generate or customize an installer package
            // For now, return placeholder URLs
            var baseUrl = GetBackendUrl();
            var url = $"{baseUrl}/downloads/agent/{platform}/athala-siem-agent-{Guid.NewGuid()}.{GetInstallerExtension(platform)}";
            var checksum = GenerateSecureToken().Substring(0, 64); // Mock checksum
            
            return (url, checksum);
        }

        private string GetInstallerExtension(string platform)
        {
            return platform.ToLowerInvariant() switch
            {
                "windows" => "msi",
                "linux" => "deb",
                "freebsd" => "txz",
                "macos" => "pkg",
                "docker" => "tar.gz",
                _ => "tar.gz"
            };
        }

        private List<string> GenerateInstallerInstructions(string platform)
        {
            return platform.ToLowerInvariant() switch
            {
                "windows" => new List<string>
                {
                    "Download the MSI installer package",
                    "Run as Administrator: msiexec /i installer.msi /quiet",
                    "Or double-click to run the GUI installer",
                    "Service will start automatically after installation"
                },
                "linux" => new List<string>
                {
                    "Download the DEB/RPM package",
                    "Install: sudo dpkg -i package.deb (or rpm -ivh package.rpm)",
                    "Start service: sudo systemctl start athala-siem-agent",
                    "Enable auto-start: sudo systemctl enable athala-siem-agent"
                },
                _ => new List<string>
                {
                    "Download the installer package",
                    "Follow platform-specific installation procedure",
                    "Configure and start the service"
                }
            };
        }
    }

    // Request/Response DTOs
    public class CreateAgentDeploymentTokenRequest
    {
        public string Name { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public string PlatformType { get; set; } = string.Empty;
        public DateTime? ExpiresAt { get; set; }
        public int? MaxUsage { get; set; }
        public Dictionary<string, object> Configuration { get; set; } = new();
    }

    public class AgentDeploymentTokenDto
    {
        public string Id { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public string PlatformType { get; set; } = string.Empty;
        public string? Token { get; set; } // Only included in creation response
        public DateTime? ExpiresAt { get; set; }
        public bool IsActive { get; set; }
        public int UsageCount { get; set; }
        public int? MaxUsage { get; set; }
        public DateTime CreatedAt { get; set; }
        public string? CreatedBy { get; set; }
        public DateTime? LastUsed { get; set; }
    }

    public class AgentDeploymentScriptResponse
    {
        public string Platform { get; set; } = string.Empty;
        public string Script { get; set; } = string.Empty;
        public List<string> Instructions { get; set; } = new();
        public object ConfigurationTemplate { get; set; } = new();
        public List<string> Prerequisites { get; set; } = new();
    }

    public class AgentRegistrationRequest
    {
        public string DeploymentToken { get; set; } = string.Empty;
        public string Hostname { get; set; } = string.Empty;
        public string IpAddress { get; set; } = string.Empty;
        public string Platform { get; set; } = string.Empty;
        public string OsVersion { get; set; } = string.Empty;
        public string AgentVersion { get; set; } = string.Empty;
        public Dictionary<string, string> SystemInfo { get; set; } = new();
    }

    public class AgentRegistrationResponse
    {
        public string AgentId { get; set; } = string.Empty;
        public string ApiKey { get; set; } = string.Empty;
        public string BackendUrl { get; set; } = string.Empty;
        public string Configuration { get; set; } = string.Empty;
        public int UpdateIntervalSeconds { get; set; }
        public int HeartbeatIntervalSeconds { get; set; }
    }

    public class AgentConfigurationResponse
    {
        public string AgentId { get; set; } = string.Empty;
        public string Configuration { get; set; } = string.Empty;
        public DateTime LastUpdated { get; set; }
        public bool RequiresRestart { get; set; }
    }

    public class UpdateAgentConfigurationRequest
    {
        public Dictionary<string, object> Configuration { get; set; } = new();
        public bool RequiresRestart { get; set; } = false;
    }

    public class DeploymentStatistics
    {
        public int TotalTokens { get; set; }
        public int ActiveTokens { get; set; }
        public int TotalDeployments { get; set; }
        public int OnlineAgents { get; set; }
        public int OfflineAgents { get; set; }
        public List<PlatformCount> PlatformDistribution { get; set; } = new();
        public List<DeploymentTrend> RecentDeployments { get; set; } = new();
    }

    public class PlatformCount
    {
        public string Platform { get; set; } = string.Empty;
        public int Count { get; set; }
    }

    public class DeploymentTrend
    {
        public DateTime Date { get; set; }
        public int Count { get; set; }
    }

    public class GenerateInstallerRequest
    {
        public string TokenId { get; set; } = string.Empty;
        public string Platform { get; set; } = string.Empty;
        public Dictionary<string, object>? CustomConfiguration { get; set; }
    }

    public class AgentInstallerResponse
    {
        public string Platform { get; set; } = string.Empty;
        public string InstallerUrl { get; set; } = string.Empty;
        public string ChecksumSha256 { get; set; } = string.Empty;
        public DateTime ExpiresAt { get; set; }
        public List<string> Instructions { get; set; } = new();
    }
} 
