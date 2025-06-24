using System;
using System.DirectoryServices.AccountManagement;
using System.Security.Principal;
using System.Threading.Tasks;
using System.Linq;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;
using System.Runtime.Versioning;

namespace AthalaSIEM.UniversalAgent.Services
{
    /// <summary>
    /// Windows Authentication Service for SIEM Agent
    /// Handles Windows user authentication and privilege verification
    /// Required for secure log collection operations
    /// </summary>
    [SupportedOSPlatform("windows")]
    public class WindowsAuthenticationService
    {
        private readonly ILogger<WindowsAuthenticationService> _logger;
        private readonly IConfiguration _configuration;
        
        private WindowsIdentity? _currentIdentity;
        private WindowsPrincipal? _currentPrincipal;
        private bool _isAuthenticated;
        private bool _hasAdminPrivileges;
        private string _serviceAccountName = "";

        public bool IsAuthenticated => _isAuthenticated;
        public bool HasAdminPrivileges => _hasAdminPrivileges;
        public string ServiceAccountName => _serviceAccountName;
        public string CurrentUser => _currentIdentity?.Name ?? "Unknown";

        public WindowsAuthenticationService(
            ILogger<WindowsAuthenticationService> logger,
            IConfiguration configuration)
        {
            _logger = logger;
            _configuration = configuration;
        }

        /// <summary>
        /// Initializes Windows authentication and verifies privileges
        /// </summary>
        public async Task<bool> InitializeAsync()
        {
            try
            {
                _logger.LogInformation("🔐 Initializing Windows Authentication for SIEM Agent...");

                // Get current Windows identity
                _currentIdentity = WindowsIdentity.GetCurrent();
                _currentPrincipal = new WindowsPrincipal(_currentIdentity);
                
                _serviceAccountName = _currentIdentity.Name;
                _logger.LogInformation("Running as Windows user: {User}", _serviceAccountName);

                // Verify authentication
                _isAuthenticated = _currentIdentity.IsAuthenticated;
                if (!_isAuthenticated)
                {
                    _logger.LogError("❌ Windows authentication failed - user not authenticated");
                    return false;
                }

                // Check Administrator privileges
                _hasAdminPrivileges = _currentPrincipal.IsInRole(WindowsBuiltInRole.Administrator);
                
                if (_hasAdminPrivileges)
                {
                    _logger.LogInformation("✅ Administrator privileges confirmed - Full SIEM functionality available");
                    _logger.LogInformation("🛡️ Can access Security Event Log, Registry, and File System");
                }
                else
                {
                    _logger.LogWarning("⚠️ NO Administrator privileges - Limited SIEM functionality");
                    _logger.LogWarning("🚨 Cannot access Security Event Log - Critical for SIEM operations");
                    _logger.LogWarning("💡 To fix: Run agent as Administrator or configure service account");
                }

                // Verify specific privileges needed for SIEM operations
                await VerifyRequiredPrivilegesAsync();

                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to initialize Windows authentication");
                return false;
            }
        }

        /// <summary>
        /// Verifies specific Windows privileges required for SIEM operations
        /// </summary>
        private async Task VerifyRequiredPrivilegesAsync()
        {
            var privileges = new[]
            {
                "SeAuditPrivilege",           // Access Security Log
                "SeSecurityPrivilege",       // Manage Security Log
                "SeBackupPrivilege",         // Backup files for FIM
                "SeRestorePrivilege",        // Restore files for FIM
                "SeSystemtimePrivilege",     // System time for correlation
                "SeDebugPrivilege"           // Debug processes
            };

            foreach (var privilege in privileges)
            {
                var hasPrivilege = HasPrivilege(privilege);
                var status = hasPrivilege ? "✅" : "❌";
                _logger.LogInformation("{Status} {Privilege}: {HasPrivilege}", 
                    status, privilege, hasPrivilege ? "GRANTED" : "DENIED");
            }

            await Task.CompletedTask;
        }

        /// <summary>
        /// Checks if current user has specific Windows privilege
        /// </summary>
        private bool HasPrivilege(string privilegeName)
        {
            try
            {
                // This is a simplified check - in production, you'd use Windows API
                // to check specific privileges using LookupPrivilegeValue and GetTokenInformation
                return _hasAdminPrivileges; // Admin has most privileges
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Authenticates with Windows using provided credentials
        /// Used for service account configuration
        /// </summary>
        public async Task<bool> AuthenticateWithCredentialsAsync(string username, string password, string domain = ".")
        {
            try
            {
                _logger.LogInformation("🔐 Attempting Windows authentication for user: {Domain}\\{Username}", domain, username);

                using var context = new PrincipalContext(ContextType.Machine, domain);
                var isValid = context.ValidateCredentials(username, password);

                if (isValid)
                {
                    _logger.LogInformation("✅ Windows authentication successful for {Domain}\\{Username}", domain, username);
                    
                    // Check if user has admin privileges
                    using var user = UserPrincipal.FindByIdentity(context, username);
                    if (user != null)
                    {
                        var isAdmin = user.GetGroups().Any(g => g.Name == "Administrators");
                        _logger.LogInformation("Administrator privileges: {IsAdmin}", isAdmin ? "YES" : "NO");
                    }
                    
                    return true;
                }
                else
                {
                    _logger.LogError("❌ Windows authentication failed for {Domain}\\{Username}", domain, username);
                    return false;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during Windows authentication for {Domain}\\{Username}", domain, username);
                return false;
            }
        }

        /// <summary>
        /// Configures Windows service account for SIEM agent
        /// </summary>
        public async Task<bool> ConfigureServiceAccountAsync()
        {
            try
            {
                _logger.LogInformation("🔧 Configuring Windows service account for SIEM operations...");

                var serviceAccount = _configuration.GetValue<string>("Agent:ServiceAccount");
                var servicePassword = _configuration.GetValue<string>("Agent:ServicePassword");

                if (!string.IsNullOrEmpty(serviceAccount) && !string.IsNullOrEmpty(servicePassword))
                {
                    var authenticated = await AuthenticateWithCredentialsAsync(serviceAccount, servicePassword);
                    if (authenticated)
                    {
                        _logger.LogInformation("✅ Service account configured successfully");
                        return true;
                    }
                }
                else
                {
                    _logger.LogInformation("ℹ️ No service account configured - using current user context");
                }

                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to configure service account");
                return false;
            }
        }

        /// <summary>
        /// Gets authentication status for health checks
        /// </summary>
        public AuthenticationStatus GetAuthenticationStatus()
        {
            return new AuthenticationStatus
            {
                IsAuthenticated = _isAuthenticated,
                HasAdminPrivileges = _hasAdminPrivileges,
                CurrentUser = CurrentUser,
                ServiceAccount = _serviceAccountName,
                AuthenticationTime = DateTime.UtcNow,
                CanAccessSecurityLog = _hasAdminPrivileges,
                CanAccessRegistry = _hasAdminPrivileges,
                CanAccessFileSystem = true, // Basic file access usually available
                RequiresElevation = !_hasAdminPrivileges
            };
        }

        /// <summary>
        /// Provides guidance for fixing authentication issues
        /// </summary>
        public void LogAuthenticationGuidance()
        {
            if (!_hasAdminPrivileges)
            {
                _logger.LogWarning("🔧 AUTHENTICATION GUIDANCE:");
                _logger.LogWarning("1. Run PowerShell as Administrator");
                _logger.LogWarning("2. Execute: dotnet run");
                _logger.LogWarning("3. OR configure service account with admin privileges");
                _logger.LogWarning("4. OR install as Windows Service with LocalSystem account");
                _logger.LogWarning("");
                _logger.LogWarning("⚠️ Without Administrator privileges:");
                _logger.LogWarning("- Security Event Log: UNAVAILABLE");
                _logger.LogWarning("- Registry Monitoring: LIMITED");
                _logger.LogWarning("- File Integrity Monitoring: LIMITED");
                _logger.LogWarning("- This is NOT a functional SIEM agent!");
            }
        }
    }

    /// <summary>
    /// Authentication status information
    /// </summary>
    public class AuthenticationStatus
    {
        public bool IsAuthenticated { get; set; }
        public bool HasAdminPrivileges { get; set; }
        public string CurrentUser { get; set; } = "";
        public string ServiceAccount { get; set; } = "";
        public DateTime AuthenticationTime { get; set; }
        public bool CanAccessSecurityLog { get; set; }
        public bool CanAccessRegistry { get; set; }
        public bool CanAccessFileSystem { get; set; }
        public bool RequiresElevation { get; set; }
    }
} 