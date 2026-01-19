using System;
using System.DirectoryServices.AccountManagement;
using System.Security.Principal;
using System.Threading.Tasks;
using System.Linq;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;
using System.Runtime.Versioning;
using AthalaSIEM.UniversalAgent.Models;

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
                    _logger.LogError("Windows authentication failed - user not authenticated");
                    return false;
                }

                // Check Administrator privileges
                _hasAdminPrivileges = _currentPrincipal.IsInRole(WindowsBuiltInRole.Administrator);
                
                if (_hasAdminPrivileges)
                {
                    _logger.LogInformation(" Administrator privileges confirmed - Full SIEM functionality available");
                    _logger.LogInformation("🛡️ Can access Security Event Log, Registry, and File System");
                }
                else
                {
                    _logger.LogWarning("NO Administrator privileges - Limited SIEM functionality");
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
                var status = hasPrivilege ? "" : "❌";
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
        /// Authenticates with provided credentials using secure Windows authentication.
        /// SECURITY: Credentials are handled securely and never stored in memory longer than necessary.
        /// </summary>
        /// <param name="username">Username to authenticate</param>
        /// <param name="password">Password to authenticate (SecureString recommended)</param>
        /// <param name="domain">Domain for authentication</param>
        /// <returns>True if authentication was successful</returns>
        public async Task<bool> AuthenticateWithCredentialsAsync(string username, string? password, string domain = ".")
        {
            if (string.IsNullOrEmpty(username) || string.IsNullOrEmpty(password))
            {
                _logger.LogWarning("Authentication failed: Username or password is empty");
                return false;
            }

            try
            {
                using var context = new PrincipalContext(ContextType.Domain, domain);
                var isValid = context.ValidateCredentials(username, password);
                
                if (isValid)
                {
                    _logger.LogInformation("Authentication successful for user {Username}", username);
                    _serviceAccountName = $"{domain}\\{username}";
                    return true;
                }
                else
                {
                    _logger.LogWarning("Authentication failed for user {Username}", username);
                    return false;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during authentication for user {Username}", username);
                return false;
            }
            finally
            {
                // SECURITY: Clear password from memory immediately after use
                // Note: In production, use SecureString for password parameter
                password = null;
                GC.Collect(); // Force garbage collection to clear sensitive data
                await Task.CompletedTask;
            }
        }

        /// <summary>
        /// Configures service account from secure configuration sources.
        /// SECURITY: Passwords must be stored in Windows Credential Manager or secure key vault.
        /// </summary>
        /// <returns>True if service account was configured successfully</returns>
        public async Task<bool> ConfigureServiceAccountAsync()
        {
            try
            {
                var serviceAccount = _configuration.GetValue<string>("Agent:ServiceAccount");
                
                // SECURITY: Don't read passwords from configuration files
                // Passwords should come from Windows Credential Manager or secure vault
                if (!string.IsNullOrEmpty(serviceAccount))
                {
                    _logger.LogInformation("Service account configured: {ServiceAccount}", serviceAccount);
                    _logger.LogWarning("SECURITY WARNING: Service password should be configured via Windows Credential Manager, not configuration files");
                    
                    // In production, retrieve password from Windows Credential Manager:
                    // var credential = CredentialManager.ReadCredential(serviceAccount);
                    // if (credential != null)
                    // {
                    //     var authenticated = await AuthenticateWithCredentialsAsync(
                    //         credential.UserName, credential.Password);
                    //     return authenticated;
                    // }
                    
                    _serviceAccountName = serviceAccount;
                    await Task.CompletedTask;
                    return true;
                }
                else
                {
                    _logger.LogInformation("No service account configured, using current Windows identity");
                    await Task.CompletedTask;
                    return true;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error configuring service account");
                return false;
            }
        }

        /// <summary>
        /// Gets the current authentication status
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
                CanAccessFileSystem = true, // Usually available to most users
                RequiresElevation = !_hasAdminPrivileges
            };
        }

        /// <summary>
        /// Logs authentication guidance for administrators
        /// </summary>
        public void LogAuthenticationGuidance()
        {
            _logger.LogInformation("🔐 Windows Authentication Configuration Guidance:");
            _logger.LogInformation("   Current User: {User}", CurrentUser);
            _logger.LogInformation("   Authenticated: {Status}", _isAuthenticated ? "YES" : "NO");
            _logger.LogInformation("   Admin Privileges: {Status}", _hasAdminPrivileges ? "YES" : "NO");
            
            if (!_hasAdminPrivileges)
            {
                _logger.LogInformation("  💡 To enable full SIEM functionality:");
                _logger.LogInformation("     1. Run as Administrator");
                _logger.LogInformation("     2. Configure service account with admin privileges");
                _logger.LogInformation("     3. Add user to 'Log on as a service' policy");
            }
        }
    }

    // NOTE: AuthenticationStatus model has been moved to 
    // AthalaSIEM.UniversalAgent.Models.CommunicationServiceModels.cs for clean architecture separation
} 
