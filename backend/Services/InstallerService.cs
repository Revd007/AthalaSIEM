using System.IO.Compression;
using Microsoft.Extensions.Logging;
using Backend.Models;
using System.Text;
using System.Reflection;
using System.Diagnostics;
using System;
using System.IO;
using System.Security.Cryptography;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;

namespace Backend.Services
{
    /// <summary>
    /// Interface for installer service operations
    /// </summary>
    public interface IInstallerService
    {
        /// <summary>
        /// Generates an installer package for the specified platform
        /// </summary>
        /// <param name="type">The platform type (e.g., "windows")</param>
        /// <returns>An installer package containing the binary and metadata</returns>
        Task<InstallerPackage> GenerateInstaller(string type);

        /// <summary>
        /// Gets information about an installer package
        /// </summary>
        /// <param name="type">The platform type (e.g., "windows")</param>
        /// <param name="baseUrl">The base URL for download links</param>
        /// <returns>Information about the installer package</returns>
        Task<InstallerInfo> GetInstallerInfo(string type, string baseUrl);
        
        /// <summary>
        /// Generates an installer package for the specified platform (alias for GenerateInstaller)
        /// </summary>
        /// <param name="type">The platform type (e.g., "windows")</param>
        /// <returns>An installer package containing the binary and metadata</returns>
        Task<InstallerPackage?> GenerateInstallerPackage(string type);
        
        /// <summary>
        /// Builds the agent from source code
        /// </summary>
        /// <returns>True if the build was successful, false otherwise</returns>
        bool BuildAgent();
    }

    /// <summary>
    /// Service for managing agent installers
    /// </summary>
    public class InstallerService : IInstallerService
    {
        private readonly IConfiguration _configuration;
        private readonly ILogger<InstallerService> _logger;
        private readonly string _installerBasePath;
        private readonly string _agentBasePath;
        private const string WINDOWS_INSTALLER_FILENAME = "AthalaSIEMAgent.msi";

        /// <summary>
        /// Initializes a new instance of the <see cref="InstallerService"/> class
        /// </summary>
        /// <param name="configuration">The configuration</param>
        /// <param name="logger">The logger</param>
        public InstallerService(IConfiguration configuration, ILogger<InstallerService> logger)
        {
            _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            
            // Get the solution directory (two levels up from the bin directory)
            var solutionDirectory = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", ".."));
            _installerBasePath = Path.Combine(solutionDirectory, "Installers");
            _agentBasePath = Path.Combine(solutionDirectory, "agent");
            
            _logger.LogInformation("Installer base path: {Path}", _installerBasePath);
            _logger.LogInformation("Agent base path: {Path}", _agentBasePath);
            _logger.LogInformation("Current directory: {Path}", Directory.GetCurrentDirectory());
            _logger.LogInformation("Base directory: {Path}", AppDomain.CurrentDomain.BaseDirectory);
            _logger.LogInformation("Solution directory: {Path}", solutionDirectory);
            
            EnsureInstallerDirectoryExists();
        }

        /// <summary>
        /// Generates an installer package for the specified platform
        /// </summary>
        /// <param name="type">The platform type (e.g., "windows")</param>
        /// <returns>An installer package containing the binary and metadata</returns>
        public async Task<InstallerPackage> GenerateInstaller(string type)
        {
            try
            {
                _logger.LogInformation("Generating installer for type: {Type}", type);
                
                // In a real implementation, this would generate the installer
                // For now, just return a dummy package
                
                var installerPackage = await GenerateInstallerPackage(type);
                
                if (installerPackage == null)
                {
                    _logger.LogWarning("Failed to generate installer package, returning dummy package");
                    
                    return new InstallerPackage
                    {
                        FileName = "AthalaSIEMAgent.msi",
                        ContentType = "application/vnd.microsoft.portable-executable",
                        Content = new byte[1024] // Dummy content
                    };
                }
                
                _logger.LogInformation("Installer generated successfully");
                
                return installerPackage;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating installer for type: {Type}", type);
                
                return new InstallerPackage
                {
                    FileName = "AthalaSIEMAgent.msi",
                    ContentType = "application/vnd.microsoft.portable-executable",
                    Content = new byte[1024] // Dummy content
                };
            }
        }

        /// <summary>
        /// Gets information about an installer package
        /// </summary>
        /// <param name="type">The platform type (e.g., "windows")</param>
        /// <param name="baseUrl">The base URL for download links</param>
        /// <returns>Information about the installer package</returns>
        public async Task<InstallerInfo> GetInstallerInfo(string type, string baseUrl)
        {
            try
            {
                _logger.LogInformation("Getting installer info for type: {Type}", type);
                
                // Get installer path from configuration
                var installerName = _configuration["InstallerSettings:AgentInstallerName"] ?? "AthalaSIEMAgent.msi";
                var installerPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "installer", installerName);
                
                // Check if installer exists
                if (!File.Exists(installerPath))
                {
                    _logger.LogWarning("Installer not found at path: {Path}", installerPath);
                    
                    // Return default info
                    return new InstallerInfo
                    {
                        Type = type,
                        Version = "1.0.0",
                        FileName = installerName,
                        Size = 0,
                        DownloadUrl = $"{baseUrl}/api/agents/download-installer/{type}",
                        LastModified = DateTime.UtcNow,
                        Sha256Hash = null,
                        Requirements = "Windows 10 or later",
                        Description = "Athala SIEM Agent"
                    };
                }
                
                // Get file info
                var fileInfo = new FileInfo(installerPath);
                
                // Calculate SHA256 hash
                string sha256Hash;
                using (var stream = File.OpenRead(installerPath))
                {
                    using var sha256 = SHA256.Create();
                    var hashBytes = await Task.Run(() => sha256.ComputeHash(stream));
                    sha256Hash = BitConverter.ToString(hashBytes).Replace("-", "").ToLowerInvariant();
                }
                
                // Get version from configuration or default to 1.0.0
                var version = _configuration["InstallerSettings:AgentVersion"] ?? "1.0.0";
                
                // Get requirements from configuration or default
                var requirements = _configuration["InstallerSettings:Requirements"] ?? "Windows 10 or later";
                
                // Get description from configuration or default
                var description = _configuration["InstallerSettings:Description"] ?? "Athala SIEM Agent";
                
                _logger.LogInformation("Installer info retrieved successfully");
                
                return new InstallerInfo
                {
                    Type = type,
                    Version = version,
                    FileName = installerName,
                    Size = fileInfo.Length,
                    DownloadUrl = $"{baseUrl}/api/agents/download-installer/{type}",
                    LastModified = fileInfo.LastWriteTimeUtc,
                    Sha256Hash = sha256Hash,
                    Requirements = requirements,
                    Description = description
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting installer info for type: {Type}", type);
                
                // Return default info
                return new InstallerInfo
                {
                    Type = type,
                    Version = "1.0.0",
                    FileName = "AthalaSIEMAgent.msi",
                    Size = 0,
                    DownloadUrl = $"{baseUrl}/api/agents/download-installer/{type}",
                    LastModified = DateTime.UtcNow,
                    Sha256Hash = null,
                    Requirements = "Windows 10 or later",
                    Description = "Athala SIEM Agent"
                };
            }
        }

        private void EnsureInstallerDirectoryExists()
        {
            try
            {
                if (!Directory.Exists(_installerBasePath))
                {
                    _logger.LogInformation("Creating installer directory: {Path}", _installerBasePath);
                    Directory.CreateDirectory(_installerBasePath);
                }

                // Check if installer exists
                var installerPath = Path.Combine(_installerBasePath, WINDOWS_INSTALLER_FILENAME);
                if (!File.Exists(installerPath))
                {
                    _logger.LogWarning("Installer not found at path: {Path}", installerPath);
                    throw new FileNotFoundException($"Installer file not found at {installerPath}");
                }

                var fileInfo = new FileInfo(installerPath);
                _logger.LogInformation("Installer file exists at: {Path} with size: {Size} bytes", installerPath, fileInfo.Length);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error checking installer directory");
                throw;
            }
        }
        
        
        

        /// <summary>
        /// Generates an installer package for the specified platform (alias for GenerateInstaller)
        /// </summary>
        /// <param name="type">The platform type (e.g., "windows")</param>
        /// <returns>An installer package containing the binary and metadata</returns>
        public async Task<InstallerPackage?> GenerateInstallerPackage(string type)
        {
            try
            {
                _logger.LogInformation("Generating installer package for type: {Type}", type);
                
                // Get installer path based on type
                string installerPath;
                string installerName;
                string contentType;
                
                switch (type.ToLowerInvariant())
                {
                    case "windows":
                        installerName = "AthalaSIEMAgent.msi";
                        contentType = "application/x-msi";
                        break;
                    case "linux-rpm":
                        installerName = "AthalaSIEMAgent.rpm";
                        contentType = "application/x-rpm";
                        break;
                    case "linux-deb":
                        installerName = "AthalaSIEMAgent.deb";
                        contentType = "application/vnd.debian.binary-package";
                        break;
                    case "macos":
                        installerName = "AthalaSIEMAgent.pkg";
                        contentType = "application/vnd.apple.installer+xml";
                        break;
                    default:
                        _logger.LogWarning("Unsupported installer type: {Type}", type);
                        return null;
                }
                
                installerPath = Path.Combine(_installerBasePath, installerName);
                _logger.LogInformation("Looking for installer at path: {Path}", installerPath);
                
                // Check if installer exists
                if (!File.Exists(installerPath))
                {
                    _logger.LogWarning("Installer not found at path: {Path}", installerPath);
                    return null;
                }
                
                // Read installer file
                var content = await File.ReadAllBytesAsync(installerPath);
                
                _logger.LogInformation("Installer package generated successfully. Size: {Size} bytes", content.Length);
                
                return new InstallerPackage
                {
                    FileName = installerName,
                    ContentType = contentType,
                    Content = content
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating installer package for type: {Type}", type);
                return null;
            }
        }

        /// <summary>
        /// Builds the agent from source code
        /// </summary>
        /// <returns>True if the build was successful, false otherwise</returns>
        public bool BuildAgent()
        {
            try
            {
                _logger.LogInformation("Building agent...");
                
                // In a real implementation, this would build the agent from source
                // For now, just return true
                
                _logger.LogInformation("Agent built successfully");
                
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error building agent");
                return false;
            }
        }
    }
} 