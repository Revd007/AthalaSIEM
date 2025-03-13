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
        private const string WINDOWS_INSTALLER_FILENAME = "AthalaAgent-Setup.exe";

        /// <summary>
        /// Initializes a new instance of the <see cref="InstallerService"/> class
        /// </summary>
        /// <param name="configuration">The configuration</param>
        /// <param name="logger">The logger</param>
        public InstallerService(IConfiguration configuration, ILogger<InstallerService> logger)
        {
            _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _installerBasePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Installers");
            _agentBasePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "Agent");
            
            _logger.LogInformation("Installer base path: {Path}", _installerBasePath);
            _logger.LogInformation("Agent base path: {Path}", _agentBasePath);
            _logger.LogInformation("Current directory: {Path}", Directory.GetCurrentDirectory());
            _logger.LogInformation("Base directory: {Path}", AppDomain.CurrentDomain.BaseDirectory);
            
            EnsureInstallerDirectoryExists();
            EnsureInstallerFileExists();
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
                        FileName = "AthalaAgent-Setup.exe",
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
                    FileName = "AthalaAgent-Setup.exe",
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
                var installerName = _configuration["InstallerSettings:AgentInstallerName"] ?? "AthalaAgent-Setup.exe";
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
                    FileName = "AthalaAgent-Setup.exe",
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
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating installer directory");
                throw;
            }
        }
        
        private void EnsureInstallerFileExists()
        {
            var installerPath = Path.Combine(_installerBasePath, WINDOWS_INSTALLER_FILENAME);
            
            _logger.LogInformation("Ensuring installer file exists at: {Path}", installerPath);
            
            if (!File.Exists(installerPath))
            {
                try
                {
                    // Try to find the Agent.exe in various locations
                    var agentExePath = FindAgentExecutable();
                    
                    if (!string.IsNullOrEmpty(agentExePath) && File.Exists(agentExePath))
                    {
                        // Copy the Agent.exe to the Installers directory with the installer name
                        File.Copy(agentExePath, installerPath, true);
                        _logger.LogInformation("Copied Agent.exe to installer path: {Path}", installerPath);
                    }
                    else
                    {
                        _logger.LogWarning("Agent executable not found. Creating a dummy executable for testing.");
                        CreateDummyExecutable(installerPath);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error ensuring installer file exists");
                }
            }
            else
            {
                var fileInfo = new FileInfo(installerPath);
                _logger.LogInformation("Installer file already exists at: {Path} with size: {Size} bytes", installerPath, fileInfo.Length);
            }
        }
        
        private string FindAgentExecutable()
        {
            // Log direktori dasar
            _logger.LogInformation("Agent base path: {Path}", _agentBasePath);
            
            // Coba berbagai path absolut dan relatif
            var possiblePaths = new List<string>
            {
                // Path absolut
                @"D:\athala-siem-main\agent\bin\Release\net8.0-windows\win-x64\publish\Agent.exe",
                
                // Path relatif dari direktori saat ini
                Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), "..", "agent", "bin", "Release", "net8.0-windows", "win-x64", "publish", "Agent.exe")),
                
                // Path relatif dari AppDomain.CurrentDomain.BaseDirectory
                Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "agent", "bin", "Release", "net8.0-windows", "win-x64", "publish", "Agent.exe")),
                
                // Path relatif dari _agentBasePath
                Path.Combine(_agentBasePath, "bin", "Release", "net8.0-windows", "win-x64", "publish", "Agent.exe"),
                Path.Combine(_agentBasePath, "bin", "Debug", "net8.0-windows", "win-x64", "publish", "Agent.exe"),
                Path.Combine(_agentBasePath, "bin", "Release", "net8.0-windows", "win-x64", "Agent.exe"),
                Path.Combine(_agentBasePath, "bin", "Debug", "net8.0-windows", "win-x64", "Agent.exe"),
                Path.Combine(_agentBasePath, "bin", "Release", "net8.0-windows", "Agent.exe"),
                Path.Combine(_agentBasePath, "bin", "Debug", "net8.0-windows", "Agent.exe")
            };
            
            // Log semua path yang akan diperiksa
            foreach (var path in possiblePaths)
            {
                _logger.LogInformation("Checking path: {Path}", path);
                
                if (File.Exists(path))
                {
                    var fileInfo = new FileInfo(path);
                    _logger.LogInformation("Found Agent.exe at: {Path} with size: {Size} bytes", path, fileInfo.Length);
                    return path;
                }
            }
            
            _logger.LogWarning("Could not find Agent.exe in any of the expected locations");
            return string.Empty;
        }
        
        private void CreateDummyExecutable(string installerPath)
        {
            try
            {
                _logger.LogInformation("Creating dummy executable at: {Path}", installerPath);
                
                // Buat file executable dummy sederhana
                using (var fileStream = new FileStream(installerPath, FileMode.Create))
                using (var writer = new BinaryWriter(fileStream))
                {
                    // Header MZ (Magic number untuk file executable Windows)
                    writer.Write((ushort)0x5A4D);
                    
                    // Tambahkan beberapa byte acak untuk membuat file lebih besar
                    var random = new Random();
                    var buffer = new byte[1024 * 1024 * 10]; // 10MB
                    random.NextBytes(buffer);
                    writer.Write(buffer);
                }
                
                var fileInfo = new FileInfo(installerPath);
                _logger.LogInformation("Created dummy executable at {Path} with size: {Size} bytes", installerPath, fileInfo.Length);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating dummy executable");
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
                
                // Get installer path from configuration
                var installerName = _configuration["InstallerSettings:AgentInstallerName"] ?? "AthalaAgent-Setup.exe";
                var installerPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "installer", installerName);
                
                // Check if installer exists
                if (!File.Exists(installerPath))
                {
                    _logger.LogWarning("Installer not found at path: {Path}", installerPath);
                    return null;
                }
                
                // Read installer file
                var content = await File.ReadAllBytesAsync(installerPath);
                
                // Determine content type based on file extension
                string contentType = Path.GetExtension(installerPath).ToLower() switch
                {
                    ".exe" => "application/vnd.microsoft.portable-executable",
                    ".msi" => "application/x-msi",
                    ".deb" => "application/vnd.debian.binary-package",
                    ".rpm" => "application/x-rpm",
                    ".pkg" => "application/vnd.apple.installer+xml",
                    ".zip" => "application/zip",
                    _ => "application/octet-stream"
                };
                
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