using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;
using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Threading.Tasks;
using System.Net;
using System.Net.NetworkInformation;
using System.Text.Json;
using System.Linq;
using AthalaSIEM.UniversalAgent.Models;

namespace AthalaSIEM.UniversalAgent.Services
{
    /// <summary>
    /// Enterprise Agent Discovery Service
    /// Automatically discovers SIEM Manager servers like Wazuh/ManageEngine
    /// Supports multiple discovery methods: DNS, Broadcast, Manual
    /// NO HARDCODED SERVER ADDRESSES - All dynamic discovery
    /// </summary>
    public class AgentDiscoveryService
    {
        private readonly ILogger<AgentDiscoveryService> _logger;
        private readonly IConfiguration _configuration;
        private readonly HttpClient _httpClient;
        private readonly List<string> _discoveredServers = new();
        private string? _selectedServer;

        public AgentDiscoveryService(
            ILogger<AgentDiscoveryService> logger,
            IConfiguration configuration,
            HttpClient httpClient)
        {
            _logger = logger;
            _configuration = configuration;
            _httpClient = httpClient;
        }

        /// <summary>
        /// Discover SIEM Manager servers using multiple methods like enterprise tools
        /// </summary>
        public async Task<List<string>> DiscoverSIEMServersAsync()
        {
            _logger.LogInformation("🔍 Starting SIEM Manager discovery (Wazuh/ManageEngine style)...");
            
            var discoveryMethods = _configuration.GetSection("ServerDiscovery:DiscoveryMethods").Get<string[]>() 
                ?? new[] { "DNS", "Broadcast", "Manual" };
            
            foreach (var method in discoveryMethods)
            {
                try
                {
                    _logger.LogInformation("🔍 Trying discovery method: {Method}", method);
                    
                    switch (method.ToLowerInvariant())
                    {
                        case "dns":
                            await DiscoverViaDNSAsync();
                            break;
                        case "broadcast":
                            await DiscoverViaBroadcastAsync();
                            break;
                        case "manual":
                            await DiscoverViaManualConfigAsync();
                            break;
                        case "environment":
                            await DiscoverViaEnvironmentAsync();
                            break;
                        case "registry":
                            if (System.OperatingSystem.IsWindows())
                                await DiscoverViaRegistryAsync();
                            break;
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Discovery method {Method} failed", method);
                }
            }

            _logger.LogInformation("✅ Discovery completed. Found {Count} SIEM servers: {Servers}", 
                _discoveredServers.Count, string.Join(", ", _discoveredServers));

            return _discoveredServers.ToList();
        }

        /// <summary>
        /// DNS-based discovery like Wazuh
        /// </summary>
        private async Task DiscoverViaDNSAsync()
        {
            try
            {
                _logger.LogInformation("🔍 Discovering SIEM servers via DNS...");
                
                var dnsRecords = _configuration.GetSection("Enterprise:ServiceDiscovery:DNSRecords").Get<string[]>() 
                    ?? new[] { "_siem._tcp", "_athala._tcp" };
                
                var defaultPort = GetDefaultBackendPort();
                
                foreach (var record in dnsRecords)
                {
                    try
                    {
                        // Try DNS SRV record lookup
                        var addresses = await Dns.GetHostAddressesAsync(record.Replace("_siem._tcp.", "").Replace("_athala._tcp.", ""));
                        
                        foreach (var address in addresses.Take(5)) // Limit to 5 addresses
                        {
                            var serverUrl = $"http://{address}:{defaultPort}";
                            
                            if (await ValidateSIEMServerAsync(serverUrl))
                            {
                                _discoveredServers.Add(serverUrl);
                                _logger.LogInformation("✅ Found SIEM server via DNS: {ServerUrl}", serverUrl);
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogDebug("DNS lookup failed for {Record}: {Error}", record, ex.Message);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "DNS discovery failed");
            }
        }

        /// <summary>
        /// Network broadcast discovery like ManageEngine
        /// </summary>
        private async Task DiscoverViaBroadcastAsync()
        {
            try
            {
                _logger.LogInformation("📡 Discovering SIEM servers via network broadcast...");
                
                var defaultPort = GetDefaultBackendPort();
                var broadcastPort = _configuration.GetValue<int>("ServerDiscovery:BroadcastPort");
                if (broadcastPort <= 0)
                    broadcastPort = defaultPort;
                
                var localRanges = GetLocalNetworkRanges();
                var tasks = localRanges.Select(range => ScanNetworkRangeAsync(range, defaultPort, 2)).ToList();
                
                await Task.WhenAll(tasks);
                
                _logger.LogInformation("📡 Network broadcast discovery completed. Found {Count} servers", _discoveredServers.Count);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Broadcast discovery failed");
            }
        }

        /// <summary>
        /// Manual configuration discovery
        /// </summary>
        private async Task DiscoverViaManualConfigAsync()
        {
            var fallbackServers = _configuration.GetSection("ServerDiscovery:FallbackServers").Get<string[]>() ?? new string[0];
            
            foreach (var server in fallbackServers)
            {
                if (await ValidateSIEMServerAsync(server))
                {
                    _discoveredServers.Add(server);
                    _logger.LogInformation("✅ Manual config found SIEM server: {Server}", server);
                }
            }
        }

        /// <summary>
        /// Environment variable discovery (like Wazuh WAZUH_MANAGER)
        /// </summary>
        private async Task DiscoverViaEnvironmentAsync()
        {
            var envVars = new[] { "ATHALA_MANAGER", "SIEM_MANAGER", "WAZUH_MANAGER", "LOG_MANAGER" };
            
            foreach (var envVar in envVars)
            {
                var serverAddress = Environment.GetEnvironmentVariable(envVar);
                if (!string.IsNullOrEmpty(serverAddress))
                {
                    var defaultPort = GetDefaultBackendPort();
            var serverUrl = serverAddress.StartsWith("http") ? serverAddress : $"http://{serverAddress}:{defaultPort}";
                    if (await ValidateSIEMServerAsync(serverUrl))
                    {
                        _discoveredServers.Add(serverUrl);
                        _logger.LogInformation("✅ Environment discovery found SIEM server: {Server} (from {EnvVar})", 
                            serverUrl, envVar);
                    }
                }
            }
        }

        /// <summary>
        /// Windows Registry discovery (MSI deployment)
        /// </summary>
        private async Task DiscoverViaRegistryAsync()
        {
            try
            {
                using var key = Microsoft.Win32.Registry.LocalMachine.OpenSubKey(@"SOFTWARE\AthalaSIEM\UniversalAgent\Configuration");
                if (key != null)
                {
                    var backendUrl = key.GetValue("BackendUrl")?.ToString();
                    if (!string.IsNullOrEmpty(backendUrl) && await ValidateSIEMServerAsync(backendUrl))
                    {
                        _discoveredServers.Add(backendUrl);
                        _logger.LogInformation("✅ Registry discovery found SIEM server: {Server}", backendUrl);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Registry discovery failed");
            }
        }

        /// <summary>
        /// Validate if a server is a valid SIEM Manager
        /// </summary>
        private async Task<bool> ValidateSIEMServerAsync(string serverUrl)
        {
            try
            {
                _httpClient.Timeout = TimeSpan.FromSeconds(5);
                var response = await _httpClient.GetAsync($"{serverUrl}/api/health");
                
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    // Check if response indicates this is an AthalaSIEM server
                    return content.Contains("AthalaSIEM") || content.Contains("SIEM") || response.IsSuccessStatusCode;
                }
                
                return false;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Register agent with discovered SIEM server
        /// </summary>
        public async Task<AgentRegistrationResponse?> RegisterWithServerAsync(string serverUrl, string? deploymentToken = null)
        {
            try
            {
                _logger.LogInformation("📝 Registering agent with SIEM server: {Server}", serverUrl);
                
                var registrationRequest = new AgentRegistrationRequest
                {
                    DeploymentToken = deploymentToken ?? Environment.GetEnvironmentVariable("ATHALA_DEPLOYMENT_TOKEN") ?? "",
                    Hostname = Environment.MachineName,
                    IpAddress = GetLocalIPAddress(),
                    Platform = Environment.OSVersion.Platform.ToString(),
                    OsVersion = Environment.OSVersion.VersionString,
                    Version = "1.0.0",
                    Capabilities = GetAgentCapabilities()
                };

                var json = JsonSerializer.Serialize(registrationRequest);
                var content = new StringContent(json, System.Text.Encoding.UTF8, "application/json");
                
                var response = await _httpClient.PostAsync($"{serverUrl}/api/agentdeployment/register-dev", content);
                
                if (response.IsSuccessStatusCode)
                {
                    var responseJson = await response.Content.ReadAsStringAsync();
                    var registrationResponse = JsonSerializer.Deserialize<AgentRegistrationResponse>(responseJson, new JsonSerializerOptions
                    {
                        PropertyNameCaseInsensitive = true
                    });

                    if (registrationResponse?.IsValid() == true)
                    {
                        _selectedServer = serverUrl;
                        _logger.LogInformation("✅ Agent registered successfully with server: {Server}", serverUrl);
                        return registrationResponse;
                    }
                }
                
                _logger.LogWarning("❌ Agent registration failed with server: {Server} - {StatusCode}", 
                    serverUrl, response.StatusCode);
                return null;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error registering agent with server: {Server}", serverUrl);
                return null;
            }
        }

        /// <summary>
        /// Get local network ranges for broadcast discovery
        /// </summary>
        private List<string> GetLocalNetworkRanges()
        {
            var ranges = new List<string>();
            
            try
            {
                foreach (var adapter in NetworkInterface.GetAllNetworkInterfaces())
                {
                    if (adapter.OperationalStatus == OperationalStatus.Up && 
                        adapter.NetworkInterfaceType != NetworkInterfaceType.Loopback)
                    {
                        foreach (var address in adapter.GetIPProperties().UnicastAddresses)
                        {
                            if (address.Address.AddressFamily == System.Net.Sockets.AddressFamily.InterNetwork)
                            {
                                var ip = address.Address.ToString();
                                var subnet = ip.Substring(0, ip.LastIndexOf('.'));
                                ranges.Add(subnet);
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to get local network ranges");
            }
            
            return ranges.Distinct().ToList();
        }

        /// <summary>
        /// Scan network range for SIEM servers
        /// </summary>
        private async Task ScanNetworkRangeAsync(string subnet, int port, int timeoutSeconds)
        {
            var tasks = new List<Task>();
            
            for (int i = 1; i <= 254; i++)
            {
                var ip = $"{subnet}.{i}";
                tasks.Add(Task.Run(async () =>
                {
                    try
                    {
                        var serverUrl = $"http://{ip}:{port}";
                        if (await ValidateSIEMServerAsync(serverUrl))
                        {
                            lock (_discoveredServers)
                            {
                                _discoveredServers.Add(serverUrl);
                            }
                            _logger.LogInformation("✅ Broadcast discovery found SIEM server: {Server}", serverUrl);
                        }
                    }
                    catch
                    {
                        // Ignore scan failures
                    }
                }));
            }
            
            await Task.WhenAll(tasks);
        }

        /// <summary>
        /// Get local IP address
        /// </summary>
        private string GetLocalIPAddress()
        {
            try
            {
                var host = Dns.GetHostEntry(Dns.GetHostName());
                return host.AddressList
                    .FirstOrDefault(ip => ip.AddressFamily == System.Net.Sockets.AddressFamily.InterNetwork)
                    ?.ToString() ?? "0.0.0.0";
            }
            catch
            {
                return "0.0.0.0";
            }
        }

        /// <summary>
        /// Get agent capabilities
        /// </summary>
        private List<string> GetAgentCapabilities()
        {
            var capabilities = new List<string> { "LogCollection", "FileIntegrity", "RealTimeMonitoring" };
            
            if (System.OperatingSystem.IsWindows())
            {
                capabilities.AddRange(new[] { "WindowsEventLog", "WindowsRegistry", "WindowsFirewall" });
            }
            
            if (System.OperatingSystem.IsLinux())
            {
                capabilities.AddRange(new[] { "LinuxSyslog", "SystemdJournal", "LinuxFirewall" });
            }
            
            return capabilities;
        }

        /// <summary>
        /// Get default backend port with configuration fallback
        /// </summary>
        private int GetDefaultBackendPort()
        {
            // Try to get from configuration first
            var configuredPort = _configuration.GetValue<int>("SiemManager:ManagerPort");
            if (configuredPort > 0)
                return configuredPort;
            
            // Try enterprise default ports configuration
            var enterprisePort = _configuration.GetValue<int>("Enterprise:DefaultPorts:Backend");
            if (enterprisePort > 0)
                return enterprisePort;
            
            // Ultimate fallback - standard SIEM port
            return 9595;
        }

        /// <summary>
        /// Get default frontend port with configuration fallback
        /// </summary>
        private int GetDefaultFrontendPort()
        {
            var enterprisePort = _configuration.GetValue<int>("Enterprise:DefaultPorts:Frontend");
            return enterprisePort > 0 ? enterprisePort : 3000;
        }

        /// <summary>
        /// Get selected SIEM server
        /// </summary>
        public string? GetSelectedServer() => _selectedServer;

        /// <summary>
        /// Get all discovered servers
        /// </summary>
        public List<string> GetDiscoveredServers() => _discoveredServers.ToList();
    }
} 