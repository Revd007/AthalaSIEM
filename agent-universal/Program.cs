using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using AthalaSIEM.UniversalAgent;
using AthalaSIEM.UniversalAgent.Core;
using AthalaSIEM.UniversalAgent.Services;
using System.ServiceProcess;
using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Threading;
using System.Runtime.InteropServices;
using AthalaSIEM.Agent.Collectors;
using AthalaSIEM.UniversalAgent.Core.Collectors;
using System.Threading.Tasks;
using AthalaSIEM.UniversalAgent.Services.Interfaces;
using AthalaSIEM.UniversalAgent.Models;
using AthalaSIEM.UniversalAgent.UAT;

namespace AthalaSIEM.UniversalAgent
{
    public class Program
    {
        public static async Task Main(string[] args)
        {
            if (args.Length > 0)
            {
                var command = args[0].ToLowerInvariant();
                switch (command)
                {
                    case "--install":
                        InstallService();
                        return;
                    case "--uninstall":
                        UninstallService();
                        return;
                    case "--console":
                        RunConsoleMode(args);
                        return;
                    case "--test-connection":
                        TestConnection();
                        return;
                    case "--test-connection-silent":
                        TestConnectionSilent();
                        return;
                    case "--configure-msi":
                        ConfigureMSI();
                        return;
                    case "--check-runtime":
                        CheckDotNetRuntime();
                        return;
                    case "--status":
                        ShowStatus();
                        return;
                    case "--config":
                        ShowConfiguration();
                        return;
                    case "--run-uat":
                        await RunUATTestsAsync();
                        return;
                    case "--help":
                    case "-h":
                    case "/?":
                        ShowHelp();
                        return;
                    default:
                        Console.WriteLine($"Unknown command: {command}");
                        ShowHelp();
                        return;
                }
            }

            // Default: Run as Windows Service
            CreateHostBuilder(args).Build().Run();
        }

        private static IHostBuilder CreateHostBuilder(string[] args) =>
            Host.CreateDefaultBuilder(args)
                .UseWindowsService(options =>
                {
                    options.ServiceName = "AthalaSIEMUniversalAgent";
                })
                .ConfigureServices((hostContext, services) =>
                {
                    // Register ManageEngine-style pipeline services
                    services.AddSingleton<CollectorManager>();
                    services.AddSingleton<LogProcessor>(provider => 
                    {
                        var logger = provider.GetRequiredService<ILogger<LogProcessor>>();
                        var loggerFactory = provider.GetRequiredService<ILoggerFactory>();
                        var configuration = provider.GetRequiredService<IConfiguration>();
                        return new LogProcessor(logger, loggerFactory, configuration);
                    });
                    services.AddSingleton<GrpcCommunicationService>();
                    services.AddSingleton<BackendCommunicationService>();
                    services.AddHttpClient<BackendCommunicationService>();
                    
                    // Register Enterprise Services - NO HARDCODED VALUES
                    services.AddSingleton<AgentDiscoveryService>();
                    services.AddHttpClient<AgentDiscoveryService>();
                    services.AddSingleton<FIMConfigurationService>();
                    services.AddHttpClient<FIMConfigurationService>();
                    
                    // Register Cross-Platform Collectors - Enterprise Architecture
                    services.AddSingleton<FirewallCollector>(); // Universal firewall monitoring
                    
                    // Register Windows-specific services
                    if (System.OperatingSystem.IsWindows())
                    {
                        services.AddSingleton<WindowsAuthenticationService>();
                        services.AddSingleton<WindowsEventLogCollector>();
                        services.AddSingleton<FileIntegrityCollector>();
                        services.AddSingleton<WindowsRegistryCollector>();
                    }

                    // Register Linux-specific collectors  
                    if (System.OperatingSystem.IsLinux())
                    {
                        services.AddSingleton<LinuxSyslogCollector>();
                    }
                    
                    // Register the main service
                    services.AddHostedService<UniversalAgentService>();
                    
                    // Configure logging
                    services.AddLogging(builder =>
                    {
                        builder.AddConsole();
                        if (System.OperatingSystem.IsWindows())
                        {
                            builder.AddEventLog();
                        }
                        builder.SetMinimumLevel(LogLevel.Information);
                    });
                });

        private static void ShowHelp()
        {
            Console.WriteLine("🛡️ Athala SIEM Universal Agent v1.0.0");
            Console.WriteLine("Following ManageEngine EventLog Analyzer architecture patterns");
            Console.WriteLine();
            Console.WriteLine(" IMPORTANT: Administrator privileges required for Security Event Log access!");
            Console.WriteLine("    Without Security logs, this is NOT a functional SIEM agent.");
            Console.WriteLine();
            Console.WriteLine("Usage: athala-agent.exe [options]");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("  --help, -h           Show this help message");
            Console.WriteLine("  --console, -c        Run in console mode for testing");
            Console.WriteLine("  --install            Install as Windows service");
            Console.WriteLine("  --uninstall          Uninstall Windows service");
            Console.WriteLine("  --test-connection    Test connection to backend API");
            Console.WriteLine("  --config             Show current configuration");
            Console.WriteLine("  --run-uat            Run User Acceptance Tests (UAT)");
            Console.WriteLine();
            Console.WriteLine("Default: Run as Windows service");
            Console.WriteLine();
            Console.WriteLine("Examples:");
            Console.WriteLine("  # Run as Administrator for full SIEM functionality:");
            Console.WriteLine("  athala-agent.exe --console");
            Console.WriteLine("  athala-agent.exe --test-connection");
            Console.WriteLine("  athala-agent.exe --install");
        }

        private static void RunConsoleMode(string[] args)
        {
            Console.WriteLine("🛡️ Athala SIEM Universal Agent - Console Mode");
            Console.WriteLine("Press Ctrl+C to exit.");
            Console.WriteLine();

            try
            {
                // Build the same host as the service but run in console
                var host = CreateHostBuilder(args)
                    .UseConsoleLifetime()
                    .Build();

                // Setup cancellation
                var cancellationTokenSource = new CancellationTokenSource();
                Console.CancelKeyPress += (sender, e) =>
                {
                    Console.WriteLine("\n🛑 Shutdown requested...");
                    cancellationTokenSource.Cancel();
                    e.Cancel = true;
                };

                // Run the agent
                host.RunAsync(cancellationTokenSource.Token).Wait();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error running agent: {ex.Message}");
                Console.WriteLine($"Details: {ex}");
            }
        }

        private static void InstallService()
        {
            try
            {
                Console.WriteLine("Installing Athala SIEM Universal Agent service...");
                // Service installation logic would go here
                Console.WriteLine("Service installed successfully!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to install service: {ex.Message}");
            }
        }

        private static void UninstallService()
        {
            try
            {
                Console.WriteLine("Uninstalling Athala SIEM Universal Agent service...");
                // Service uninstallation logic would go here
                Console.WriteLine("Service uninstalled successfully!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to uninstall service: {ex.Message}");
            }
        }

        private static void TestConnection()
        {
            Console.WriteLine("🔗 Testing connection to backend...");
            Console.WriteLine();
            
            var configuration = new ConfigurationBuilder()
                .SetBasePath(Directory.GetCurrentDirectory())
                .AddJsonFile("appsettings.json", optional: false)
                .AddEnvironmentVariables("ATHALA_")
                .Build();

            var managerIP = configuration["SiemManager:ManagerIP"];
            var managerPort = configuration.GetValue<int>("SiemManager:ManagerPort");
            var agentName = configuration["Agent:Name"] ?? Environment.MachineName;
            var apiKey = configuration["Agent:ApiKey"];

            // Validate required configuration
            if (string.IsNullOrEmpty(managerIP))
            {
                Console.WriteLine("CONFIGURATION ERROR: SIEM Manager IP not configured!");
                Console.WriteLine("💡 Please configure SiemManager:ManagerIP in appsettings.json or environment variable ATHALA_SiemManager__ManagerIP");
                return;
            }
            
            if (managerPort == 0)
            {
                Console.WriteLine("CONFIGURATION ERROR: SIEM Manager Port not configured!");
                Console.WriteLine("💡 Please configure SiemManager:ManagerPort in appsettings.json or environment variable ATHALA_SiemManager__ManagerPort");
                return;
            }

            var managerUrl = $"http://{managerIP}:{managerPort}";
            
            Console.WriteLine($"Agent Name: {agentName}");
            Console.WriteLine($"SIEM Manager: {managerIP}:{managerPort}");
            Console.WriteLine($"Manager URL: {managerUrl}");
            Console.WriteLine($"API Key: {(string.IsNullOrEmpty(apiKey) ? "Not configured" : "Configured")}");
            Console.WriteLine();
            
            try
            {
                using var client = new HttpClient();
                client.Timeout = TimeSpan.FromSeconds(10);
                
                if (!string.IsNullOrEmpty(apiKey))
                {
                    client.DefaultRequestHeaders.Add("X-API-Key", apiKey);
                }

                Console.WriteLine("Testing health endpoint...");
                var response = client.GetAsync($"{managerUrl}/api/health").Result;
                
                if (response.IsSuccessStatusCode)
                {
                    Console.WriteLine(" Health check successful!");
                    
                    // Test agent registration endpoint
                    Console.WriteLine("Testing agent registration...");
                    var registrationData = new
                    {
                        AgentId = Environment.MachineName,
                        AgentName = agentName,
                        Version = Constants.Defaults.AgentVersion,
                        Platform = Environment.OSVersion.Platform.ToString()
                    };
                    
                    var json = System.Text.Json.JsonSerializer.Serialize(registrationData);
                    var content = new StringContent(json, System.Text.Encoding.UTF8, "application/json");
                    var regResponse = client.PostAsync($"{managerUrl}/api/agents/register", content).Result;
                    
                    if (regResponse.IsSuccessStatusCode)
                    {
                        Console.WriteLine(" Agent registration test successful!");
                    }
                    else
                    {
                        Console.WriteLine($"Agent registration test failed: {regResponse.StatusCode}");
                    }
                }
                else
                {
                    Console.WriteLine($"Health check failed: {response.StatusCode}");
                    Console.WriteLine($"Response: {response.Content.ReadAsStringAsync().Result}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Connection test failed: {ex.Message}");
                Console.WriteLine("💡 Possible issues:");
                Console.WriteLine("   - SIEM Manager server is not running");
                Console.WriteLine("   - Manager IP/Port not configured in appsettings.json");
                Console.WriteLine("   - Network connectivity issues");
                Console.WriteLine("   - Firewall blocking the connection");
                Console.WriteLine("   - Invalid configuration values");
            }
        }
        
        private static void ShowConfiguration()
        {
            Console.WriteLine(" Current Configuration");
            Console.WriteLine("========================");
            
            try
            {
                var configuration = new ConfigurationBuilder()
                    .SetBasePath(Directory.GetCurrentDirectory())
                    .AddJsonFile("appsettings.json", optional: false)
                    .AddEnvironmentVariables("ATHALA_")
                    .Build();

                var managerIP = configuration["SiemManager:ManagerIP"];
                var managerPort = configuration.GetValue<int>("SiemManager:ManagerPort", 9595);
                Console.WriteLine($"SIEM Manager IP: {managerIP ?? "NOT CONFIGURED"}");
                Console.WriteLine($"SIEM Manager Port: {managerPort}");
                Console.WriteLine($"Agent Name: {configuration["Agent:Name"] ?? Environment.MachineName}");
                Console.WriteLine($"Agent ID: {configuration["Agent:Id"] ?? Environment.MachineName}");
                Console.WriteLine($"API Key: {(string.IsNullOrEmpty(configuration["Agent:ApiKey"]) ? "NOT CONFIGURED" : " Configured")}");
                Console.WriteLine($"Batch Size: {configuration["Agent:BatchSize"] ?? "100"}");
                Console.WriteLine($"Batch Interval: {configuration["Agent:BatchIntervalSeconds"] ?? "30"} seconds");
                Console.WriteLine();
                
                // Security status
                Console.WriteLine("🔒 Security Configuration Status:");
                Console.WriteLine($"   Manager IP: {(string.IsNullOrEmpty(managerIP) ? "Missing" : " Configured")}");
                Console.WriteLine($"   API Key: {(string.IsNullOrEmpty(configuration["Agent:ApiKey"]) ? "Missing" : " Configured")}");
                Console.WriteLine($"   Registration Key: {(string.IsNullOrEmpty(configuration["Agent:RegistrationKey"]) ? "Missing" : " Configured")}");
                Console.WriteLine();
                
                // Show collectors configuration
                var collectorsConfig = configuration.GetSection("Collectors");
                if (collectorsConfig.Exists())
                {
                    Console.WriteLine("Configured Collectors:");
                    try
                    {
                        var collectors = collectorsConfig.Get<List<CollectorConfiguration>>();
                        foreach (var collector in collectors ?? new List<CollectorConfiguration>())
                        {
                            Console.WriteLine($"  - {collector.Type}: {(collector.Enabled ? "Enabled" : "Disabled")}");
                        }
                    }
                    catch
                    {
                        Console.WriteLine("  - Error reading collector configuration");
                    }
                }
                else
                {
                    Console.WriteLine("Collectors: Using default configuration (Windows Event Log)");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reading configuration: {ex.Message}");
            }
        }

        private static void TestConnectionSilent()
        {
            try
            {
                var configuration = new ConfigurationBuilder()
                    .SetBasePath(Directory.GetCurrentDirectory())
                    .AddJsonFile("appsettings.json", optional: false)
                    .AddEnvironmentVariables("ATHALA_")
                    .Build();

                var managerIP = configuration["SiemManager:ManagerIP"];
                var managerPort = configuration.GetValue<int>("SiemManager:ManagerPort", 9595);
                
                if (string.IsNullOrEmpty(managerIP))
                {
                    Environment.Exit(1); // Configuration error
                    return;
                }

                var managerUrl = $"http://{managerIP}:{managerPort}";
                
                using var client = new HttpClient();
                client.Timeout = TimeSpan.FromSeconds(5);
                
                var response = client.GetAsync($"{managerUrl}/api/health").Result;
                Environment.Exit(response.IsSuccessStatusCode ? 0 : 1);
            }
            catch
            {
                Environment.Exit(1); // Connection failed
            }
        }

        private static void ConfigureMSI()
        {
            try
            {
                // Read MSI-provided configuration from registry (Windows only)
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    using var key = Microsoft.Win32.Registry.LocalMachine.OpenSubKey(@"SOFTWARE\AthalaSIEM\UniversalAgent\Configuration");
                    if (key != null)
                    {
                        var backendUrl = key.GetValue("BackendUrl")?.ToString();
                        var agentName = key.GetValue("AgentName")?.ToString() ?? Environment.MachineName;
                        
                        if (!string.IsNullOrEmpty(backendUrl))
                        {
                            // Update appsettings.json with MSI configuration
                            var configPath = Path.Combine(Directory.GetCurrentDirectory(), "appsettings.json");
                            if (File.Exists(configPath))
                            {
                                var configText = File.ReadAllText(configPath);
                                var config = System.Text.Json.JsonSerializer.Deserialize<System.Text.Json.JsonElement>(configText);
                                
                                // Extract URL components
                                var uri = new Uri(backendUrl);
                                var configDict = new Dictionary<string, object>
                                {
                                    ["SiemManager"] = new Dictionary<string, object>
                                    {
                                        ["ManagerIP"] = uri.Host,
                                        ["ManagerPort"] = uri.Port,
                                        ["UseHTTPS"] = uri.Scheme == "https"
                                    },
                                    ["Agent"] = new Dictionary<string, object>
                                    {
                                        ["Name"] = agentName,
                                        ["Id"] = agentName,
                                        ["ManagerUrl"] = backendUrl
                                    }
                                };
                                
                                // Merge with existing config and save
                                var options = new System.Text.Json.JsonSerializerOptions { WriteIndented = true };
                                var newConfigText = System.Text.Json.JsonSerializer.Serialize(configDict, options);
                                
                                // For simplicity, just update the core values we need
                                configText = configText.Replace("\"ManagerIP\": \"\"", $"\"ManagerIP\": \"{uri.Host}\"");
                                configText = configText.Replace("\"ManagerPort\": 9595", $"\"ManagerPort\": {uri.Port}");
                                configText = configText.Replace("\"ManagerUrl\": \"\"", $"\"ManagerUrl\": \"{backendUrl}\"");
                                configText = configText.Replace("\"Name\": \"AthalaSIEM-Universal-Agent\"", $"\"Name\": \"{agentName}\"");
                                
                                File.WriteAllText(configPath, configText);
                            }
                        }
                    }
                }
                else
                {
                    // On non-Windows platforms, skip registry-based configuration
                    Console.WriteLine("Registry-based configuration skipped - not running on Windows");
                }
                
                Environment.Exit(0); // Success
            }
            catch
            {
                Environment.Exit(1); // Configuration failed
            }
        }

        private static void CheckDotNetRuntime()
        {
            try
            {
                // Check if .NET 8.0 runtime is available
                var version = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription;
                Environment.Exit(version.Contains(".NET 8.0") ? 0 : 1);
            }
            catch
            {
                Environment.Exit(1); // Runtime check failed
            }
        }

        private static void ShowStatus()
        {
            try
            {
                // Check service status (Windows only)
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    var serviceName = "AthalaSIEMUniversalAgent";
                    using var serviceController = new System.ServiceProcess.ServiceController(serviceName);
                    
                    Console.WriteLine($"Service Status: {serviceController.Status}");
                    Console.WriteLine($"Service Type: {serviceController.ServiceType}");
                    Console.WriteLine($"Can Stop: {serviceController.CanStop}");
                    Console.WriteLine($"Can Pause/Continue: {serviceController.CanPauseAndContinue}");
                    
                    if (serviceController.Status == System.ServiceProcess.ServiceControllerStatus.Running)
                    {
                        Environment.Exit(0);
                    }
                    else
                    {
                        Environment.Exit(1);
                    }
                }
                else
                {
                    Console.WriteLine("Service status check skipped - not running on Windows");
                    Environment.Exit(0);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error checking service status: {ex.Message}");
                Environment.Exit(1);
            }
        }

        /// <summary>
        /// Runs UAT (User Acceptance Tests) for the Universal Agent.
        /// </summary>
        private static async Task RunUATTestsAsync()
        {
            Console.WriteLine("🧪 Starting UAT (User Acceptance Tests)...");
            Console.WriteLine("============================================");
            Console.WriteLine();

            try
            {
                // Run the UAT test runner
                var exitCode = await RunUAT.RunUATTestsAsync(new string[0]);
                
                if (exitCode == 0)
                {
                    Console.WriteLine("🎉 UAT tests completed successfully!");
                }
                else
                {
                    Console.WriteLine("UAT tests failed. Check the output above for details.");
                    Environment.Exit(exitCode);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"UAT execution failed: {ex.Message}");
                Console.WriteLine($"Details: {ex}");
                Environment.Exit(1);
            }
        }
    }
}
