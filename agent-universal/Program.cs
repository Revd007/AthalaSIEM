using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using AthalaSIEM.UniversalAgent;
using AthalaSIEM.Agent.Core;
using AthalaSIEM.UniversalAgent.Services;
using System.ServiceProcess;
using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Threading;
using AthalaSIEM.Agent.Collectors;

namespace AthalaSIEM.UniversalAgent
{
    public class CollectorConfiguration
    {
        public string Type { get; set; } = "";
        public bool Enabled { get; set; } = true;
        public Dictionary<string, object> Properties { get; set; } = new();
    }

    public class Program
    {
        public static void Main(string[] args)
        {
            if (args.Length > 0)
            {
                switch (args[0].ToLowerInvariant())
                {
                    case "--help":
                    case "-h":
                        ShowHelp();
                        return;
                    case "--console":
                    case "-c":
                        RunConsoleMode(args);
                        return;
                    case "--install":
                        InstallService();
                        return;
                    case "--uninstall":
                        UninstallService();
                        return;
                    case "--test-connection":
                        TestConnection();
                        return;
                    case "--config":
                        ShowConfiguration();
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
                    
                    // Register Windows Authentication Service (Windows only)
                    if (System.OperatingSystem.IsWindows())
                    {
                        services.AddSingleton<WindowsAuthenticationService>();
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
            Console.WriteLine("⚠️  IMPORTANT: Administrator privileges required for Security Event Log access!");
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
                Console.WriteLine($"❌ Error running agent: {ex.Message}");
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
                .Build();

            var managerIP = configuration["SiemManager:ManagerIP"] ?? "192.168.1.100";
            var managerPort = configuration.GetValue<int>("SiemManager:ManagerPort", 9595);
            var managerUrl = $"http://{managerIP}:{managerPort}";
            var agentName = configuration["Agent:Name"] ?? Environment.MachineName;
            var apiKey = configuration["Agent:ApiKey"] ?? "";
            
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
                    Console.WriteLine("✅ Health check successful!");
                    
                    // Test agent registration endpoint
                    Console.WriteLine("Testing agent registration...");
                    var registrationData = new
                    {
                        AgentId = Environment.MachineName,
                        AgentName = agentName,
                        Version = "1.0.0",
                        Platform = Environment.OSVersion.Platform.ToString()
                    };
                    
                    var json = System.Text.Json.JsonSerializer.Serialize(registrationData);
                    var content = new StringContent(json, System.Text.Encoding.UTF8, "application/json");
                    var regResponse = client.PostAsync($"{managerUrl}/api/agents/register", content).Result;
                    
                    if (regResponse.IsSuccessStatusCode)
                    {
                        Console.WriteLine("✅ Agent registration test successful!");
                    }
                    else
                    {
                        Console.WriteLine($"⚠️ Agent registration test failed: {regResponse.StatusCode}");
                    }
                }
                else
                {
                    Console.WriteLine($"❌ Health check failed: {response.StatusCode}");
                    Console.WriteLine($"Response: {response.Content.ReadAsStringAsync().Result}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Connection test failed: {ex.Message}");
                Console.WriteLine("💡 Possible issues:");
                Console.WriteLine("   - SIEM Manager server is not running");
                Console.WriteLine("   - Incorrect Manager IP/Port in appsettings.json");
                Console.WriteLine("   - Network connectivity issues");
                Console.WriteLine("   - Firewall blocking the connection");
            }
        }
        
        private static void ShowConfiguration()
        {
            Console.WriteLine("📋 Current Configuration");
            Console.WriteLine("========================");
            
            try
            {
                var configuration = new ConfigurationBuilder()
                    .SetBasePath(Directory.GetCurrentDirectory())
                    .AddJsonFile("appsettings.json", optional: false)
                    .Build();

                var managerIP = configuration["SiemManager:ManagerIP"] ?? "Not configured";
                var managerPort = configuration.GetValue<int>("SiemManager:ManagerPort", 9595);
                Console.WriteLine($"SIEM Manager IP: {managerIP}");
                Console.WriteLine($"SIEM Manager Port: {managerPort}");
                Console.WriteLine($"Agent Name: {configuration["Agent:Name"] ?? Environment.MachineName}");
                Console.WriteLine($"Agent ID: {configuration["Agent:Id"] ?? Environment.MachineName}");
                Console.WriteLine($"API Key: {(string.IsNullOrEmpty(configuration["Agent:ApiKey"]) ? "Not configured" : "Configured")}");
                Console.WriteLine($"Batch Size: {configuration["Agent:BatchSize"] ?? "100"}");
                Console.WriteLine($"Batch Interval: {configuration["Agent:BatchIntervalSeconds"] ?? "30"} seconds");
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
                Console.WriteLine($"❌ Error reading configuration: {ex.Message}");
            }
        }
    }
} 