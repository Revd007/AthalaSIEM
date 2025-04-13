using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;
using System;
using System.IO;
using System.Threading.Tasks;
using System.Linq;
using AthalaSIEM.Agent.Services;
using AthalaSIEM.Agent.Models;
using AthalaSIEM.Agent.Collectors;
using AthalaSIEM.Agent.Communication;
using AthalaSIEM.Agent.Security;
using AthalaSIEM.Agent.Configuration;
using System.Net.Http;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using Serilog;
using Polly;
using Polly.Extensions.Http;
using Polly.Retry;

namespace AthalaSIEM.Agent
{
    /// <summary>
    /// Main program entry point
    /// </summary>
    public class Program
    {
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        public static async Task Main(string[] args)
        {
            // Get the directory of the executable - this is always correct
            string executablePath = AppContext.BaseDirectory;
            
            // Force working directory to be the same as executable path
            try
            {
                Directory.SetCurrentDirectory(executablePath);
            }
            catch
            {
                // Non-critical if it fails
            }
            
            // These will be used by Microsoft.Extensions.Configuration.FileConfigurationExtensions
            Environment.SetEnvironmentVariable("DOTNET_CONTENTROOT", executablePath);
            Environment.SetEnvironmentVariable("ASPNETCORE_CONTENTROOT", executablePath);
            
            // Define standard application data folder for emergency fallback
            string appDataFolderPath = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData),
                "Athala SIEM Agent");
            
            // Create it if it doesn't exist
            if (!Directory.Exists(appDataFolderPath))
            {
                try
                {
                    Directory.CreateDirectory(appDataFolderPath);
                }
                catch { /* Ignore creation errors */ }
            }
            
            // Log startup information to help debug
            try
            {
                string startupLogPath = Path.Combine(appDataFolderPath, "startup.log");
                File.AppendAllText(startupLogPath, $"[{DateTime.Now}] Starting from directory: {executablePath}\n");
                File.AppendAllText(startupLogPath, $"[{DateTime.Now}] Current directory: {Directory.GetCurrentDirectory()}\n");
            }
            catch { /* Ignore errors writing startup log */ }
            
            // Build a list of potential configuration file paths
            var configFiles = new List<string>
            {
                // First priority - executable directory (most reliable)
                Path.Combine(executablePath, "appsettings.json"),
            };
            
            // Add current working directory if different
            string currentDir = Directory.GetCurrentDirectory();
            if (currentDir != executablePath)
            {
                configFiles.Add(Path.Combine(currentDir, "appsettings.json"));
            }
            
            // Add standard installation paths
            configFiles.Add(@"C:\Program Files (x86)\Athala SIEM Agent\appsettings.json");
            configFiles.Add(@"C:\Program Files\Athala SIEM Agent\appsettings.json");
                
            // ProgramData fallback (last resort)
            configFiles.Add(Path.Combine(appDataFolderPath, "appsettings.json"));
            
            // Check all possible paths and use the first one that exists
            string? primaryConfigPath = null;
            foreach (var path in configFiles)
            {
                if (File.Exists(path))
                {
                    primaryConfigPath = path;
                    break;
                }
            }
            
            // If no configuration file found anywhere, create a default one in ProgramData
            if (primaryConfigPath == null)
            {
                try
                {
                    // Create a default config in ProgramData as a safe location accessible to services
                    string fallbackConfigPath = Path.Combine(appDataFolderPath, "appsettings.json");
                    string minimalConfig = @"{
""Logging"": {
  ""LogLevel"": {
    ""Default"": ""Information"",
    ""Microsoft"": ""Warning"",
    ""Microsoft.Hosting.Lifetime"": ""Information""
  }
},
""Agent"": {
  ""AgentName"": ""AthalaSIEM Agent"",
  ""BackendApiUrl"": ""https://localhost:9596"",
  ""BackendGrpcUrl"": ""https://localhost:50051""
}
}";
                    File.WriteAllText(fallbackConfigPath, minimalConfig);
                    primaryConfigPath = fallbackConfigPath;
                    
                    // Log this emergency fallback
                    File.AppendAllText(
                        Path.Combine(appDataFolderPath, "startup.log"),
                        $"[{DateTime.Now}] Created emergency fallback configuration at: {fallbackConfigPath}\n");
                }
                catch (Exception) // Use proper discard without declaring a variable
                {
                    // Last resort - use temp directory
                    try
                    {
                        string tempPath = Path.Combine(Path.GetTempPath(), "AthalaSIEM");
                        if (!Directory.Exists(tempPath))
                        {
                            Directory.CreateDirectory(tempPath);
                        }
                        
                        string fallbackPath = Path.Combine(tempPath, "appsettings.json");
                        if (!File.Exists(fallbackPath))
                        {
                            string minimalConfig = @"{
  ""Logging"": {
    ""LogLevel"": {
      ""Default"": ""Information"",
      ""Microsoft"": ""Warning"",
      ""Microsoft.Hosting.Lifetime"": ""Information""
    }
  },
  ""Agent"": {
    ""AgentName"": ""AthalaSIEM Agent"",
    ""BackendApiUrl"": ""https://localhost:9596"",
    ""BackendGrpcUrl"": ""https://localhost:50051""
  }
}";
                            File.WriteAllText(fallbackPath, minimalConfig);
                        }
                        
                        primaryConfigPath = fallbackPath;
                        Console.WriteLine($"Using temporary fallback configuration at: {primaryConfigPath}");
                    }
                    catch
                    {
                        // Absolute last resort, just use temp path
                        primaryConfigPath = Path.GetTempPath();
                        Console.WriteLine($"Using temp path for configuration: {primaryConfigPath}");
                    }
                }
            }
            
            // If we found a config path, log it
            if (primaryConfigPath != null)
            {
                try
                {
                    File.AppendAllText(
                        Path.Combine(appDataFolderPath, "startup.log"),
                        $"[{DateTime.Now}] Using configuration file: {primaryConfigPath}\n");
                }
                catch { /* Ignore logging errors */ }
            }
            
            // Create a configuration builder
            var configBuilder = new ConfigurationBuilder();
            
            if (primaryConfigPath != null)
            {
                // Set base path to the directory containing the config file
                string configDirectory = Path.GetDirectoryName(primaryConfigPath) ?? executablePath;
                configBuilder.SetBasePath(configDirectory);
                
                // Add the configuration file
                configBuilder.AddJsonFile(Path.GetFileName(primaryConfigPath), optional: false, reloadOnChange: true);
                
                Console.WriteLine($"Loading configuration from: {primaryConfigPath}");
                
                // Store the config file path for later reference by other parts of the application
                configBuilder.AddInMemoryCollection(new Dictionary<string, string?>
                {
                    { "ConfigFilePath", primaryConfigPath }
                });
            }
            else
            {
                // No configuration file found - use the executable directory as base
                configBuilder.SetBasePath(executablePath);
                
                // Add optional config file - will likely fail but at least we tried
                configBuilder.AddJsonFile("appsettings.json", optional: true, reloadOnChange: true);
                
                Console.WriteLine($"WARNING: No configuration file found in any of the expected locations!");
            }
            
            // Add environment variables and command line args
            configBuilder.AddEnvironmentVariables().AddCommandLine(args);
            
            // Build the configuration
            var configuration = configBuilder.Build();

            // Ensure logs and config directories exist
            string logPath = Path.Combine(executablePath, "logs");
            string configPath = Path.Combine(executablePath, "config");
            
            try
            {
                // Create logs directory
                if (!Directory.Exists(logPath))
                {
                    Directory.CreateDirectory(logPath);
                }
                
                // Create config directory
                if (!Directory.Exists(configPath))
                {
                    Directory.CreateDirectory(configPath);
                }
            }
            catch (Exception) // Use proper discard without declaring a variable
            {
                // If we can't create directories in the application path, 
                // fall back to a default location in CommonApplicationData
                string appDataPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData), "Athala SIEM Agent");
                
                try
                {
                    // Ensure parent directory exists
                    if (!Directory.Exists(appDataPath))
                    {
                        Directory.CreateDirectory(appDataPath);
                    }
                    
                    // Create logs directory in appdata
                    logPath = Path.Combine(appDataPath, "logs");
                    if (!Directory.Exists(logPath))
                    {
                        Directory.CreateDirectory(logPath);
                    }
                    
                    // Create config directory in appdata
                    configPath = Path.Combine(appDataPath, "config");
                    if (!Directory.Exists(configPath))
                    {
                        Directory.CreateDirectory(configPath);
                    }
                }
                catch
                {
                    // Last resort - use temp directory
                    string tempPath = Path.Combine(Path.GetTempPath(), "AthalaSIEM");
                    if (!Directory.Exists(tempPath))
                    {
                        Directory.CreateDirectory(tempPath);
                    }
                    
                    logPath = Path.Combine(tempPath, "logs");
                    if (!Directory.Exists(logPath))
                    {
                        Directory.CreateDirectory(logPath);
                    }
                    
                    configPath = Path.Combine(tempPath, "config");
                    if (!Directory.Exists(configPath))
                    {
                        Directory.CreateDirectory(configPath);
                    }
                }
            }
            
            // Initialize logging
            Log.Logger = new LoggerConfiguration()
                .MinimumLevel.Information()
                .WriteTo.Console()
                .WriteTo.File(Path.Combine(logPath, "agent-.log"), rollingInterval: Serilog.RollingInterval.Day)
                .CreateLogger();

            try
            {
                Log.Information("Starting Athala SIEM Agent from directory: {0}", executablePath);
                Log.Information("Logs directory: {0}", logPath);
                
                // Parse command line arguments
                if (args.Length > 0)
                {
                    // Check for silent installation mode with token
                    if (TryParseCommandLineArgs(args, out var parsedArgs))
                    {
                        await HandleAutomatedDeployment(parsedArgs, configuration);
                return;
                    }
                }
                
                // Normal startup
                await CreateHostBuilder(args).Build().RunAsync();
            }
            catch (Exception ex)
            {
                Log.Fatal(ex, "The agent terminated unexpectedly");
                throw;
            }
            finally
            {
                Log.CloseAndFlush();
            }
        }
        
        /// <summary>
        /// Handles automated deployment with command line parameters
        /// </summary>
        private static async Task HandleAutomatedDeployment(Dictionary<string, string> args, IConfiguration configuration)
        {
            Log.Information("Running in automated deployment mode");
            
            // Build services manually
            var serviceCollection = new ServiceCollection();
            
            // Add configuration
            serviceCollection.AddSingleton<IConfiguration>(configuration);
            
            // Register required services
            ConfigureServices(serviceCollection, configuration);
            
            // Build service provider
            var serviceProvider = serviceCollection.BuildServiceProvider();
            
            // Get agent identity service
            var agentIdentityService = serviceProvider.GetRequiredService<IAgentIdentityService>();
            
            try
            {
                // Check if a deployment token was provided
                if (args.TryGetValue("token", out var token) && !string.IsNullOrEmpty(token))
                {
                    Log.Information("Registering agent with deployment token");
                    
                    // Register with token
                    var result = await agentIdentityService.RegisterWithTokenAsync(token);
                    
                    if (result.Success)
                    {
                        Log.Information("Agent registered successfully with deployment token");
                        
                        // Start the actual agent as a service
                        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                        {
                            // For Windows, we'll often be called from the installer, so just exit
                            // The Windows service will be managed by the service control manager
                            Log.Information("Installation completed. Agent service will start automatically.");
                            return;
                        }
                        else
                        {
                            // For Linux, we can start the agent now
                            Log.Information("Starting agent service");
                            await CreateHostBuilder(Array.Empty<string>()).Build().RunAsync();
                        }
                    }
                    else
                    {
                        Log.Error("Failed to register agent with token: {ErrorMessage}", result.Message);
                        Environment.Exit(1);
                    }
                }
                else
                {
                    Log.Information("No deployment token provided, starting normal execution");
                    await CreateHostBuilder(Array.Empty<string>()).Build().RunAsync();
                }
            }
            catch (Exception ex)
            {
                Log.Error(ex, "Automated deployment failed");
                Environment.Exit(1);
            }
        }
        
        /// <summary>
        /// Parses command line arguments into a dictionary
        /// </summary>
        private static bool TryParseCommandLineArgs(string[] args, out Dictionary<string, string> parsedArgs)
        {
            parsedArgs = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            bool isAutomatedMode = false;
            
            for (int i = 0; i < args.Length; i++)
            {
                string arg = args[i];
                
                // Check for automation flags
                if (arg.Equals("--token", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                {
                    parsedArgs["token"] = args[i + 1];
                    i++;
                    isAutomatedMode = true;
                }
                else if (arg.Equals("--silent", StringComparison.OrdinalIgnoreCase))
                {
                    parsedArgs["silent"] = "true";
                    isAutomatedMode = true;
                }
                else if (arg.Equals("--register", StringComparison.OrdinalIgnoreCase))
                {
                    parsedArgs["register"] = "true";
                    isAutomatedMode = true;
                }
                else if (arg.Equals("--server-url", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                {
                    parsedArgs["serverUrl"] = args[i + 1];
                    i++;
                    isAutomatedMode = true;
                }
                else if (arg.Equals("--port", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                {
                    parsedArgs["port"] = args[i + 1];
                    i++;
                    isAutomatedMode = true;
                }
                else if (arg.Equals("--agent-name", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                {
                    parsedArgs["agentName"] = args[i + 1];
                    i++;
                    isAutomatedMode = true;
                }
            }
            
            return isAutomatedMode;
        }
        
        /// <summary>
        /// Shows the configuration UI
        /// </summary>
        private static async Task ShowConfigurationUI(IServiceProvider services, string token = "")
        {
            var logger = services.GetRequiredService<ILogger<Program>>();
            logger.LogInformation("Showing configuration UI{0}", 
                !string.IsNullOrEmpty(token) ? " with deployment token" : "");
            
            var configLauncher = services.GetRequiredService<AgentConfigurationLauncher>();
            
            // Use the token-enabled method if a token is provided
            bool isConfigured = await configLauncher.ShowConfigurationFormAsync(token, true);
            
            if (isConfigured)
            {
                logger.LogInformation("Agent successfully configured");
            }
            else
            {
                logger.LogWarning("Agent configuration incomplete. Agent may not function correctly.");
            }
        }
        
        /// <summary>
        /// Shows help information
        /// </summary>
        private static void ShowHelp()
        {
            Console.WriteLine("AthalaSIEM Agent");
            Console.WriteLine("Usage: AthalaSIEM.Agent [options]");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("  -c, --configure    Show the configuration UI");
            Console.WriteLine("  -h, --help         Show this help information");
            Console.WriteLine();
            Console.WriteLine("When run without arguments, the agent will start as a service.");
            Console.WriteLine("On first run, the configuration UI will be shown if running in interactive mode.");
        }

        /// <summary>
        /// Creates the host builder
        /// </summary>
        /// <param name="args">Command line arguments</param>
        /// <returns>Host builder</returns>
        public static IHostBuilder CreateHostBuilder(string[] args) =>
            Host.CreateDefaultBuilder(args)
                .UseContentRoot(AppContext.BaseDirectory) // Force content root to be the executable directory
                .UseWindowsService(options =>
                {
                    options.ServiceName = "AthalaSIEM Agent";
                })
                .ConfigureHostConfiguration(config => 
                {
                    // Set content root path explicitly in host configuration
                    config.AddInMemoryCollection(new Dictionary<string, string?>
                    {
                        { HostDefaults.ContentRootKey, AppContext.BaseDirectory }
                    });
                })
                .ConfigureAppConfiguration((hostingContext, config) =>
                {
                    var basePath = AppContext.BaseDirectory;
                    config.SetBasePath(basePath);
                    
                    var configFilePath = Path.Combine(basePath, "appsettings.json");
                    if (File.Exists(configFilePath))
                    {
                        config.AddJsonFile(configFilePath, optional: false, reloadOnChange: true);
                    }
                    else
                    {
                        var commonAppData = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData);
                        var altConfigPath = Path.Combine(commonAppData, "Athala SIEM Agent", "appsettings.json");
                        if (File.Exists(altConfigPath))
                        {
                            config.AddJsonFile(altConfigPath, optional: false, reloadOnChange: true);
                        }
                    }
                    
                    config.AddEnvironmentVariables();
                })
                .ConfigureServices((hostContext, services) =>
                {
                    // Configure settings
                    var agentSettings = hostContext.Configuration.GetSection("Agent").Get<AgentSettings>();
                    services.AddSingleton(agentSettings ?? new AgentSettings());

                    // Register agent identity service first since it provides the agent ID
                    services.AddSingleton<IAgentIdentityService, AgentIdentityService>();
                    
                    // Register the agentId string dependency - needed by AgentHealthMonitor
                    services.AddSingleton(serviceProvider =>
                    {
                        var identityService = serviceProvider.GetRequiredService<IAgentIdentityService>();
                        // Get agent ID or use a default if not registered yet
                        string agentId = identityService.GetAgentIdAsync().GetAwaiter().GetResult();
                        return !string.IsNullOrEmpty(agentId) ? agentId : "unregistered-agent";
                    });

                    // Register services
                    services.AddSingleton<IAgentHealthMonitor, AgentHealthMonitor>();
                    services.AddSingleton<ILogCollectorFactory, LogCollectorFactory>();
                    services.AddSingleton<ILogNormalizer, LogNormalizer>();
                    services.AddSingleton<IEncryptionService, AesEncryptionService>();
                    services.AddSingleton<ILogForwarder, GrpcLogForwarder>();
                    
                    // Register configuration UI services
                    services.AddSingleton<AgentConfigurationLauncher>();

                    // Register and initialize log collectors
                    var collectors = agentSettings?.Collectors ?? new List<CollectorSettings>();
                    foreach (var collectorConfig in collectors)
                    {
                        if (collectorConfig.Enabled)
                        {
                            switch (collectorConfig.Type)
                            {
                                case "WindowsEventLog":
                                    services.AddSingleton<ILogCollector>(sp => 
                                    {
                                        var collector = new WindowsEventLogCollector(
                                            sp.GetRequiredService<ILogger<WindowsEventLogCollector>>(),
                                            sp.GetRequiredService<ILogNormalizer>());
                                        collector.Initialize(collectorConfig);
                                        return collector;
                                    });
                                    break;
                                case "LinuxSyslog":
                                    if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                                    {
                                        services.AddSingleton<ILogCollector>(sp => 
                                        {
                                            var collector = new LinuxSyslogCollector(
                                                sp.GetRequiredService<ILogger<LinuxSyslogCollector>>(),
                                                sp.GetRequiredService<ILogNormalizer>());
                                            collector.Initialize(collectorConfig);
                                            return collector;
                                        });
                                    }
                                    break;
                            }
                        }
                    }

                    // Register gRPC client
                    services.AddGrpcClient<SiemService.SiemServiceClient>((services, options) =>
                    {
                        var settings = hostContext.Configuration.GetSection("Agent").Get<AgentSettings>();
                        options.Address = new Uri(settings?.BackendGrpcUrl ?? "https://localhost:9596");
                    })
                    .ConfigurePrimaryHttpMessageHandler(() =>
                    {
                        return new HttpClientHandler
                        {
                            ServerCertificateCustomValidationCallback = 
                                HttpClientHandler.DangerousAcceptAnyServerCertificateValidator
                        };
                    })
                    .AddPolicyHandler(GetRetryPolicy());

                    // Register hosted service
                    services.AddHostedService<SiemAgentService>();
                })
                .ConfigureLogging((hostContext, logging) =>
                {
                    logging.ClearProviders();
                    logging.AddConfiguration(hostContext.Configuration.GetSection("Logging"));
                    logging.AddConsole();
                    logging.AddDebug();
                    
                    // Only add EventLog logging on Windows
                    if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                    {
                        logging.AddEventLog(options =>
                        {
                            options.SourceName = "AthalaSIEM Agent";
                            options.LogName = "Application";
                        });
                    }
                    
                    // Get the application data folder for logs
                    string appDataPath = Path.Combine(
                        Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData),
                        "Athala SIEM Agent");
                        
                    // Try to find the config file path from the host context
                    string? configFilePath = hostContext.Configuration["ConfigFilePath"];
                    bool isEmergencyConfig = hostContext.Configuration["IsEmergencyConfig"] == "true";
                    
                    // Get the log path from various sources
                    string? logPath = null;
                    
                    // First, check if the registry specifies a working directory
                    try
                    {
                        using var key = Microsoft.Win32.Registry.LocalMachine.OpenSubKey(
                            @"SYSTEM\CurrentControlSet\Services\AthalaSIEMAgent\Parameters");
                        if (key != null)
                        {
                            string? workingDir = key.GetValue("WorkingDirectory") as string;
                            if (!string.IsNullOrEmpty(workingDir) && Directory.Exists(workingDir))
                            {
                                // Try to use a logs subfolder in the working directory
                                string potentialLogPath = Path.Combine(workingDir, "logs");
                                try
                                {
                                    if (!Directory.Exists(potentialLogPath))
                                    {
                                        Directory.CreateDirectory(potentialLogPath);
                                    }
                                    logPath = potentialLogPath;
                                    Console.WriteLine($"Using registry-defined log path: {logPath}");
                                }
                                catch
                                {
                                    // If we can't create the logs directory in the working directory,
                                    // just use the working directory itself
                                    logPath = workingDir;
                                    Console.WriteLine($"Using registry-defined working directory for logs: {logPath}");
                                }
                            }
                        }
                    }
                    catch (Exception regEx)
                    {
                        Console.WriteLine($"Error reading registry for logs path: {regEx.Message}");
                    }
                    
                    // If we have a valid config file and couldn't find a registry path, try to use its directory
                    if (logPath == null && !string.IsNullOrEmpty(configFilePath) && !isEmergencyConfig)
                    {
                        try
                        {
                            string? configDir = Path.GetDirectoryName(configFilePath);
                            if (!string.IsNullOrEmpty(configDir) && Directory.Exists(configDir))
                            {
                                // Try to use a logs subfolder in the config directory
                                string potentialLogPath = Path.Combine(configDir, "logs");
                                if (!Directory.Exists(potentialLogPath))
                                {
                                    Directory.CreateDirectory(potentialLogPath);
                                }
                                logPath = potentialLogPath;
                                Console.WriteLine($"Using config directory for logs: {logPath}");
                            }
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Error creating logs directory near config file: {ex.Message}");
                        }
                    }
                    
                    // If we still don't have a log path, use the ProgramData folder
                    if (logPath == null)
                    {
                        try
                        {
                            string potentialLogPath = Path.Combine(appDataPath, "logs");
                            if (!Directory.Exists(potentialLogPath))
                            {
                                Directory.CreateDirectory(potentialLogPath);
                            }
                            logPath = potentialLogPath;
                            Console.WriteLine($"Using ProgramData for logs: {logPath}");
                        }
                        catch (Exception) // Properly discard unused exception variable
                        {
                            // Last resort - use temp directory
                            try
                            {
                                logPath = Path.Combine(Path.GetTempPath(), "AthalaSIEM", "logs");
                                Directory.CreateDirectory(logPath);
                                Console.WriteLine($"Using temp directory for logs: {logPath}");
                            }
                            catch
                            {
                                // Absolute last resort, just use temp path
                                logPath = Path.GetTempPath();
                                Console.WriteLine($"Using temp path for logs: {logPath}");
                            }
                        }
                    }
                    
                    // Add file logging with the determined path
                    logging.AddFile(Path.Combine(logPath, "agent-{Date}.log"));
                });

        private static void ConfigureServices(IServiceCollection services, IConfiguration configuration)
        {
            // Configure settings
            var agentSettings = configuration.GetSection("Agent").Get<AgentSettings>();
            services.AddSingleton(agentSettings ?? new AgentSettings());

            // Register agent identity service first since it provides the agent ID
            services.AddSingleton<IAgentIdentityService, AgentIdentityService>();
            
            // Register the agentId string dependency - needed by AgentHealthMonitor
            services.AddSingleton(serviceProvider =>
            {
                var identityService = serviceProvider.GetRequiredService<IAgentIdentityService>();
                // Get agent ID or use a default if not registered yet
                string agentId = identityService.GetAgentIdAsync().GetAwaiter().GetResult();
                return !string.IsNullOrEmpty(agentId) ? agentId : "unregistered-agent";
            });

            // Register services
            services.AddSingleton<IAgentHealthMonitor, AgentHealthMonitor>();
            services.AddSingleton<ILogCollectorFactory, LogCollectorFactory>();
            services.AddSingleton<ILogNormalizer, LogNormalizer>();
            services.AddSingleton<IEncryptionService, AesEncryptionService>();
            services.AddSingleton<ILogForwarder, GrpcLogForwarder>();
            
            // Register configuration UI services
            services.AddSingleton<AgentConfigurationLauncher>();

            // Register and initialize log collectors
            var collectors = agentSettings?.Collectors ?? new List<CollectorSettings>();
            foreach (var collectorConfig in collectors)
            {
                if (collectorConfig.Enabled)
                {
                    switch (collectorConfig.Type)
                    {
                        case "WindowsEventLog":
                            services.AddSingleton<ILogCollector>(sp => 
                            {
                                var collector = new WindowsEventLogCollector(
                                    sp.GetRequiredService<ILogger<WindowsEventLogCollector>>(),
                                    sp.GetRequiredService<ILogNormalizer>());
                                collector.Initialize(collectorConfig);
                                return collector;
                            });
                            break;
                        case "LinuxSyslog":
                            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                            {
                                services.AddSingleton<ILogCollector>(sp => 
                                {
                                    var collector = new LinuxSyslogCollector(
                                        sp.GetRequiredService<ILogger<LinuxSyslogCollector>>(),
                                        sp.GetRequiredService<ILogNormalizer>());
                                    collector.Initialize(collectorConfig);
                                    return collector;
                                });
                            }
                            break;
                    }
                }
            }

            // Register gRPC client
            services.AddGrpcClient<SiemService.SiemServiceClient>((services, options) =>
            {
                var settings = configuration.GetSection("Agent").Get<AgentSettings>();
                options.Address = new Uri(settings?.BackendGrpcUrl ?? "https://localhost:9596");
            })
            .ConfigurePrimaryHttpMessageHandler(() =>
            {
                return new HttpClientHandler
                {
                    ServerCertificateCustomValidationCallback = 
                        HttpClientHandler.DangerousAcceptAnyServerCertificateValidator
                };
            })
            .AddPolicyHandler(GetRetryPolicy());
        }

        /// <summary>
        /// Creates a retry policy for gRPC client
        /// </summary>
        private static IAsyncPolicy<HttpResponseMessage> GetRetryPolicy()
        {
            return HttpPolicyExtensions
                .HandleTransientHttpError()
                .OrResult(msg => msg.StatusCode == System.Net.HttpStatusCode.NotFound)
                .WaitAndRetryAsync(3, retryAttempt => TimeSpan.FromSeconds(Math.Pow(2, retryAttempt)));
        }
    }
}
