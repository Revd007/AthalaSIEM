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
            // Set up configuration
            var configuration = new ConfigurationBuilder()
                .SetBasePath(Directory.GetCurrentDirectory())
                .AddJsonFile("appsettings.json", optional: true, reloadOnChange: true)
                .AddEnvironmentVariables()
                .AddCommandLine(args)
                .Build();

            // Initialize logging
            Log.Logger = new LoggerConfiguration()
                .MinimumLevel.Information()
                .WriteTo.Console()
                .WriteTo.File("logs/agent-.log", rollingInterval: Serilog.RollingInterval.Day)
                .CreateLogger();

            try
            {
                Log.Information("Starting Athala SIEM Agent");
                
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
                    bool success = await agentIdentityService.RegisterWithTokenAsync(token);
                    
                    if (success)
                    {
                        Log.Information("Agent registered successfully with deployment token");
                        
                        // Start the actual agent as a service
                        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                        {
                            // For Windows, we'll often be called from the installer, so just exit
                            // The Windows service will be managed by the service control manager
                            Log.Information("Installation completed. Agent service will start automatically.");
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
                        Log.Error("Failed to register agent with deployment token");
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
                .UseWindowsService(options =>
                {
                    options.ServiceName = "AthalaSIEM Agent";
                })
                .ConfigureAppConfiguration((hostContext, config) =>
                {
                    config.SetBasePath(Directory.GetCurrentDirectory());
                    config.AddJsonFile("appsettings.json", optional: false, reloadOnChange: true);
                    config.AddJsonFile($"appsettings.{hostContext.HostingEnvironment.EnvironmentName}.json", optional: true, reloadOnChange: true);
                    config.AddEnvironmentVariables();
                    config.AddCommandLine(args);
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

                    // Register gRPC client
                    services.AddGrpcClient<SiemService.SiemServiceClient>((services, options) =>
                    {
                        var settings = hostContext.Configuration.GetSection("Agent").Get<AgentSettings>();
                        options.Address = new Uri(settings?.BackendGrpcUrl ?? "https://localhost:7078");
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
                        logging.AddEventLog();
                    }
                    
                    logging.AddFile(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "logs", "agent-{Date}.log"));
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

            // Register gRPC client
            services.AddGrpcClient<SiemService.SiemServiceClient>((services, options) =>
            {
                var settings = configuration.GetSection("Agent").Get<AgentSettings>();
                options.Address = new Uri(settings?.BackendGrpcUrl ?? "https://localhost:7078");
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
