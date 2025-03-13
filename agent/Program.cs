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

namespace AthalaSIEM.Agent
{
    /// <summary>
    /// Main program entry point
    /// </summary>
    public class Program
    {
        /// <summary>
        /// Application entry point
        /// </summary>
        /// <param name="args">Command line arguments</param>
        public static async Task Main(string[] args)
        {
            // Check for configuration command
            bool showConfig = args.Any(arg => arg.Equals("--configure", StringComparison.OrdinalIgnoreCase) || 
                                              arg.Equals("-c", StringComparison.OrdinalIgnoreCase));
            
            // Skip configuration if running as a service
            if (showConfig && !Environment.UserInteractive)
            {
                Console.WriteLine("Cannot show configuration UI when running as a service. Please run with the --configure flag as a regular user.");
                return;
            }
            
            // Check for help command
            if (args.Any(arg => arg.Equals("--help", StringComparison.OrdinalIgnoreCase) || 
                                arg.Equals("-h", StringComparison.OrdinalIgnoreCase)))
            {
                ShowHelp();
                return;
            }
            
            // Build the host
            var host = CreateHostBuilder(args).Build();
            
            // If configuration is requested or this is first run in interactive mode
            if (showConfig)
            {
                await ShowConfigurationUI(host.Services);
            }
            else
            {
                // Check if this is first run and in interactive mode
                bool isInteractive = AgentConfigurationLauncher.IsInteractiveMode();
                if (isInteractive)
                {
                    var configLauncher = host.Services.GetRequiredService<AgentConfigurationLauncher>();
                    bool isFirstRun = configLauncher.IsFirstTimeInstallation();
                    bool isConfigured = await configLauncher.IsAgentConfiguredAsync();
                    
                    if (isFirstRun || !isConfigured)
                    {
                        await ShowConfigurationUI(host.Services);
                    }
                }
            }
            
            // Run the host
            await host.RunAsync();
        }
        
        /// <summary>
        /// Shows the configuration UI
        /// </summary>
        private static async Task ShowConfigurationUI(IServiceProvider services)
        {
            var logger = services.GetRequiredService<ILogger<Program>>();
            logger.LogInformation("Showing configuration UI");
            
            var configLauncher = services.GetRequiredService<AgentConfigurationLauncher>();
            bool isConfigured = await configLauncher.ShowConfigurationFormAsync(true);
            
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
                        options.Address = new Uri(settings?.BackendGrpcUrl ?? "https://localhost:5002");
                    })
                    .ConfigurePrimaryHttpMessageHandler(() =>
                    {
                        return new HttpClientHandler
                        {
                            ServerCertificateCustomValidationCallback = 
                                HttpClientHandler.DangerousAcceptAnyServerCertificateValidator
                        };
                    });

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
    }
}
