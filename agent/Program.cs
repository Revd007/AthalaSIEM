using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;
using System;
using System.IO;
using System.Threading.Tasks;
using AthalaSIEM.Agent.Services;
using AthalaSIEM.Agent.Models;
using AthalaSIEM.Agent.Collectors;
using AthalaSIEM.Agent.Communication;
using AthalaSIEM.Agent.Security;
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
            var host = CreateHostBuilder(args).Build();
            await host.RunAsync();
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
                    services.Configure<AgentSettings>(hostContext.Configuration.GetSection("Agent"));

                    // Register services
                    services.AddSingleton<IAgentHealthMonitor, AgentHealthMonitor>();
                    services.AddSingleton<ILogCollectorFactory, LogCollectorFactory>();
                    services.AddSingleton<ILogNormalizer, LogNormalizer>();
                    services.AddSingleton<IEncryptionService, AesEncryptionService>();
                    services.AddSingleton<IAgentIdentityService, AgentIdentityService>();
                    services.AddSingleton<ILogForwarder, GrpcLogForwarder>();

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
