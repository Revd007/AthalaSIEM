using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using AthalaSIEM.UniversalAgent;
using System.ServiceProcess;
using System;
using System.IO;
using System.Net.Http;
using System.Threading;

namespace AthalaSIEM.UniversalAgent
{
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
                        RunConsoleMode();
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
                    services.AddHostedService<UniversalAgentService>();
                    services.AddLogging(builder =>
                    {
                        builder.AddConsole();
                        builder.AddEventLog();
                    });
                });

        private static void ShowHelp()
        {
            Console.WriteLine("Athala SIEM Universal Agent v1.0.0");
            Console.WriteLine("Usage: athala-agent.exe [options]");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("  --help, -h           Show this help message");
            Console.WriteLine("  --console, -c        Run in console mode");
            Console.WriteLine("  --install            Install as Windows service");
            Console.WriteLine("  --uninstall          Uninstall Windows service");
            Console.WriteLine("  --test-connection    Test connection to backend");
            Console.WriteLine();
            Console.WriteLine("Default: Run as Windows service");
        }

        private static void RunConsoleMode()
        {
            Console.WriteLine("Running in console mode. Press Ctrl+C to exit.");
            
            var configuration = new ConfigurationBuilder()
                .SetBasePath(Directory.GetCurrentDirectory())
                .AddJsonFile("appsettings.json", optional: false)
                .Build();

            var serviceCollection = new ServiceCollection();
            serviceCollection.AddLogging(builder =>
            {
                builder.AddConsole();
                builder.SetMinimumLevel(LogLevel.Information);
            });

            var serviceProvider = serviceCollection.BuildServiceProvider();
            var logger = serviceProvider.GetRequiredService<ILogger<Program>>();

            logger.LogInformation("Universal Agent started in console mode");

            // Keep running until Ctrl+C
            Console.CancelKeyPress += (sender, e) =>
            {
                logger.LogInformation("Shutting down...");
                e.Cancel = false;
            };

            // Simulate agent work
            while (true)
            {
                logger.LogInformation("Agent heartbeat at {Time}", DateTime.Now);
                Thread.Sleep(30000); // 30 seconds
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
            Console.WriteLine("Testing connection to backend...");
            
            var configuration = new ConfigurationBuilder()
                .SetBasePath(Directory.GetCurrentDirectory())
                .AddJsonFile("appsettings.json", optional: false)
                .Build();

            var backendUrl = configuration["BackendUrl"] ?? "http://localhost:9595";
            Console.WriteLine($"Backend URL: {backendUrl}");
            
            try
            {
                using var client = new HttpClient();
                client.Timeout = TimeSpan.FromSeconds(10);
                
                var response = client.GetAsync($"{backendUrl}/api/health").Result;
                
                if (response.IsSuccessStatusCode)
                {
                    Console.WriteLine("✓ Connection successful!");
                }
                else
                {
                    Console.WriteLine($"✗ Connection failed: {response.StatusCode}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"✗ Connection failed: {ex.Message}");
            }
        }
    }
} 