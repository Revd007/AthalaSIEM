using System;
using System.Threading.Tasks;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using AthalaSIEM.UniversalAgent.UAT;
using AthalaSIEM.UniversalAgent.Core;
using AthalaSIEM.Agent.Collectors;
using AthalaSIEM.UniversalAgent.Services;
using AthalaSIEM.UniversalAgent.Services.Interfaces;
using AthalaSIEM.UniversalAgent.Models;

namespace AthalaSIEM.UniversalAgent.UAT
{
    /// <summary>
    /// UAT test runner utility class
    /// Usage: Called from Program.cs --run-uat command
    /// </summary>
    public class RunUAT
    {
        public static async Task<int> RunUATTestsAsync(string[] args)
        {
            Console.WriteLine("🧪 AthalaSIEM Universal Agent - UAT Test Runner");
            Console.WriteLine("=================================================");
            
            try
            {
                // Build configuration with UAT settings
                var configuration = new ConfigurationBuilder()
                    .SetBasePath(AppContext.BaseDirectory)
                    .AddJsonFile("appsettings.json", optional: false)
                    .AddJsonFile("appsettings.uat.json", optional: false)
                    .AddEnvironmentVariables("ATHALA_")
                    .Build();

                // Build service provider with UAT services
                var serviceProvider = BuildServiceProvider(configuration);

                // Create and run UAT test runner
                var logger = serviceProvider.GetRequiredService<ILogger<UATTestRunner>>();
                var testRunner = new UATTestRunner(logger, configuration, serviceProvider);

                Console.WriteLine("🚀 Starting UAT Test Suite...");
                var result = await testRunner.RunAllTestsAsync();

                // Display results
                DisplayResults(result);

                // Return appropriate exit code
                return result.OverallStatus == "PASSED" ? 0 : 1;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ UAT Test Suite failed with exception: {ex.Message}");
                Console.WriteLine($"Details: {ex}");
                return 1;
            }
        }

        /// <summary>
        /// Builds the service provider with all necessary services for UAT testing.
        /// </summary>
        /// <param name="configuration">Configuration instance.</param>
        /// <returns>Configured service provider.</returns>
        private static ServiceProvider BuildServiceProvider(IConfiguration configuration)
        {
            var services = new ServiceCollection();

            // Add logging
            services.AddLogging(builder =>
            {
                builder.AddConfiguration(configuration.GetSection("Logging"));
                builder.AddConsole();
                builder.AddDebug();
            });

            // Add configuration
            services.AddSingleton(configuration);

            // Add HTTP client
            services.AddHttpClient();

            // Add core services
            services.AddSingleton<LogProcessor>();
            services.AddSingleton<CollectorManager>();

            // Add collectors
            services.AddSingleton<FileIntegrityCollector>();
            
            // Windows-specific collectors
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                services.AddSingleton<WindowsEventLogCollector>();
                services.AddSingleton<WindowsRegistryCollector>();
            }

            // Add communication service
            services.AddSingleton<IBackendCommunicationService, BackendCommunicationService>();

            // Add UAT test runner
            services.AddSingleton<UATTestRunner>();

            return services.BuildServiceProvider();
        }

        /// <summary>
        /// Displays test results in a formatted way.
        /// </summary>
        /// <param name="result">Overall test result.</param>
        private static void DisplayResults(UATOverallResult result)
        {
            Console.WriteLine();
            Console.WriteLine("📊 UAT Test Results Summary");
            Console.WriteLine("============================");
            Console.WriteLine($"Overall Status: {GetStatusEmoji(result.OverallStatus)} {result.OverallStatus}");
            Console.WriteLine($"Total Duration: {result.TotalDuration.TotalMinutes:F2} minutes");
            Console.WriteLine($"Tests Passed: {result.PassedTests}/{result.TotalTests}");
            Console.WriteLine($"Tests Failed: {result.FailedTests}");
            Console.WriteLine();

            foreach (var testResult in result.TestResults)
            {
                var statusEmoji = testResult.Passed ? "✅" : "❌";
                Console.WriteLine($"{statusEmoji} {testResult.TestName} - {(testResult.Passed ? "PASSED" : "FAILED")} ({testResult.Duration.TotalMilliseconds:F0}ms)");
                
                if (!testResult.Passed && testResult.Errors.Any())
                {
                    foreach (var error in testResult.Errors)
                    {
                        Console.WriteLine($"   ❌ {error}");
                    }
                }

                if (testResult.Warnings.Any())
                {
                    foreach (var warning in testResult.Warnings)
                    {
                        Console.WriteLine($"   ⚠️ {warning}");
                    }
                }
            }

            Console.WriteLine();
            if (result.OverallStatus == "PASSED")
            {
                Console.WriteLine("🎉 All UAT tests completed successfully!");
                Console.WriteLine("📊 Check UAT-Reports directory for detailed reports.");
            }
            else
            {
                Console.WriteLine("❌ Some UAT tests failed. Please review the errors above.");
                Console.WriteLine("📊 Check UAT-Reports directory for detailed reports.");
            }
        }

        /// <summary>
        /// Gets appropriate emoji for test status.
        /// </summary>
        /// <param name="status">Test status string.</param>
        /// <returns>Emoji representation.</returns>
        private static string GetStatusEmoji(string status)
        {
            return status switch
            {
                "PASSED" => "✅",
                "FAILED" => "❌",
                _ => "❓"
            };
        }
    }
} 
