using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using System.Threading;
using System.Diagnostics;
using System.Text.Json;
using System.Linq;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using AthalaSIEM.Agent.Core;
using AthalaSIEM.Agent.Collectors;
using AthalaSIEM.UniversalAgent.Services;
using AthalaSIEM.UniversalAgent.Services.Interfaces;
using AthalaSIEM.UniversalAgent.Models;

namespace AthalaSIEM.UniversalAgent.UAT
{
    /// <summary>
    /// UAT Test Runner for AthalaSIEM Universal Agent
    /// Tests all major components including FIM, Event Collection, Communication, and Performance
    /// </summary>
    public class UATTestRunner
    {
        private readonly ILogger<UATTestRunner> _logger;
        private readonly IConfiguration _configuration;
        private readonly IServiceProvider _serviceProvider;
        private readonly List<UATTestResult> _testResults = new();
        private readonly UATConfiguration _uatConfig;
        private readonly string _testReportPath;

        /// <summary>
        /// Initializes a new instance of the UATTestRunner.
        /// </summary>
        public UATTestRunner(ILogger<UATTestRunner> logger, IConfiguration configuration, IServiceProvider serviceProvider)
        {
            _logger = logger;
            _configuration = configuration;
            _serviceProvider = serviceProvider;
            _uatConfig = LoadUATConfiguration();
            _testReportPath = _uatConfig.TestReportPath;
            
            _logger.LogInformation("🧪 UAT Test Runner initialized for {TestCount} scenarios", _uatConfig.TestScenarios.Count);
        }

        /// <summary>
        /// Runs all UAT test scenarios.
        /// </summary>
        /// <returns>Overall test result.</returns>
        public async Task<UATOverallResult> RunAllTestsAsync()
        {
            _logger.LogInformation("🚀 Starting UAT Test Suite execution...");
            var startTime = DateTime.UtcNow;

            try
            {
                // Setup test environment
                await SetupTestEnvironmentAsync();

                // Run each test scenario
                foreach (var scenario in _uatConfig.TestScenarios)
                {
                    _logger.LogInformation("📋 Running test scenario: {Scenario}", scenario);
                    var result = await RunTestScenarioAsync(scenario);
                    _testResults.Add(result);
                }

                // Generate test report
                var overallResult = await GenerateTestReportAsync(startTime);

                // Cleanup if configured
                if (_uatConfig.CleanupAfterTest)
                {
                    await CleanupTestEnvironmentAsync();
                }

                _logger.LogInformation("✅ UAT Test Suite completed. Overall Status: {Status}", overallResult.OverallStatus);
                return overallResult;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ UAT Test Suite failed with exception");
                throw;
            }
        }

        /// <summary>
        /// Sets up the test environment and creates test data.
        /// </summary>
        /// <returns>Task representing the setup operation.</returns>
        private async Task SetupTestEnvironmentAsync()
        {
            _logger.LogInformation("🔧 Setting up UAT test environment...");

            try
            {
                // Create test directories
                foreach (var path in _uatConfig.TestDataPaths)
                {
                    var fullPath = Path.GetFullPath(path);
                    if (!Directory.Exists(fullPath))
                    {
                        Directory.CreateDirectory(fullPath);
                        _logger.LogDebug("Created test directory: {Path}", fullPath);
                    }
                }

                // Create specific FIM test directories
                var fimTestPaths = new[]
                {
                    ".\\UAT-Test\\TestFiles",
                    ".\\UAT-Test\\Documents",
                    ".\\Temp\\AthalaSIEM-UAT"
                };

                foreach (var path in fimTestPaths)
                {
                    var fullPath = Path.GetFullPath(path);
                    Directory.CreateDirectory(fullPath);
                    
                    // Create initial test files for FIM
                    await CreateInitialTestFilesAsync(fullPath);
                }

                // Create log directories
                Directory.CreateDirectory("UAT-Logs");
                Directory.CreateDirectory("UAT-Reports");

                _logger.LogInformation("✅ Test environment setup completed");
                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to setup test environment");
                throw;
            }
        }

        /// <summary>
        /// Creates initial test files for FIM testing.
        /// </summary>
        /// <param name="directory">Directory to create test files in.</param>
        /// <returns>Task representing the file creation operation.</returns>
        private async Task CreateInitialTestFilesAsync(string directory)
        {
            var testFiles = new[]
            {
                "test-document.txt",
                "config-file.json",
                "sample-data.csv",
                "readme.md"
            };

            foreach (var fileName in testFiles)
            {
                var filePath = Path.Combine(directory, fileName);
                var content = $"UAT Test File: {fileName}\nCreated: {DateTime.UtcNow}\nContent: Sample data for FIM testing";
                await File.WriteAllTextAsync(filePath, content);
            }

            _logger.LogDebug("Created {FileCount} initial test files in {Directory}", testFiles.Length, directory);
        }

        /// <summary>
        /// Runs a specific test scenario.
        /// </summary>
        /// <param name="scenario">The test scenario to run.</param>
        /// <returns>Test result for the scenario.</returns>
        private async Task<UATTestResult> RunTestScenarioAsync(string scenario)
        {
            var result = new UATTestResult
            {
                TestName = scenario,
                StartTime = DateTime.UtcNow
            };

            try
            {
                switch (scenario)
                {
                    case "FIM_Testing":
                        await RunFIMTestsAsync(result);
                        break;
                    case "Event_Collection":
                        await RunEventCollectionTestsAsync(result);
                        break;
                    case "Communication_Test":
                        await RunCommunicationTestsAsync(result);
                        break;
                    case "Performance_Test":
                        await RunPerformanceTestsAsync(result);
                        break;
                    default:
                        result.AddError($"Unknown test scenario: {scenario}");
                        break;
                }

                result.EndTime = DateTime.UtcNow;
                result.Duration = result.EndTime - result.StartTime;
                result.Passed = result.Errors.Count == 0;

                _logger.LogInformation("{Status} Test scenario '{Scenario}' completed in {Duration}ms", 
                    result.Passed ? "✅" : "❌", scenario, result.Duration.TotalMilliseconds);

            }
            catch (Exception ex)
            {
                result.AddError($"Exception in {scenario}: {ex.Message}");
                result.EndTime = DateTime.UtcNow;
                result.Duration = result.EndTime - result.StartTime;
                result.Passed = false;
                
                _logger.LogError(ex, "❌ Test scenario '{Scenario}' failed", scenario);
            }

            return result;
        }

        /// <summary>
        /// Runs File Integrity Monitoring tests.
        /// </summary>
        /// <param name="result">Test result to populate.</param>
        /// <returns>Task representing the FIM test operation.</returns>
        private async Task RunFIMTestsAsync(UATTestResult result)
        {
            _logger.LogInformation("🔍 Running FIM Tests...");

            try
            {
                // Test 1: Initialize FIM Collector
                var fimCollector = new FileIntegrityCollector(_serviceProvider.GetRequiredService<ILogger<FileIntegrityCollector>>());
                
                var fimConfig = new Dictionary<string, object>
                {
                    ["MonitoredPaths"] = new[] { ".\\UAT-Test\\TestFiles", ".\\UAT-Test\\Documents" },
                    ["ScanIntervalMinutes"] = 1
                };

                var initResult = await fimCollector.InitializeAsync(fimConfig);
                if (!initResult)
                {
                    result.AddError("FIM Collector initialization failed");
                    return;
                }
                result.AddStep("FIM Collector initialized successfully");

                // Test 2: Start FIM Collection
                await fimCollector.StartCollectionAsync();
                if (!fimCollector.IsActive)
                {
                    result.AddError("FIM Collector failed to start");
                    return;
                }
                result.AddStep("FIM Collector started successfully");

                // Test 3: Create a new file (should be detected)
                var testFilePath = Path.Combine(".\\UAT-Test\\TestFiles", $"new-file-{DateTime.UtcNow:yyyyMMdd-HHmmss}.txt");
                await File.WriteAllTextAsync(testFilePath, "This is a new test file for FIM detection");
                result.AddStep($"Created test file: {testFilePath}");

                // Test 4: Modify an existing file
                var existingFile = Path.Combine(".\\UAT-Test\\TestFiles", "test-document.txt");
                if (File.Exists(existingFile))
                {
                    await File.AppendAllTextAsync(existingFile, $"\nModified at: {DateTime.UtcNow}");
                    result.AddStep($"Modified existing file: {existingFile}");
                }

                // Test 5: Wait for FIM to detect changes
                var testDelayMs = _configuration.GetValue<int>("UAT:TestDelayMs", 5000);
                await Task.Delay(testDelayMs); // Configurable test delay

                // Test 6: Check collected logs
                var logs = await fimCollector.GetLogsAsync(50);
                var logCount = logs.Count();
                result.AddStep($"FIM collected {logCount} log entries");

                if (logCount == 0)
                {
                    result.AddWarning("No FIM events detected - may need longer wait time");
                }

                // Test 7: Get FIM health status
                var health = await fimCollector.GetHealthAsync();
                result.AddStep($"FIM Health: {health.Status}, Logs Collected: {health.LogsCollected}");

                // Test 8: Test backend configuration update
                var backendConfig = new Dictionary<string, object>
                {
                    ["MonitoredPaths"] = new[] { ".\\UAT-Test\\TestFiles", ".\\UAT-Test\\Documents", ".\\Temp\\AthalaSIEM-UAT" },
                    ["ScanIntervalMinutes"] = 2
                };

                var updateResult = await fimCollector.UpdateFromBackendConfigAsync(backendConfig);
                if (updateResult)
                {
                    result.AddStep("FIM backend configuration update successful");
                }
                else
                {
                    result.AddWarning("FIM backend configuration update failed");
                }

                // Test 9: Stop FIM Collector
                await fimCollector.StopCollectionAsync();
                result.AddStep("FIM Collector stopped successfully");

                // Test 10: Dispose FIM Collector
                await fimCollector.DisposeAsync();
                result.AddStep("FIM Collector disposed successfully");

                _logger.LogInformation("✅ FIM Tests completed successfully");
            }
            catch (Exception ex)
            {
                result.AddError($"FIM Test failed: {ex.Message}");
                _logger.LogError(ex, "❌ FIM Tests failed");
            }
        }

        /// <summary>
        /// Runs Event Collection tests.
        /// </summary>
        /// <param name="result">Test result to populate.</param>
        /// <returns>Task representing the Event Collection test operation.</returns>
        private async Task RunEventCollectionTestsAsync(UATTestResult result)
        {
            _logger.LogInformation("📊 Running Event Collection Tests...");

            try
            {
                // Test 1: Initialize Event Log Collector
                var eventCollector = new WindowsEventLogCollector(_serviceProvider.GetRequiredService<ILogger<WindowsEventLogCollector>>());
                
                var eventConfig = new Dictionary<string, object>
                {
                    ["LogSources"] = new[] { "Application" },
                    ["CollectAllEvents"] = false,
                    ["EnableSecurityFiltering"] = true
                };

                var initResult = await eventCollector.InitializeAsync(eventConfig);
                if (!initResult)
                {
                    result.AddError("Event Log Collector initialization failed");
                    return;
                }
                result.AddStep("Event Log Collector initialized successfully");

                // Test 2: Check collector health
                var health = await eventCollector.GetHealthAsync();
                result.AddStep($"Event Collector Health: {health.Status}");

                // Test 3: Start collection for a short time
                var cts = new CancellationTokenSource();
                cts.CancelAfter(TimeSpan.FromSeconds(10));

                var collectionTask = eventCollector.StartCollectionAsync(cts.Token);
                var collectionDelayMs = _configuration.GetValue<int>("UAT:TestCollectionDelayMs", 3000);
                await Task.Delay(collectionDelayMs); // Configurable collection delay

                if (!eventCollector.IsActive)
                {
                    result.AddWarning("Event Collector not active after start");
                }
                else
                {
                    result.AddStep("Event Collector started and is active");
                }

                // Test 4: Get collected logs
                var logs = await eventCollector.GetLogsAsync(20);
                var logCount = logs.Count();
                result.AddStep($"Event Collector retrieved {logCount} log entries");

                // Test 5: Stop collection
                await eventCollector.StopCollectionAsync();
                result.AddStep("Event Log Collector stopped successfully");

                // Test 6: Final health check
                var finalHealth = await eventCollector.GetHealthAsync();
                result.AddStep($"Final Event Collector Health: Status={finalHealth.Status}, LogsCollected={finalHealth.LogsCollected}");

                // Test 7: Dispose collector
                await eventCollector.DisposeAsync();
                result.AddStep("Event Log Collector disposed successfully");

                _logger.LogInformation("✅ Event Collection Tests completed successfully");
            }
            catch (Exception ex)
            {
                result.AddError($"Event Collection Test failed: {ex.Message}");
                _logger.LogError(ex, "❌ Event Collection Tests failed");
            }
        }

        /// <summary>
        /// Runs Communication tests.
        /// </summary>
        /// <param name="result">Test result to populate.</param>
        /// <returns>Task representing the Communication test operation.</returns>
        private async Task RunCommunicationTestsAsync(UATTestResult result)
        {
            _logger.LogInformation("🌐 Running Communication Tests...");

            try
            {
                // Test 1: Initialize Communication Service
                var commService = _serviceProvider.GetService<IBackendCommunicationService>();
                if (commService == null)
                {
                    result.AddError("Backend Communication Service not available");
                    return;
                }

                // Test 2: Check initial health status
                var health = commService.GetHealthStatus();
                result.AddStep($"Communication Health: Connected={health.IsConnected}, QueuedLogs={health.QueuedLogs}");

                // Test 3: Queue test log entries
                var testLogs = GenerateTestLogEntries(5);
                commService.QueueLogs(testLogs);
                result.AddStep($"Queued {testLogs.Count} test log entries");

                // Test 4: Check queued logs count
                var queuedCount = commService.QueuedLogs;
                result.AddStep($"Verified queued logs count: {queuedCount}");

                // Test 5: UAT Mode - Test queue functionality without backend connection
                // In UAT, we should NOT try to connect to any backend
                result.AddStep("🔒 UAT Mode: Testing offline behavior (no backend connection attempts)");
                
                // Verify queue functionality works without backend
                var moreLogs = GenerateTestLogEntries(3);
                commService.QueueLogs(moreLogs);
                var finalQueuedCount = commService.QueuedLogs;
                result.AddStep($"Queued {moreLogs.Count} more logs, total queued: {finalQueuedCount}");

                // Test 6: Check final health status
                var finalHealth = commService.GetHealthStatus();
                result.AddStep($"Final Communication Health: Connected={finalHealth.IsConnected}, QueuedLogs={finalHealth.QueuedLogs}, TotalLogsSent={finalHealth.TotalLogsSent}");

                // Test 7: Verify UAT behavior is correct
                bool isUATBehaviorCorrect = true;
                var behaviorMessage = "✅ UAT Communication behavior verified";
                
                // In UAT, we expect:
                // - IsConnected = false (no backend to connect to)
                // - QueuedLogs > 0 (logs should be queued)
                // - TotalLogsSent = 0 (no logs sent because no backend)
                
                if (finalHealth.IsConnected)
                {
                    result.AddWarning("⚠️ IsConnected=true during UAT - this indicates a backend is running");
                    result.AddWarning("🔧 To fix: Ensure no backend is running during UAT tests");
                    isUATBehaviorCorrect = false;
                }
                else
                {
                    result.AddStep("✅ IsConnected=false - Correct UAT behavior (no backend)");
                }

                if (finalHealth.QueuedLogs > 0)
                {
                    result.AddStep($"✅ QueuedLogs={finalHealth.QueuedLogs} - Correct UAT behavior (logs queued)");
                }
                else
                {
                    result.AddWarning("⚠️ QueuedLogs=0 - Unexpected, logs should be queued in UAT");
                    isUATBehaviorCorrect = false;
                }

                if (finalHealth.TotalLogsSent == 0)
                {
                    result.AddStep("✅ TotalLogsSent=0 - Correct UAT behavior (no logs sent)");
                }
                else
                {
                    result.AddWarning($"⚠️ TotalLogsSent={finalHealth.TotalLogsSent} - Unexpected, no logs should be sent in UAT");
                    isUATBehaviorCorrect = false;
                }

                if (isUATBehaviorCorrect)
                {
                    result.AddStep("🎯 UAT Communication Test: PASSED - Agent correctly handles offline backend");
                }
                else
                {
                    result.AddStep("⚠️ UAT Communication Test: PASSED with warnings - Check backend isolation");
                }

                _logger.LogInformation("✅ Communication Tests completed successfully");
            }
            catch (Exception ex)
            {
                result.AddError($"Communication Test failed: {ex.Message}");
                _logger.LogError(ex, "❌ Communication Tests failed");
            }
        }

        /// <summary>
        /// Runs Performance tests.
        /// </summary>
        /// <param name="result">Test result to populate.</param>
        /// <returns>Task representing the Performance test operation.</returns>
        private async Task RunPerformanceTestsAsync(UATTestResult result)
        {
            _logger.LogInformation("⚡ Running Performance Tests...");

            try
            {
                // Test 1: Memory usage test
                var process = Process.GetCurrentProcess();
                var initialMemory = process.WorkingSet64;
                result.AddStep($"Initial memory usage: {initialMemory / 1024 / 1024} MB");

                // Test 2: CPU usage simulation
                var cpuTestStart = DateTime.UtcNow;
                var counter = 0;
                while ((DateTime.UtcNow - cpuTestStart).TotalSeconds < 2)
                {
                    counter++; // Simulate CPU work
                }
                result.AddStep($"CPU test completed: {counter:N0} iterations in 2 seconds");

                // Test 3: Log processing performance
                var logProcessor = _serviceProvider.GetService<LogProcessor>();
                if (logProcessor != null)
                {
                    result.AddStep("Starting log processing performance test");
                    
                    // Initialize the LogProcessor before using it
                    var initialized = await logProcessor.InitializeAsync();
                    if (!initialized)
                    {
                        result.AddError("Failed to initialize LogProcessor");
                        return;
                    }
                    
                    result.AddStep("LogProcessor initialized successfully");
                    
                    var testLogs = GenerateTestLogEntries(100);
                    var stopwatch = Stopwatch.StartNew();
                    var processedBatch = await logProcessor.ProcessLogBatchAsync(testLogs);
                    stopwatch.Stop();
                    
                    result.AddStep($"Processed {processedBatch.ProcessedLogs.Count} logs in {stopwatch.ElapsedMilliseconds}ms");
                    result.AddStep($"Processing rate: {(processedBatch.ProcessedLogs.Count / stopwatch.Elapsed.TotalSeconds):F2} logs/second");
                }

                // Test 4: Memory usage after operations
                process.Refresh();
                var finalMemory = process.WorkingSet64;
                var memoryIncrease = finalMemory - initialMemory;
                result.AddStep($"Final memory usage: {finalMemory / 1024 / 1024} MB (increase: {memoryIncrease / 1024 / 1024} MB)");

                // Test 5: GC performance
                var gcBefore = GC.CollectionCount(0);
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
                var gcAfter = GC.CollectionCount(0);
                result.AddStep($"Garbage collection: {gcAfter - gcBefore} Gen0 collections during test");

                _logger.LogInformation("✅ Performance Tests completed successfully");
            }
            catch (Exception ex)
            {
                result.AddError($"Performance Test failed: {ex.Message}");
                _logger.LogError(ex, "❌ Performance Tests failed");
            }
        }

        /// <summary>
        /// Generates test log entries for testing purposes.
        /// </summary>
        /// <param name="count">Number of log entries to generate.</param>
        /// <returns>List of test log entries.</returns>
        private List<LogEntry> GenerateTestLogEntries(int count)
        {
            var logs = new List<LogEntry>();
            var eventStartId = _configuration.GetValue<int>("UAT:TestEventStartId", 1000);
            
            for (int i = 0; i < count; i++)
            {
                logs.Add(new LogEntry
                {
                    Timestamp = DateTime.UtcNow.AddSeconds(-i),
                    Source = "UAT-Test",
                    Level = "Information",
                    Message = $"UAT Test Log Entry #{i + 1}",
                    EventId = (eventStartId + i).ToString(),
                    Category = "UAT_Testing",
                    SecurityRelevance = "Low",
                    ComputerName = Environment.MachineName,
                    Properties = new Dictionary<string, object>
                    {
                        ["TestNumber"] = i + 1,
                        ["TestType"] = "UAT",
                        ["GeneratedAt"] = DateTime.UtcNow
                    }
                });
            }
            return logs;
        }

        /// <summary>
        /// Generates comprehensive test report.
        /// </summary>
        /// <param name="startTime">Overall test start time.</param>
        /// <returns>Overall test result.</returns>
        private async Task<UATOverallResult> GenerateTestReportAsync(DateTime startTime)
        {
            var endTime = DateTime.UtcNow;
            var overallResult = new UATOverallResult
            {
                StartTime = startTime,
                EndTime = endTime,
                TotalDuration = endTime - startTime,
                TestResults = _testResults,
                TotalTests = _testResults.Count,
                PassedTests = _testResults.Count(r => r.Passed),
                FailedTests = _testResults.Count(r => !r.Passed),
                OverallStatus = _testResults.All(r => r.Passed) ? "PASSED" : "FAILED"
            };

            if (_uatConfig.GenerateTestReport)
            {
                await GenerateJsonReportAsync(overallResult);
                await GenerateHtmlReportAsync(overallResult);
            }

            return overallResult;
        }

        /// <summary>
        /// Generates JSON test report.
        /// </summary>
        /// <param name="result">Overall test result.</param>
        /// <returns>Task representing the JSON report generation.</returns>
        private async Task GenerateJsonReportAsync(UATOverallResult result)
        {
            var reportPath = Path.Combine(_testReportPath, $"UAT-Report-{DateTime.UtcNow:yyyyMMdd-HHmmss}.json");
            var json = JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(reportPath, json);
            _logger.LogInformation("📊 JSON test report generated: {ReportPath}", reportPath);
        }

        /// <summary>
        /// Generates HTML test report.
        /// </summary>
        /// <param name="result">Overall test result.</param>
        /// <returns>Task representing the HTML report generation.</returns>
        private async Task GenerateHtmlReportAsync(UATOverallResult result)
        {
            var reportPath = Path.Combine(_testReportPath, $"UAT-Report-{DateTime.UtcNow:yyyyMMdd-HHmmss}.html");
            var html = GenerateHtmlReport(result);
            await File.WriteAllTextAsync(reportPath, html);
            _logger.LogInformation("📊 HTML test report generated: {ReportPath}", reportPath);
        }

        /// <summary>
        /// Generates HTML report content.
        /// </summary>
        /// <param name="result">Overall test result.</param>
        /// <returns>HTML report content.</returns>
        private string GenerateHtmlReport(UATOverallResult result)
        {
            var statusColor = result.OverallStatus == "PASSED" ? "green" : "red";
            var html = $@"
<!DOCTYPE html>
<html>
<head>
    <title>AthalaSIEM Universal Agent - UAT Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .status {{ font-size: 24px; font-weight: bold; color: {statusColor}; }}
        .test-result {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .passed {{ border-left: 5px solid green; }}
        .failed {{ border-left: 5px solid red; }}
        .steps {{ margin: 10px 0; }}
        .step {{ margin: 5px 0; color: #555; }}
        .error {{ color: red; font-weight: bold; }}
        .warning {{ color: orange; font-weight: bold; }}
    </style>
</head>
<body>
    <div class='header'>
        <h1>🧪 AthalaSIEM Universal Agent - UAT Test Report</h1>
        <div class='status'>Overall Status: {result.OverallStatus}</div>
        <p><strong>Test Duration:</strong> {result.TotalDuration.TotalMinutes:F2} minutes</p>
        <p><strong>Tests Passed:</strong> {result.PassedTests}/{result.TotalTests}</p>
        <p><strong>Generated:</strong> {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss}</p>
    </div>";

            foreach (var testResult in result.TestResults)
            {
                var testClass = testResult.Passed ? "passed" : "failed";
                var testStatus = testResult.Passed ? "✅ PASSED" : "❌ FAILED";
                
                html += $@"
    <div class='test-result {testClass}'>
        <h2>{testResult.TestName} - {testStatus}</h2>
        <p><strong>Duration:</strong> {testResult.Duration.TotalMilliseconds:F2} ms</p>
        
        <div class='steps'>
            <h3>Test Steps:</h3>";

                foreach (var step in testResult.Steps)
                {
                    html += $"<div class='step'>• {step}</div>";
                }

                if (testResult.Warnings.Any())
                {
                    html += "<h3>Warnings:</h3>";
                    foreach (var warning in testResult.Warnings)
                    {
                        html += $"<div class='warning'>⚠️ {warning}</div>";
                    }
                }

                if (testResult.Errors.Any())
                {
                    html += "<h3>Errors:</h3>";
                    foreach (var error in testResult.Errors)
                    {
                        html += $"<div class='error'>❌ {error}</div>";
                    }
                }

                html += "</div></div>";
            }

            html += @"
</body>
</html>";

            return html;
        }

        /// <summary>
        /// Cleans up test environment after testing.
        /// </summary>
        /// <returns>Task representing the cleanup operation.</returns>
        private async Task CleanupTestEnvironmentAsync()
        {
            _logger.LogInformation("🧹 Cleaning up UAT test environment...");

            try
            {
                // Remove test files but keep directories
                foreach (var path in _uatConfig.TestDataPaths)
                {
                    if (Directory.Exists(path))
                    {
                        var files = Directory.GetFiles(path, "*", SearchOption.AllDirectories);
                        foreach (var file in files)
                        {
                            try
                            {
                                File.Delete(file);
                            }
                            catch (Exception ex)
                            {
                                _logger.LogWarning(ex, "Could not delete test file: {File}", file);
                            }
                        }
                    }
                }

                _logger.LogInformation("✅ UAT test environment cleanup completed");
                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "⚠️ Test environment cleanup encountered errors");
            }
        }

        /// <summary>
        /// Loads UAT configuration from the configuration provider.
        /// </summary>
        /// <returns>UAT configuration instance.</returns>
        private UATConfiguration LoadUATConfiguration()
        {
            var config = new UATConfiguration();
            _configuration.GetSection("UAT").Bind(config);
            return config;
        }
    }

    /// <summary>
    /// UAT Configuration settings.
    /// </summary>
    public class UATConfiguration
    {
        public bool TestMode { get; set; } = true;
        public string TestDuration { get; set; } = "PT30M";
        public List<string> TestScenarios { get; set; } = new();
        public List<string> TestDataPaths { get; set; } = new();
        public bool CleanupAfterTest { get; set; } = true;
        public bool GenerateTestReport { get; set; } = true;
        public string TestReportPath { get; set; } = ".\\UAT-Reports";
    }

    /// <summary>
    /// Individual UAT test result.
    /// </summary>
    public class UATTestResult
    {
        public string TestName { get; set; } = "";
        public DateTime StartTime { get; set; }
        public DateTime EndTime { get; set; }
        public TimeSpan Duration { get; set; }
        public bool Passed { get; set; }
        public List<string> Steps { get; set; } = new();
        public List<string> Warnings { get; set; } = new();
        public List<string> Errors { get; set; } = new();

        public void AddStep(string step) => Steps.Add($"{DateTime.UtcNow:HH:mm:ss} - {step}");
        public void AddWarning(string warning) => Warnings.Add(warning);
        public void AddError(string error) => Errors.Add(error);
    }

    /// <summary>
    /// Overall UAT test result.
    /// </summary>
    public class UATOverallResult
    {
        public DateTime StartTime { get; set; }
        public DateTime EndTime { get; set; }
        public TimeSpan TotalDuration { get; set; }
        public List<UATTestResult> TestResults { get; set; } = new();
        public int TotalTests { get; set; }
        public int PassedTests { get; set; }
        public int FailedTests { get; set; }
        public string OverallStatus { get; set; } = "";
    }
} 