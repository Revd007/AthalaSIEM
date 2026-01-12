using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using AthalaSIEM.Agent.Core.Buffer;
using AthalaSIEM.Agent.Core.Collectors;
using AthalaSIEM.Agent.Core.Exporters;
using AthalaSIEM.Agent.Core.Normalizers;
using AthalaSIEM.Agent.Core.Parsers;
using AthalaSIEM.Agent.Communication;
using AthalaSIEM.Agent.Models;
using AthalaSIEM.Agent.Security;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace AthalaSIEM.Agent.Core.Pipeline;

public static class PipelineServiceRegistration
{
    public static void RegisterPipelineServices(
        IServiceCollection services,
        IConfiguration configuration,
        AgentSettings? agentSettings)
    {
        var agentId = agentSettings?.AgentId ?? Guid.NewGuid().ToString();
        var agentName = agentSettings?.AgentName ?? "AthalaSIEM Agent";
        var hostName = Environment.MachineName;

        // Register parsers
        services.AddSingleton<IParser, JsonParser>();
        services.AddSingleton<IParser, WindowsEventLogParser>();
        services.AddSingleton<IParser, SyslogParser>();

        // Register normalizer
        services.AddSingleton<INormalizer>(sp => new AthalaEcsNormalizer(
            sp.GetRequiredService<ILogger<AthalaEcsNormalizer>>(),
            agentId,
            agentName,
            hostName));

        // Register buffer
        var bufferPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData),
            "Athala SIEM Agent",
            "buffer",
            "events.json");

        services.AddSingleton<IBuffer>(sp => new DiskBackedBuffer(
            sp.GetRequiredService<ILogger<DiskBackedBuffer>>(),
            bufferPath,
            maxMemoryCount: 1000,
            maxDiskSizeBytes: 100 * 1024 * 1024));

        // Register collectors
        if (OperatingSystem.IsWindows())
        {
            services.AddSingleton<ICollector>(sp =>
            {
                var logNames = configuration.GetSection("Collectors:EventLogs:LogNames").Get<string[]>() 
                    ?? new[] { "Security", "System", "Application" };
                return new WindowsEventLogCollector(
                    sp.GetRequiredService<ILogger<WindowsEventLogCollector>>(),
                    logNames,
                    enabled: configuration.GetValue<bool>("Collectors:EventLogs:Enabled", true));
            });
        }

        if (OperatingSystem.IsLinux())
        {
            services.AddSingleton<ICollector>(sp => new JournalctlCollector(
                sp.GetRequiredService<ILogger<JournalctlCollector>>(),
                enabled: true));
        }

        services.AddSingleton<ICollector>(sp =>
        {
            var syslogPort = configuration.GetValue<int>("Platforms:Network:DefaultPort", 514);
            return new SyslogCollector(
                sp.GetRequiredService<ILogger<SyslogCollector>>(),
                syslogPort,
                enabled: configuration.GetValue<bool>("Collectors:Syslog:Enabled", true));
        });

        if (configuration.GetValue<bool>("Collectors:Container:Enabled", false))
        {
            services.AddSingleton<ICollector>(sp =>
            {
                var dockerHost = configuration.GetValue<string>("Collectors:Container:DockerHost", "unix:///var/run/docker.sock");
                return new DockerEventCollector(
                    sp.GetRequiredService<ILogger<DockerEventCollector>>(),
                    dockerHost,
                    enabled: true);
            });
        }

        // Register exporters
        var testMode = configuration.GetValue<bool>("Pipeline:TestMode", false);
        var outputPath = configuration.GetValue<string>("Pipeline:TestOutputPath", 
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData), 
                "Athala SIEM Agent", "test_output", "events.jsonl"));

        if (testMode)
        {
            services.AddSingleton<IExporter>(sp => new FileExporter(
                sp.GetRequiredService<ILogger<FileExporter>>(),
                outputPath,
                enabled: true));

            services.AddSingleton<IExporter>(sp => new ConsoleExporter(
                sp.GetRequiredService<ILogger<ConsoleExporter>>(),
                enabled: true));
        }
        else
        {
            services.AddSingleton<IExporter>(sp =>
            {
                var logForwarder = sp.GetService<ILogForwarder>();
                if (logForwarder == null)
                    throw new InvalidOperationException("ILogForwarder is required for gRPC exporter");
                
                return new GrpcExporter(
                    logForwarder,
                    sp.GetRequiredService<ILogger<GrpcExporter>>(),
                    enabled: true);
            });
        }

        // Register pipeline orchestrator
        services.AddHostedService<PipelineOrchestrator>();
    }
}
