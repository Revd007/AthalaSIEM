using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Microsoft.Extensions.Logging;
using AthalaSIEM.UniversalAgent.Core.Parser;
using AthalaSIEM.UniversalAgent.Core.Normalizer;
using AthalaSIEM.UniversalAgent.Core.Exporter;
using AthalaSIEM.UniversalAgent.Core.Collector;
using AthalaSIEM.UniversalAgent.Core;

namespace AthalaSIEM.UniversalAgent.Core.Pipeline
{
    /// <summary>
    /// Pipeline Factory
    /// Creates configured pipelines for different deployment modes
    /// 
    /// Modes:
    /// - Test: File/Console export, no backend required
    /// - Production: HTTP/gRPC export to backend
    /// - Hybrid: Multiple exporters for redundancy
    /// </summary>
    public class PipelineFactory
    {
        private readonly ILoggerFactory _loggerFactory;
        private readonly string _agentId;
        private readonly string _agentName;
        private readonly string _agentVersion;

        public PipelineFactory(
            ILoggerFactory loggerFactory,
            string agentId,
            string agentName,
            string agentVersion = "1.0.0")
        {
            _loggerFactory = loggerFactory ?? throw new ArgumentNullException(nameof(loggerFactory));
            _agentId = agentId ?? throw new ArgumentNullException(nameof(agentId));
            _agentName = agentName ?? throw new ArgumentNullException(nameof(agentName));
            _agentVersion = agentVersion;
        }

        /// <summary>
        /// Create test mode pipeline (file export, no backend required)
        /// GOLDEN RULE: Agent MUST run without backend
        /// </summary>
        public EventPipeline CreateTestPipeline(string outputDirectory = "./test-output")
        {
            var logger = _loggerFactory.CreateLogger<EventPipeline>();

            // Create all available parsers
            var parsers = CreateParsers();

            // Create normalizer
            var normalizer = CreateNormalizer();

            // Create test mode exporters
            var exporters = new List<IExporter>
            {
                new FileExporter(
                    _loggerFactory.CreateLogger<FileExporter>(),
                    outputDirectory,
                    $"athalasiem-{DateTime.UtcNow:yyyyMMdd-HHmmss}.jsonl"),
                new ConsoleExporter(
                    _loggerFactory.CreateLogger<ConsoleExporter>(),
                    prettyPrint: false)
            };

            return new EventPipeline(logger, parsers, normalizer, exporters);
        }

        /// <summary>
        /// Create production pipeline (HTTP export to backend)
        /// </summary>
        public EventPipeline CreateProductionPipeline(string backendUrl, string? apiKey = null)
        {
            var logger = _loggerFactory.CreateLogger<EventPipeline>();

            // Create all available parsers
            var parsers = CreateParsers();

            // Create normalizer
            var normalizer = CreateNormalizer();

            // Create production exporters
            var exporters = new List<IExporter>
            {
                new HttpExporter(
                    _loggerFactory.CreateLogger<HttpExporter>(),
                    $"{backendUrl}/api/logs/batch",
                    apiKey,
                    maxRetries: 3,
                    batchSize: 100,
                    enableCompression: true)
            };

            return new EventPipeline(logger, parsers, normalizer, exporters);
        }

        /// <summary>
        /// Create hybrid pipeline (multiple exporters for redundancy)
        /// </summary>
        public EventPipeline CreateHybridPipeline(string backendUrl, string? apiKey, string fallbackDirectory)
        {
            var logger = _loggerFactory.CreateLogger<EventPipeline>();

            // Create all available parsers
            var parsers = CreateParsers();

            // Create normalizer
            var normalizer = CreateNormalizer();

            // Create hybrid exporters
            var exporters = new List<IExporter>
            {
                new HttpExporter(
                    _loggerFactory.CreateLogger<HttpExporter>(),
                    $"{backendUrl}/api/logs/batch",
                    apiKey),
                new FileExporter(
                    _loggerFactory.CreateLogger<FileExporter>(),
                    fallbackDirectory,
                    "fallback-events.jsonl",
                    appendMode: true)
            };

            return new EventPipeline(logger, parsers, normalizer, exporters);
        }

        /// <summary>
        /// Create all available parsers for the current platform
        /// </summary>
        private List<IParser> CreateParsers()
        {
            var parsers = new List<IParser>();

            // Platform-specific parsers
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                parsers.Add(new WindowsEventLogParser(_loggerFactory.CreateLogger<WindowsEventLogParser>()));
            }

            // Cross-platform parsers
            parsers.Add(new SyslogParser(_loggerFactory.CreateLogger<SyslogParser>()));
            parsers.Add(new JournalctlParser(_loggerFactory.CreateLogger<JournalctlParser>()));
            parsers.Add(new DockerEventParser(_loggerFactory.CreateLogger<DockerEventParser>()));
            parsers.Add(new NetworkDeviceParser(_loggerFactory.CreateLogger<NetworkDeviceParser>()));
            
            // Generic parser as fallback (should be last)
            parsers.Add(new GenericLogParser(_loggerFactory.CreateLogger<GenericLogParser>()));

            return parsers;
        }

        /// <summary>
        /// Create normalizer with host information
        /// </summary>
        private INormalizer CreateNormalizer()
        {
            var hostOs = new OsInfo
            {
                Name = GetOsName(),
                Version = Environment.OSVersion.Version.ToString(),
                Platform = GetPlatform(),
                Family = GetOsFamily()
            };

            return new AthalaEcsLiteNormalizer(
                _loggerFactory.CreateLogger<AthalaEcsLiteNormalizer>(),
                _agentId,
                _agentName,
                _agentVersion,
                Environment.MachineName,
                hostOs);
        }

        /// <summary>
        /// Create collectors for the current platform
        /// Uses existing collectors from agent-universal/Collectors/ via adapter
        /// </summary>
        public List<ICollector> CreateCollectorsFromLegacy(IEnumerable<ILogCollector> legacyCollectors)
        {
            var collectors = new List<ICollector>();
            foreach (var legacyCollector in legacyCollectors)
            {
                collectors.Add(new CollectorAdapter(legacyCollector, _loggerFactory.CreateLogger<CollectorAdapter>()));
            }
            return collectors;
        }

        private string GetOsName()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return "Windows";
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                return "Linux";
            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                return "macOS";
            return "Unknown";
        }

        private string GetPlatform()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return "windows";
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                return "linux";
            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                return "darwin";
            return "unknown";
        }

        private string GetOsFamily()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return "windows";
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                return "linux";
            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                return "darwin";
            return "unknown";
        }
    }
}
