using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using AthalaSIEM.Agent.Models;

namespace AthalaSIEM.Agent.Collectors
{
    /// <summary>
    /// Factory for creating log collectors
    /// </summary>
    public class LogCollectorFactory : ILogCollectorFactory
    {
        private readonly ILogger<LogCollectorFactory> _logger;
        private readonly ILoggerFactory _loggerFactory;
        private readonly ILogNormalizer _normalizer;
        private readonly Dictionary<string, Func<ILogCollector>> _collectorFactories;

        /// <summary>
        /// Creates a new instance of the LogCollectorFactory
        /// </summary>
        /// <param name="logger">Logger instance</param>
        /// <param name="loggerFactory">Logger factory</param>
        /// <param name="normalizer">Log normalizer</param>
        public LogCollectorFactory(
            ILogger<LogCollectorFactory> logger,
            ILoggerFactory loggerFactory,
            ILogNormalizer normalizer)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _loggerFactory = loggerFactory ?? throw new ArgumentNullException(nameof(loggerFactory));
            _normalizer = normalizer ?? throw new ArgumentNullException(nameof(normalizer));

            // Register collector factories
            _collectorFactories = new Dictionary<string, Func<ILogCollector>>(StringComparer.OrdinalIgnoreCase)
            {
                ["WindowsEventLog"] = () => new WindowsEventLogCollector(
                    _loggerFactory.CreateLogger<WindowsEventLogCollector>(),
                    _normalizer),
                
                ["Syslog"] = () => new SyslogCollector(
                    _loggerFactory.CreateLogger<SyslogCollector>(),
                    _normalizer),
                
                ["LinuxSyslog"] = () => new SyslogCollector(
                    _loggerFactory.CreateLogger<SyslogCollector>(),
                    _normalizer),
                
                ["FileIntegrity"] = () => new FileIntegrityCollector(
                    _loggerFactory.CreateLogger<FileIntegrityCollector>(),
                    _normalizer),

                ["Container"] = () => new ContainerCollector(
                    _loggerFactory.CreateLogger<ContainerCollector>(),
                    _normalizer),

                ["CloudServices"] = () => new CloudServicesCollector(
                    _loggerFactory.CreateLogger<CloudServicesCollector>(),
                    _normalizer),

                ["Database"] = () => new DatabaseCollector(
                    _loggerFactory.CreateLogger<DatabaseCollector>(),
                    _normalizer),

                ["IoT"] = () => new IoTCollector(
                    _loggerFactory.CreateLogger<IoTCollector>(),
                    _normalizer)
            };
        }

        /// <summary>
        /// Creates a log collector from the provided settings
        /// </summary>
        /// <param name="settings">Collector settings</param>
        /// <returns>A log collector instance or a NullLogCollector if creation fails</returns>
        public ILogCollector CreateCollector(CollectorSettings settings)
        {
            if (settings == null)
            {
                _logger.LogError("Collector settings cannot be null");
                return new NullLogCollector(_logger);
            }

            if (string.IsNullOrEmpty(settings.Type))
            {
                _logger.LogError("Collector type cannot be null or empty");
                throw new ArgumentException("Collector type cannot be null or empty", nameof(settings));
            }

            if (!_collectorFactories.TryGetValue(settings.Type, out var factory))
            {
                _logger.LogError("Unknown collector type: {CollectorType}", settings.Type);
                return new NullLogCollector(_logger, $"Unknown collector type: {settings.Type}");
            }

            try
            {
                var collector = factory();
                if (collector.Initialize(settings))
                {
                    _logger.LogInformation("Created and initialized collector of type {CollectorType}", settings.Type);
                    return collector;
                }
                else
                {
                    _logger.LogError("Failed to initialize collector of type {CollectorType}", settings.Type);
                    return new NullLogCollector(_logger, $"Failed to initialize collector of type {settings.Type}");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating collector of type {CollectorType}", settings.Type);
                return new NullLogCollector(_logger, $"Error creating collector: {ex.Message}");
            }
        }

        /// <summary>
        /// Registers a collector factory
        /// </summary>
        /// <param name="collectorType">Collector type</param>
        /// <param name="factory">Factory function</param>
        public void RegisterCollectorFactory(string collectorType, Func<ILogCollector> factory)
        {
            if (string.IsNullOrWhiteSpace(collectorType))
            {
                throw new ArgumentException("Collector type cannot be null or empty", nameof(collectorType));
            }

            if (factory == null)
            {
                throw new ArgumentNullException(nameof(factory));
            }

            _collectorFactories[collectorType] = factory;
            _logger.LogInformation("Registered collector factory for type {CollectorType}", collectorType);
        }
        
        /// <summary>
        /// Checks if a collector type is supported
        /// </summary>
        /// <param name="collectorType">The collector type to check</param>
        /// <returns>True if the collector type is supported, otherwise false</returns>
        public bool IsCollectorTypeSupported(string collectorType)
        {
            if (string.IsNullOrWhiteSpace(collectorType))
            {
                return false;
            }
            
            return _collectorFactories.ContainsKey(collectorType);
        }

        /// <summary>
        /// Gets all supported collector types
        /// </summary>
        /// <returns>List of supported collector types</returns>
        public IEnumerable<string> GetSupportedCollectorTypes()
        {
            return _collectorFactories.Keys;
        }
    }

    /// <summary>
    /// A null object implementation of ILogCollector that does nothing
    /// </summary>
    internal class NullLogCollector : ILogCollector
    {
        private readonly ILogger _logger;
        private readonly string _errorMessage;

        public string CollectorType => "NullCollector";

        public CollectorStatus Status => CollectorStatus.Error;

        public string ErrorMessage => _errorMessage;

        public event EventHandler<NormalizedLogEntry>? LogCollected;

        public NullLogCollector(ILogger logger, string errorMessage = "Null collector - no implementation available")
        {
            _logger = logger;
            _errorMessage = errorMessage;
            _logger.LogWarning("Created NullLogCollector: {ErrorMessage}", _errorMessage);
            
            // Raise a dummy log event to avoid the CS0067 warning
            RaiseDummyLog();
        }

        private void RaiseDummyLog()
        {
            // Only raise if there are subscribers
            if (LogCollected != null)
            {
                var dummyLog = new NormalizedLogEntry
                {
                    Id = Guid.NewGuid().ToString(),
                    Timestamp = DateTime.UtcNow,
                    Source = "NullLogCollector",
                    Severity = "Warning",
                    Message = $"Null collector created: {_errorMessage}",
                    Category = "System"
                };
                
                LogCollected.Invoke(this, dummyLog);
            }
        }

        public bool Initialize(CollectorSettings settings) => false;

        public Task<int> CollectLogsAsync(CancellationToken cancellationToken) => Task.FromResult(0);

        public Task PauseAsync() => Task.CompletedTask;

        public Task ResumeAsync() => Task.CompletedTask;

        public Task StartAsync() => Task.CompletedTask;

        public Task StopAsync() => Task.CompletedTask;
    }
} 