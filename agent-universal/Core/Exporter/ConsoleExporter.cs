using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using AthalaSIEM.UniversalAgent.Core.Normalizer;

namespace AthalaSIEM.UniversalAgent.Core.Exporter
{
    /// <summary>
    /// Console Exporter for Test/Debug Mode
    /// Outputs normalized events to console (stdout)
    /// 
    /// GOLDEN RULE: Agent MUST run without backend (test mode)
    /// This exporter enables debugging without backend connectivity
    /// </summary>
    public class ConsoleExporter : IExporter
    {
        private readonly ILogger<ConsoleExporter> _logger;
        private readonly bool _prettyPrint;
        private readonly JsonSerializerOptions _jsonOptions;

        // Metrics
        private long _eventsExported = 0;
        private long _exportErrors = 0;
        private readonly DateTime _startTime = DateTime.UtcNow;

        public string Name => "ConsoleExporter";
        public string Mode => "Console";

        public ConsoleExporter(ILogger<ConsoleExporter> logger, bool prettyPrint = false)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _prettyPrint = prettyPrint;

            _jsonOptions = new JsonSerializerOptions
            {
                WriteIndented = prettyPrint,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
            };
        }

        public Task<bool> InitializeAsync()
        {
            _logger.LogInformation("Console exporter initialized - PrettyPrint: {PrettyPrint}", _prettyPrint);
            return Task.FromResult(true);
        }

        public Task<ExportResult> ExportAsync(IEnumerable<AthalaEcsLiteEvent> events)
        {
            if (events == null)
            {
                throw new ArgumentNullException(nameof(events));
            }

            var eventList = events.ToList();
            var exported = 0;
            var failed = 0;

            foreach (var evt in eventList)
            {
                try
                {
                    var json = JsonSerializer.Serialize(evt, _jsonOptions);
                    Console.WriteLine(json);
                    exported++;
                    _eventsExported++;
                }
                catch (Exception ex)
                {
                    failed++;
                    _exportErrors++;
                    _logger.LogWarning(ex, "Error exporting event to console");
                }
            }

            return Task.FromResult(new ExportResult
            {
                Success = failed == 0,
                ExportedCount = exported,
                FailedCount = failed
            });
        }

        public Dictionary<string, object> GetMetrics()
        {
            var uptime = DateTime.UtcNow - _startTime;
            return new Dictionary<string, object>
            {
                ["Name"] = Name,
                ["Mode"] = Mode,
                ["PrettyPrint"] = _prettyPrint,
                ["EventsExported"] = _eventsExported,
                ["ExportErrors"] = _exportErrors,
                ["UptimeSeconds"] = uptime.TotalSeconds
            };
        }
    }
}
