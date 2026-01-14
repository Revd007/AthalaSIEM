using System.Collections.Generic;
using System.Threading.Tasks;
using AthalaSIEM.UniversalAgent.Core.Normalizer;

namespace AthalaSIEM.UniversalAgent.Core.Exporter
{
    /// <summary>
    /// Interface for event exporters
    /// Exporters deliver normalized events to destinations
    /// 
    /// HARD RULES (from specification):
    /// - Exporter MUST NOT mutate events
    /// - Exporter MUST NOT detect
    /// - Exporter MUST NOT parse or normalize
    /// - Exporter only delivers events
    /// </summary>
    public interface IExporter
    {
        /// <summary>
        /// Gets the name of the exporter
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Gets the export mode (File, Console, HTTP, gRPC)
        /// </summary>
        string Mode { get; }

        /// <summary>
        /// Exports a batch of normalized events
        /// </summary>
        /// <param name="events">The normalized events to export</param>
        /// <returns>Export result with success count and errors</returns>
        Task<ExportResult> ExportAsync(IEnumerable<AthalaEcsLiteEvent> events);

        /// <summary>
        /// Initializes the exporter
        /// </summary>
        Task<bool> InitializeAsync();

        /// <summary>
        /// Gets metrics about export operations
        /// </summary>
        /// <returns>Dictionary of metrics</returns>
        Dictionary<string, object> GetMetrics();
    }

    /// <summary>
    /// Export result
    /// </summary>
    public class ExportResult
    {
        public bool Success { get; set; }
        public int ExportedCount { get; set; }
        public int FailedCount { get; set; }
        public string? ErrorMessage { get; set; }
    }
}
