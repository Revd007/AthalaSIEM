using System.Threading.Tasks;
using AthalaSIEM.UniversalAgent.Models;

namespace AthalaSIEM.UniversalAgent.Core.Normalizer
{
    /// <summary>
    /// Interface for event normalizers
    /// Normalizers MUST map events to Athala ECS-lite schema
    /// Normalizers MUST NOT detect, parse, or enrich - only normalize schema
    /// </summary>
    public interface INormalizer
    {
        /// <summary>
        /// Gets the name of the normalizer
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Normalizes a parsed event to Athala ECS-lite schema
        /// </summary>
        /// <param name="parsedEvent">The parsed event (from Parser stage)</param>
        /// <returns>Normalized event in Athala ECS-lite format</returns>
        Task<AthalaEcsLiteEvent> NormalizeAsync(ParsedEvent parsedEvent);

        /// <summary>
        /// Gets metrics about normalization operations
        /// </summary>
        /// <returns>Dictionary of metrics</returns>
        System.Collections.Generic.Dictionary<string, object> GetMetrics();
    }
}
