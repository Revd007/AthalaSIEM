using System.Threading.Tasks;
using AthalaSIEM.UniversalAgent.Models;

namespace AthalaSIEM.UniversalAgent.Core.Parser
{
    /// <summary>
    /// Interface for event parsers
    /// Parsers decode and structure raw logs but do NOT normalize schema
    /// 
    /// HARD RULES (from specification):
    /// - Parser MUST NOT detect
    /// - Parser MUST NOT normalize schema (that's Normalizer's job)
    /// - Parser MUST NOT enrich (that's Enricher's job)
    /// - Parser outputs ParsedEvent (structured but not normalized)
    /// </summary>
    public interface IParser
    {
        /// <summary>
        /// Gets the name of the parser
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Gets the source type this parser handles
        /// </summary>
        string SourceType { get; }

        /// <summary>
        /// Parses a raw event into structured format
        /// </summary>
        /// <param name="rawEvent">The raw event from Collector</param>
        /// <returns>Parsed event (structured but not normalized)</returns>
        Task<Normalizer.ParsedEvent> ParseAsync(object rawEvent);

        /// <summary>
        /// Checks if this parser can handle the given raw event
        /// </summary>
        /// <param name="rawEvent">The raw event to check</param>
        /// <returns>True if this parser can handle the event</returns>
        bool CanParse(object rawEvent);

        /// <summary>
        /// Gets metrics about parsing operations
        /// </summary>
        /// <returns>Dictionary of metrics</returns>
        System.Collections.Generic.Dictionary<string, object> GetMetrics();
    }
}
