namespace AthalaSIEM.Agent.Core.Pipeline;

public class ParsedEvent : IParsedEvent
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public string CollectorName { get; set; } = string.Empty;
    public string SourceType { get; set; } = string.Empty;
    public Dictionary<string, object> StructuredData { get; set; } = new();
    public IRawEvent OriginalRawEvent { get; set; } = null!;
}
