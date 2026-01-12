using System;
using System.Collections.Generic;

namespace AthalaSIEM.Agent.Core.Pipeline;

public class RawEvent : IRawEvent
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public string CollectorName { get; set; } = string.Empty;
    public string SourceType { get; set; } = string.Empty;
    public byte[] RawData { get; set; } = Array.Empty<byte>();
    public Dictionary<string, string> Metadata { get; set; } = new();
}
