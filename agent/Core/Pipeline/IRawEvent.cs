using System;
using System.Collections.Generic;

namespace AthalaSIEM.Agent.Core.Pipeline;

public interface IRawEvent
{
    string Id { get; }
    DateTime Timestamp { get; }
    string CollectorName { get; }
    string SourceType { get; }
    byte[] RawData { get; }
    Dictionary<string, string> Metadata { get; }
}
