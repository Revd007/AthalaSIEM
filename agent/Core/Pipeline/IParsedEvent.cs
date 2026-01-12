using System;
using System.Collections.Generic;

namespace AthalaSIEM.Agent.Core.Pipeline;

public interface IParsedEvent
{
    string Id { get; }
    DateTime Timestamp { get; }
    string CollectorName { get; }
    string SourceType { get; }
    Dictionary<string, object> StructuredData { get; }
    IRawEvent OriginalRawEvent { get; }
}
