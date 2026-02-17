using System;
using System.Collections.Generic;
using System.Xml.Linq;
using Microsoft.Extensions.Logging;
using AthalaSIEM.Agent.Core.Pipeline;

namespace AthalaSIEM.Agent.Core.Parsers;

public class WindowsEventLogParser : BaseParser
{
    private readonly ILogger<WindowsEventLogParser> _logger;

    public WindowsEventLogParser(ILogger<WindowsEventLogParser> logger)
    {
        _logger = logger;
    }

    public override string Name => "WindowsEventLogParser";

    public override bool CanParse(IRawEvent rawEvent)
    {
        return rawEvent.SourceType.Equals("WindowsEventLog", StringComparison.OrdinalIgnoreCase) ||
               rawEvent.CollectorName.Contains("WindowsEventLog", StringComparison.OrdinalIgnoreCase);
    }

    protected override IParsedEvent ParseInternal(IRawEvent rawEvent)
    {
        var structuredData = new Dictionary<string, object>();
        var text = System.Text.Encoding.UTF8.GetString(rawEvent.RawData);

        try
        {
            var xml = XDocument.Parse(text);
            var ns = XNamespace.Get("http://schemas.microsoft.com/win/2004/08/events/event");

            var systemNode = xml.Root?.Element(ns + "System");
            if (systemNode != null)
            {
                var eventId = systemNode.Element(ns + "EventID")?.Value;
                var level = systemNode.Element(ns + "Level")?.Value;
                var task = systemNode.Element(ns + "Task")?.Value;
                var channel = systemNode.Element(ns + "Channel")?.Value;
                var computer = systemNode.Element(ns + "Computer")?.Value;
                var securityUserId = systemNode.Element(ns + "Security")?.Attribute("UserID")?.Value;

                if (!string.IsNullOrEmpty(eventId))
                    structuredData["event_id"] = eventId;
                if (!string.IsNullOrEmpty(level))
                    structuredData["level"] = level;
                if (!string.IsNullOrEmpty(task))
                    structuredData["task"] = task;
                if (!string.IsNullOrEmpty(channel))
                    structuredData["channel"] = channel;
                if (!string.IsNullOrEmpty(computer))
                    structuredData["computer"] = computer;
                if (!string.IsNullOrEmpty(securityUserId))
                    structuredData["security_user_id"] = securityUserId;
            }

            var eventDataNode = xml.Root?.Element(ns + "EventData");
            if (eventDataNode != null)
            {
                var dataIndex = 0;
                foreach (var data in eventDataNode.Elements(ns + "Data"))
                {
                    var name = data.Attribute("Name")?.Value ?? $"Data{dataIndex}";
                    structuredData[$"event_data.{name}"] = data.Value;
                    dataIndex++;
                }
            }

            // Extract human-readable message from EventData/Message node if present
            var messageNode = xml.Root?.Element(ns + "EventData")?.Element(ns + "Message");
            if (messageNode != null && !string.IsNullOrEmpty(messageNode.Value))
            {
                structuredData["message"] = messageNode.Value;
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to parse Windows Event Log XML for event {EventId}", rawEvent.Id);
            structuredData["raw_message"] = text;
        }

        // CRITICAL: Extract human-readable message from metadata if parser didn't find it in XML
        // The Core collector stores FormatDescription() result in metadata["message"]
        if (!structuredData.ContainsKey("message") && rawEvent.Metadata.TryGetValue("message", out var metadataMessage))
        {
            structuredData["message"] = metadataMessage;
        }

        // Fallback: if still no message, create a descriptive one
        if (!structuredData.ContainsKey("message") || string.IsNullOrEmpty(structuredData["message"]?.ToString()))
        {
            var eventId = structuredData.GetValueOrDefault("event_id")?.ToString() ?? "Unknown";
            var task = structuredData.GetValueOrDefault("task")?.ToString() ?? "Event";
            structuredData["message"] = $"Event ID {eventId}: {task}";
        }

        return new ParsedEvent
        {
            Id = rawEvent.Id,
            Timestamp = rawEvent.Timestamp,
            CollectorName = rawEvent.CollectorName,
            SourceType = rawEvent.SourceType,
            StructuredData = structuredData,
            OriginalRawEvent = rawEvent
        };
    }
}
