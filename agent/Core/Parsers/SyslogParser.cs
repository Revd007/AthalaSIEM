using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using AthalaSIEM.Agent.Core.Pipeline;

namespace AthalaSIEM.Agent.Core.Parsers;

public class SyslogParser : BaseParser
{
    private readonly ILogger<SyslogParser> _logger;
    private static readonly Regex Rfc3164Regex = new(
        @"^<(\d+)>(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+([\w\.-]+)\s+(.*)$",
        RegexOptions.Compiled);

    public SyslogParser(ILogger<SyslogParser> logger)
    {
        _logger = logger;
    }

    public override string Name => "SyslogParser";

    public override bool CanParse(IRawEvent rawEvent)
    {
        return rawEvent.SourceType.Equals("Syslog", StringComparison.OrdinalIgnoreCase) ||
               rawEvent.CollectorName.Contains("Syslog", StringComparison.OrdinalIgnoreCase);
    }

    protected override IParsedEvent ParseInternal(IRawEvent rawEvent)
    {
        var structuredData = new Dictionary<string, object>();
        var text = System.Text.Encoding.UTF8.GetString(rawEvent.RawData);

        var match = Rfc3164Regex.Match(text);
        if (match.Success)
        {
            structuredData["priority"] = match.Groups[1].Value;
            structuredData["timestamp"] = match.Groups[2].Value;
            structuredData["hostname"] = match.Groups[3].Value;
            structuredData["message"] = match.Groups[4].Value;

            var priority = int.Parse(match.Groups[1].Value);
            var facility = priority / 8;
            var severity = priority % 8;

            structuredData["facility"] = facility;
            structuredData["severity"] = severity;
        }
        else
        {
            structuredData["raw_message"] = text;
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
