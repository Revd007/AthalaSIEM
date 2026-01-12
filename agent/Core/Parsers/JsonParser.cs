using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using AthalaSIEM.Agent.Core.Pipeline;

namespace AthalaSIEM.Agent.Core.Parsers;

public class JsonParser : BaseParser
{
    private readonly ILogger<JsonParser> _logger;

    public JsonParser(ILogger<JsonParser> logger)
    {
        _logger = logger;
    }

    public override string Name => "JsonParser";

    public override bool CanParse(IRawEvent rawEvent)
    {
        if (rawEvent.RawData == null || rawEvent.RawData.Length == 0)
            return false;

        try
        {
            var text = System.Text.Encoding.UTF8.GetString(rawEvent.RawData);
            JsonDocument.Parse(text);
            return true;
        }
        catch
        {
            return false;
        }
    }

    protected override IParsedEvent ParseInternal(IRawEvent rawEvent)
    {
        var text = System.Text.Encoding.UTF8.GetString(rawEvent.RawData);
        var jsonDoc = JsonDocument.Parse(text);
        
        var structuredData = new Dictionary<string, object>();
        FlattenJson(jsonDoc.RootElement, structuredData, "");

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

    private void FlattenJson(JsonElement element, Dictionary<string, object> result, string prefix)
    {
        switch (element.ValueKind)
        {
            case JsonValueKind.Object:
                foreach (var prop in element.EnumerateObject())
                {
                    var key = string.IsNullOrEmpty(prefix) ? prop.Name : $"{prefix}.{prop.Name}";
                    FlattenJson(prop.Value, result, key);
                }
                break;
            case JsonValueKind.Array:
                var index = 0;
                foreach (var item in element.EnumerateArray())
                {
                    FlattenJson(item, result, $"{prefix}[{index}]");
                    index++;
                }
                break;
            case JsonValueKind.String:
                result[prefix] = element.GetString() ?? string.Empty;
                break;
            case JsonValueKind.Number:
                if (element.TryGetInt64(out var intVal))
                    result[prefix] = intVal;
                else
                    result[prefix] = element.GetDouble();
                break;
            case JsonValueKind.True:
                result[prefix] = true;
                break;
            case JsonValueKind.False:
                result[prefix] = false;
                break;
            case JsonValueKind.Null:
                result[prefix] = null!;
                break;
        }
    }
}
