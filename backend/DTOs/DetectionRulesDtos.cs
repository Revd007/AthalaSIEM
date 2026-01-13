using System;
using System.Collections.Generic;

namespace Backend.DTOs
{
    public class SigmaRuleDto
    {
        public string Id { get; set; } = string.Empty;
        public string Title { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public string Status { get; set; } = "active";
        public string Level { get; set; } = "medium";
        public string Logsource { get; set; } = string.Empty;
        public List<string> Tags { get; set; } = new();
        public DateTime LastModified { get; set; }
        public int Matches { get; set; }
        public string Content { get; set; } = string.Empty;
    }

    public class YaraRuleDto
    {
        public string Id { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public string Category { get; set; } = string.Empty;
        public string Severity { get; set; } = "medium";
        public string Status { get; set; } = "active";
        public DateTime LastModified { get; set; }
        public int Matches { get; set; }
        public string Content { get; set; } = string.Empty;
    }

    public class CreateSigmaRuleRequest
    {
        public string Title { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public string? Status { get; set; }
        public string? Level { get; set; }
        public string? Logsource { get; set; }
        public List<string>? Tags { get; set; }
        public string Content { get; set; } = string.Empty;
    }

    public class CreateYaraRuleRequest
    {
        public string Name { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public string? Category { get; set; }
        public string? Severity { get; set; }
        public string? Status { get; set; }
        public string Content { get; set; } = string.Empty;
    }

    public class RuleTestResult
    {
        public string RuleId { get; set; } = string.Empty;
        public bool Success { get; set; }
        public int Matches { get; set; }
        public double ExecutionTime { get; set; }
        public DateTime TestedAt { get; set; }
    }
}
