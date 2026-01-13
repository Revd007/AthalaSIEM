using System;
using System.Collections.Generic;

namespace Backend.DTOs
{
    public class PlaybookDto
    {
        public string Id { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public string Author { get; set; } = string.Empty;
        public string Category { get; set; } = string.Empty;
        public string Status { get; set; } = "draft";
        public DateTime? LastRun { get; set; }
        public DateTime LastModified { get; set; }
        public List<PlaybookStepDto> Steps { get; set; } = new();
    }

    public class PlaybookStepDto
    {
        public string Id { get; set; } = string.Empty;
        public string Type { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public object? Config { get; set; }
    }

    public class CreatePlaybookRequest
    {
        public string Name { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public string? Author { get; set; }
        public string? Category { get; set; }
        public List<PlaybookStepDto>? Steps { get; set; }
    }

    public class PlaybookRunDto
    {
        public string Id { get; set; } = string.Empty;
        public string PlaybookId { get; set; } = string.Empty;
        public string PlaybookName { get; set; } = string.Empty;
        public string Status { get; set; } = "pending";
        public DateTime StartTime { get; set; }
        public DateTime? EndTime { get; set; }
        public List<PlaybookStepResultDto> Results { get; set; } = new();
    }

    public class PlaybookStepResultDto
    {
        public string StepId { get; set; } = string.Empty;
        public string StepName { get; set; } = string.Empty;
        public string Status { get; set; } = "pending";
        public string? Output { get; set; }
        public DateTime ExecutedAt { get; set; }
    }
}
