using System.Collections.Generic;

namespace Backend.DTOs
{
    /// <summary>
    /// Request DTO for creating configuration from template
    /// </summary>
    public class CreateFromTemplateRequest
    {
        public string ConfigurationName { get; set; } = "";
        
        public List<string> TargetAgents { get; set; } = new();
        
        public Dictionary<string, string> Variables { get; set; } = new();
    }
}
