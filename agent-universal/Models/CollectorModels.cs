using System;
using System.Collections.Generic;
using Microsoft.Win32;

namespace AthalaSIEM.UniversalAgent.Models
{
    /// <summary>
    /// Represents information about a process for behavioral analysis.
    /// </summary>
    public class ProcessInfo
    {
        /// <summary>
        /// Gets or sets the process ID.
        /// </summary>
        public int Id { get; set; }

        /// <summary>
        /// Gets or sets the process name.
        /// </summary>
        public string Name { get; set; } = "";

        /// <summary>
        /// Gets or sets the process start time.
        /// </summary>
        public DateTime StartTime { get; set; }

        /// <summary>
        /// Gets or sets the main module path.
        /// </summary>
        public string MainModule { get; set; } = "";

        /// <summary>
        /// Gets or sets the working set size.
        /// </summary>
        public long WorkingSet { get; set; }

        /// <summary>
        /// Gets or sets whether this is a baseline process.
        /// </summary>
        public bool IsBaselineProcess { get; set; }
    }

    /// <summary>
    /// Represents a command schedule configuration.
    /// </summary>
    public class CommandSchedule
    {
        /// <summary>
        /// Gets or sets the unique identifier for this command schedule.
        /// </summary>
        public string Id { get; set; } = "";

        /// <summary>
        /// Gets or sets the command to execute.
        /// </summary>
        public string Command { get; set; } = "";

        /// <summary>
        /// Gets or sets the command arguments.
        /// </summary>
        public string Arguments { get; set; } = "";

        /// <summary>
        /// Gets or sets the execution interval in minutes.
        /// </summary>
        public int IntervalMinutes { get; set; } = 60;

        /// <summary>
        /// Gets or sets whether this command schedule is enabled.
        /// </summary>
        public bool Enabled { get; set; } = true;

        /// <summary>
        /// Gets or sets the command timeout in seconds.
        /// </summary>
        public int TimeoutSeconds { get; set; } = 30;

        /// <summary>
        /// Gets or sets the description of this command.
        /// </summary>
        public string Description { get; set; } = "";

        /// <summary>
        /// Gets or sets the last execution time.
        /// </summary>
        public DateTime LastExecuted { get; set; }

        /// <summary>
        /// Gets or sets the execution count.
        /// </summary>
        public long ExecutionCount { get; set; }
    }

    /// <summary>
    /// Represents a configurable severity rule for file integrity monitoring.
    /// This replaces hardcoded path-based severity determination.
    /// </summary>
    public class SeverityRule
    {
        /// <summary>
        /// Gets or sets the severity level (Critical, High, Medium, Low).
        /// </summary>
        public string Severity { get; set; } = "Medium";

        /// <summary>
        /// Gets or sets the path patterns to match against.
        /// </summary>
        public List<string> PathPatterns { get; set; } = new();

        /// <summary>
        /// Gets or sets the priority of this rule (higher values = higher priority).
        /// </summary>
        public int Priority { get; set; } = 0;

        /// <summary>
        /// Gets or sets whether this rule is enabled.
        /// </summary>
        public bool Enabled { get; set; } = true;

        /// <summary>
        /// Gets or sets the description of this rule.
        /// </summary>
        public string Description { get; set; } = "";
    }

    /// <summary>
    /// Represents a registry monitoring rule configuration.
    /// </summary>
    public class RegistryMonitorRule
    {
        /// <summary>
        /// Gets or sets the registry hive root.
        /// </summary>
        public RegistryHive HiveRoot { get; set; }

        /// <summary>
        /// Gets or sets the registry key path to monitor.
        /// </summary>
        public string KeyPath { get; set; } = "";

        /// <summary>
        /// Gets or sets the description of this rule.
        /// </summary>
        public string Description { get; set; } = "";

        /// <summary>
        /// Gets or sets the security relevance level.
        /// </summary>
        public string SecurityRelevance { get; set; } = "Medium"; // Critical, High, Medium, Low

        /// <summary>
        /// Gets or sets whether to monitor sub-keys.
        /// </summary>
        public bool MonitorSubKeys { get; set; } = false;

        /// <summary>
        /// Gets or sets whether this rule is enabled.
        /// </summary>
        public bool Enabled { get; set; } = true;
    }

    /// <summary>
    /// Represents a registry change detection result.
    /// </summary>
    public class RegistryChange
    {
        /// <summary>
        /// Gets or sets the type of change (Added, Removed, Modified).
        /// </summary>
        public string ChangeType { get; set; } = "";

        /// <summary>
        /// Gets or sets the registry key path.
        /// </summary>
        public string KeyPath { get; set; } = "";

        /// <summary>
        /// Gets or sets the value name that changed.
        /// </summary>
        public string ValueName { get; set; } = "";

        /// <summary>
        /// Gets or sets the old value.
        /// </summary>
        public object? OldValue { get; set; }

        /// <summary>
        /// Gets or sets the new value.
        /// </summary>
        public object? NewValue { get; set; }

        /// <summary>
        /// Gets or sets the monitoring rule that detected this change.
        /// </summary>
        public RegistryMonitorRule Rule { get; set; } = new();
    }

    /// <summary>
    /// Represents an event log filter configuration.
    /// </summary>
    public class EventLogFilter
    {
        /// <summary>
        /// Gets or sets the Windows Event ID to filter.
        /// </summary>
        public int EventId { get; set; }

        /// <summary>
        /// Gets or sets the description of this event.
        /// </summary>
        public string Description { get; set; } = "";

        /// <summary>
        /// Gets or sets the security relevance level.
        /// </summary>
        public string SecurityRelevance { get; set; } = "Medium"; // Critical, High, Medium, Low

        /// <summary>
        /// Gets or sets the event category.
        /// </summary>
        public string Category { get; set; } = "";

        /// <summary>
        /// Gets or sets whether this filter is enabled.
        /// </summary>
        public bool Enabled { get; set; } = true;
    }
} 
