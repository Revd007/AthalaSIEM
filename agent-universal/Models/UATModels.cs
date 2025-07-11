using System;
using System.Collections.Generic;

namespace AthalaSIEM.UniversalAgent.Models
{
    /// <summary>
    /// Configuration for User Acceptance Testing (UAT).
    /// </summary>
    public class UATConfiguration
    {
        /// <summary>
        /// Gets or sets whether the agent is in test mode.
        /// </summary>
        public bool TestMode { get; set; } = true;

        /// <summary>
        /// Gets or sets the duration of the test (ISO 8601 format).
        /// </summary>
        public string TestDuration { get; set; } = "PT30M";

        /// <summary>
        /// Gets or sets the list of test scenarios to run.
        /// </summary>
        public List<string> TestScenarios { get; set; } = new();

        /// <summary>
        /// Gets or sets the paths for test data.
        /// </summary>
        public List<string> TestDataPaths { get; set; } = new();

        /// <summary>
        /// Gets or sets whether to clean up after testing.
        /// </summary>
        public bool CleanupAfterTest { get; set; } = true;

        /// <summary>
        /// Gets or sets whether to generate a test report.
        /// </summary>
        public bool GenerateTestReport { get; set; } = true;

        /// <summary>
        /// Gets or sets the path for test reports.
        /// </summary>
        public string TestReportPath { get; set; } = ".\\UAT-Reports";
    }

    /// <summary>
    /// Represents the result of a single UAT test.
    /// </summary>
    public class UATTestResult
    {
        /// <summary>
        /// Gets or sets the name of the test.
        /// </summary>
        public string TestName { get; set; } = "";

        /// <summary>
        /// Gets or sets when the test started.
        /// </summary>
        public DateTime StartTime { get; set; }

        /// <summary>
        /// Gets or sets when the test ended.
        /// </summary>
        public DateTime EndTime { get; set; }

        /// <summary>
        /// Gets or sets the duration of the test.
        /// </summary>
        public TimeSpan Duration { get; set; }

        /// <summary>
        /// Gets or sets whether the test passed.
        /// </summary>
        public bool Passed { get; set; }

        /// <summary>
        /// Gets or sets the list of test steps.
        /// </summary>
        public List<string> Steps { get; set; } = new();

        /// <summary>
        /// Gets or sets the list of warnings encountered.
        /// </summary>
        public List<string> Warnings { get; set; } = new();

        /// <summary>
        /// Gets or sets the list of errors encountered.
        /// </summary>
        public List<string> Errors { get; set; } = new();

        /// <summary>
        /// Adds a test step with timestamp.
        /// </summary>
        /// <param name="step">The step description.</param>
        public void AddStep(string step) => Steps.Add($"{DateTime.UtcNow:HH:mm:ss} - {step}");

        /// <summary>
        /// Adds a warning to the test result.
        /// </summary>
        /// <param name="warning">The warning message.</param>
        public void AddWarning(string warning) => Warnings.Add(warning);

        /// <summary>
        /// Adds an error to the test result.
        /// </summary>
        /// <param name="error">The error message.</param>
        public void AddError(string error) => Errors.Add(error);
    }

    /// <summary>
    /// Represents the overall result of all UAT tests.
    /// </summary>
    public class UATOverallResult
    {
        /// <summary>
        /// Gets or sets when the test suite started.
        /// </summary>
        public DateTime StartTime { get; set; }

        /// <summary>
        /// Gets or sets when the test suite ended.
        /// </summary>
        public DateTime EndTime { get; set; }

        /// <summary>
        /// Gets or sets the total duration of all tests.
        /// </summary>
        public TimeSpan TotalDuration { get; set; }

        /// <summary>
        /// Gets or sets the individual test results.
        /// </summary>
        public List<UATTestResult> TestResults { get; set; } = new();

        /// <summary>
        /// Gets or sets the total number of tests.
        /// </summary>
        public int TotalTests { get; set; }

        /// <summary>
        /// Gets or sets the number of passed tests.
        /// </summary>
        public int PassedTests { get; set; }

        /// <summary>
        /// Gets or sets the number of failed tests.
        /// </summary>
        public int FailedTests { get; set; }

        /// <summary>
        /// Gets or sets the overall status of the test suite.
        /// </summary>
        public string OverallStatus { get; set; } = "";
    }
} 