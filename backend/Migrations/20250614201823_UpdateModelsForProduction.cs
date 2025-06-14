using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace backend.Migrations
{
    /// <inheritdoc />
    public partial class UpdateModelsForProduction : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropPrimaryKey(
                name: "PK_AgentDeploymentTokens",
                table: "AgentDeploymentTokens");

            migrationBuilder.RenameColumn(
                name: "Tags",
                table: "alert_rules",
                newName: "Configuration");

            migrationBuilder.AddColumn<string>(
                name: "Details",
                table: "log_entries",
                type: "text",
                nullable: true);

            migrationBuilder.AlterColumn<string>(
                name: "Description",
                table: "alert_rules",
                type: "character varying(1000)",
                maxLength: 1000,
                nullable: true,
                oldClrType: typeof(string),
                oldType: "character varying(500)",
                oldMaxLength: 500,
                oldNullable: true);

            migrationBuilder.AlterColumn<string>(
                name: "CreatedBy",
                table: "alert_rules",
                type: "character varying(100)",
                maxLength: 100,
                nullable: true,
                oldClrType: typeof(string),
                oldType: "text");

            migrationBuilder.AddColumn<string>(
                name: "Actions",
                table: "alert_rules",
                type: "text",
                nullable: true);

            migrationBuilder.AddColumn<long>(
                name: "AlertsGenerated",
                table: "alert_rules",
                type: "bigint",
                nullable: false,
                defaultValue: 0L);

            migrationBuilder.AddColumn<string>(
                name: "CollectorType",
                table: "alert_rules",
                type: "character varying(50)",
                maxLength: 50,
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<int>(
                name: "EvaluationFrequencyMinutes",
                table: "alert_rules",
                type: "integer",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<long>(
                name: "ExecutionCount",
                table: "alert_rules",
                type: "bigint",
                nullable: false,
                defaultValue: 0L);

            migrationBuilder.AddColumn<DateTime>(
                name: "LastExecuted",
                table: "alert_rules",
                type: "timestamp with time zone",
                nullable: true);

            migrationBuilder.AddColumn<int>(
                name: "Priority",
                table: "alert_rules",
                type: "integer",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<int>(
                name: "ThresholdCount",
                table: "alert_rules",
                type: "integer",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<int>(
                name: "TimeWindowMinutes",
                table: "alert_rules",
                type: "integer",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<string>(
                name: "UpdatedBy",
                table: "alert_rules",
                type: "character varying(100)",
                maxLength: 100,
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "AgentVersion",
                table: "agents",
                type: "character varying(50)",
                maxLength: 50,
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "DeploymentTokenId",
                table: "agents",
                type: "character varying(50)",
                maxLength: 50,
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "OsVersion",
                table: "agents",
                type: "character varying(100)",
                maxLength: 100,
                nullable: true);

            migrationBuilder.AlterColumn<DateTime>(
                name: "ExpiresAt",
                table: "AgentDeploymentTokens",
                type: "timestamp with time zone",
                nullable: true,
                oldClrType: typeof(DateTime),
                oldType: "timestamp with time zone");

            migrationBuilder.AddColumn<string>(
                name: "Id",
                table: "AgentDeploymentTokens",
                type: "text",
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<string>(
                name: "Configuration",
                table: "AgentDeploymentTokens",
                type: "text",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "CreatedBy",
                table: "AgentDeploymentTokens",
                type: "text",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "Description",
                table: "AgentDeploymentTokens",
                type: "text",
                nullable: true);

            migrationBuilder.AddColumn<bool>(
                name: "IsActive",
                table: "AgentDeploymentTokens",
                type: "boolean",
                nullable: false,
                defaultValue: false);

            migrationBuilder.AddColumn<DateTime>(
                name: "LastUsed",
                table: "AgentDeploymentTokens",
                type: "timestamp with time zone",
                nullable: true);

            migrationBuilder.AddColumn<int>(
                name: "MaxUsage",
                table: "AgentDeploymentTokens",
                type: "integer",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "Name",
                table: "AgentDeploymentTokens",
                type: "text",
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<string>(
                name: "PlatformType",
                table: "AgentDeploymentTokens",
                type: "text",
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<int>(
                name: "UsageCount",
                table: "AgentDeploymentTokens",
                type: "integer",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<string>(
                name: "Configuration",
                table: "AgentConfigs",
                type: "text",
                nullable: true);

            migrationBuilder.AddColumn<DateTime>(
                name: "LastUpdated",
                table: "AgentConfigs",
                type: "timestamp with time zone",
                nullable: false,
                defaultValue: new DateTime(1, 1, 1, 0, 0, 0, 0, DateTimeKind.Unspecified));

            migrationBuilder.AddColumn<bool>(
                name: "RequiresRestart",
                table: "AgentConfigs",
                type: "boolean",
                nullable: false,
                defaultValue: false);

            migrationBuilder.AddPrimaryKey(
                name: "PK_AgentDeploymentTokens",
                table: "AgentDeploymentTokens",
                column: "Id");

            migrationBuilder.CreateTable(
                name: "AlertCorrelations",
                columns: table => new
                {
                    Id = table.Column<string>(type: "text", nullable: false),
                    Pattern = table.Column<string>(type: "character varying(200)", maxLength: 200, nullable: false),
                    CollectorTypes = table.Column<string>(type: "text", nullable: false),
                    Severity = table.Column<int>(type: "integer", nullable: false),
                    Occurrences = table.Column<int>(type: "integer", nullable: false),
                    TimeWindowMinutes = table.Column<int>(type: "integer", nullable: false),
                    FirstOccurrence = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    LastOccurrence = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    AffectedAgents = table.Column<int>(type: "integer", nullable: false),
                    RecommendedActions = table.Column<string>(type: "text", nullable: true),
                    AnalysisData = table.Column<string>(type: "text", nullable: true),
                    IsActive = table.Column<bool>(type: "boolean", nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    UpdatedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_AlertCorrelations", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "AlertMetadata",
                columns: table => new
                {
                    Id = table.Column<string>(type: "text", nullable: false),
                    AlertId = table.Column<string>(type: "text", nullable: false),
                    CollectorType = table.Column<string>(type: "character varying(50)", maxLength: 50, nullable: false),
                    ThreatLevel = table.Column<int>(type: "integer", nullable: false),
                    OriginalLogId = table.Column<string>(type: "character varying(100)", maxLength: 100, nullable: true),
                    ThreatIndicators = table.Column<string>(type: "text", nullable: true),
                    CollectorSpecificData = table.Column<string>(type: "text", nullable: true),
                    AutoEscalationEnabled = table.Column<bool>(type: "boolean", nullable: false),
                    EscalationThresholds = table.Column<string>(type: "text", nullable: true),
                    NotificationChannels = table.Column<string>(type: "text", nullable: true),
                    ThreatScore = table.Column<double>(type: "double precision", nullable: false),
                    ContextData = table.Column<string>(type: "text", nullable: true),
                    CorrelationId = table.Column<string>(type: "character varying(100)", maxLength: 100, nullable: true),
                    CreatedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    UpdatedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_AlertMetadata", x => x.Id);
                    table.ForeignKey(
                        name: "FK_AlertMetadata_alerts_AlertId",
                        column: x => x.AlertId,
                        principalTable: "alerts",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "attack_techniques",
                columns: table => new
                {
                    Id = table.Column<string>(type: "text", nullable: false),
                    TechniqueId = table.Column<string>(type: "character varying(20)", maxLength: 20, nullable: false),
                    Name = table.Column<string>(type: "character varying(200)", maxLength: 200, nullable: false),
                    Description = table.Column<string>(type: "character varying(1000)", maxLength: 1000, nullable: true),
                    Tactic = table.Column<string>(type: "character varying(50)", maxLength: 50, nullable: false),
                    Platform = table.Column<string>(type: "character varying(50)", maxLength: 50, nullable: false),
                    DataSources = table.Column<string>(type: "text", nullable: true),
                    Mitigations = table.Column<string>(type: "text", nullable: true),
                    DetectionCount = table.Column<int>(type: "integer", nullable: false),
                    LastDetected = table.Column<DateTime>(type: "timestamp with time zone", nullable: true),
                    IsActive = table.Column<bool>(type: "boolean", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_attack_techniques", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "CollectorConfigurations",
                columns: table => new
                {
                    Id = table.Column<string>(type: "text", nullable: false),
                    AgentId = table.Column<string>(type: "text", nullable: false),
                    CollectorType = table.Column<string>(type: "character varying(50)", maxLength: 50, nullable: false),
                    Enabled = table.Column<bool>(type: "boolean", nullable: false),
                    Configuration = table.Column<string>(type: "text", nullable: false),
                    CollectionIntervalSeconds = table.Column<int>(type: "integer", nullable: false),
                    MaxEventsPerBatch = table.Column<int>(type: "integer", nullable: false),
                    BufferSize = table.Column<int>(type: "integer", nullable: false),
                    EnableThreatIntelligence = table.Column<bool>(type: "boolean", nullable: false),
                    EnableRealTimeMonitoring = table.Column<bool>(type: "boolean", nullable: false),
                    LogLevelFilter = table.Column<string>(type: "character varying(100)", maxLength: 100, nullable: false),
                    CustomFilters = table.Column<string>(type: "text", nullable: true),
                    CreatedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    UpdatedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_CollectorConfigurations", x => x.Id);
                    table.ForeignKey(
                        name: "FK_CollectorConfigurations_agents_AgentId",
                        column: x => x.AgentId,
                        principalTable: "agents",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "file_integrity_baselines",
                columns: table => new
                {
                    Id = table.Column<string>(type: "text", nullable: false),
                    AgentId = table.Column<string>(type: "character varying(50)", maxLength: 50, nullable: false),
                    FilePath = table.Column<string>(type: "character varying(500)", maxLength: 500, nullable: false),
                    FileHash = table.Column<string>(type: "character varying(128)", maxLength: 128, nullable: false),
                    FileSize = table.Column<long>(type: "bigint", nullable: false),
                    LastModified = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    CreatedTime = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    FileAttributes = table.Column<string>(type: "character varying(100)", maxLength: 100, nullable: true),
                    BaselineCreatedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    BaselineUpdatedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    IsActive = table.Column<bool>(type: "boolean", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_file_integrity_baselines", x => x.Id);
                    table.ForeignKey(
                        name: "FK_file_integrity_baselines_agents_AgentId",
                        column: x => x.AgentId,
                        principalTable: "agents",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "file_integrity_events",
                columns: table => new
                {
                    Id = table.Column<string>(type: "text", nullable: false),
                    AgentId = table.Column<string>(type: "character varying(50)", maxLength: 50, nullable: false),
                    FilePath = table.Column<string>(type: "character varying(500)", maxLength: 500, nullable: false),
                    ChangeType = table.Column<string>(type: "character varying(20)", maxLength: 20, nullable: false),
                    BaselineHash = table.Column<string>(type: "character varying(128)", maxLength: 128, nullable: true),
                    CurrentHash = table.Column<string>(type: "character varying(128)", maxLength: 128, nullable: true),
                    BaselineSize = table.Column<long>(type: "bigint", nullable: true),
                    CurrentSize = table.Column<long>(type: "bigint", nullable: true),
                    BaselineModified = table.Column<DateTime>(type: "timestamp with time zone", nullable: true),
                    CurrentModified = table.Column<DateTime>(type: "timestamp with time zone", nullable: true),
                    FileAttributes = table.Column<string>(type: "character varying(100)", maxLength: 100, nullable: true),
                    Severity = table.Column<string>(type: "character varying(20)", maxLength: 20, nullable: false),
                    DetectedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    ProcessedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    IsAcknowledged = table.Column<bool>(type: "boolean", nullable: false),
                    AcknowledgedBy = table.Column<string>(type: "character varying(50)", maxLength: 50, nullable: true),
                    AcknowledgedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: true),
                    Details = table.Column<string>(type: "text", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_file_integrity_events", x => x.Id);
                    table.ForeignKey(
                        name: "FK_file_integrity_events_agents_AgentId",
                        column: x => x.AgentId,
                        principalTable: "agents",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "file_integrity_rules",
                columns: table => new
                {
                    Id = table.Column<string>(type: "text", nullable: false),
                    Name = table.Column<string>(type: "character varying(100)", maxLength: 100, nullable: false),
                    Description = table.Column<string>(type: "character varying(500)", maxLength: 500, nullable: true),
                    IsEnabled = table.Column<bool>(type: "boolean", nullable: false),
                    MonitoredPaths = table.Column<string>(type: "text", nullable: false),
                    ExcludePatterns = table.Column<string>(type: "text", nullable: true),
                    RealTimeMonitoring = table.Column<bool>(type: "boolean", nullable: false),
                    ScanIntervalMinutes = table.Column<int>(type: "integer", nullable: false),
                    Severity = table.Column<string>(type: "character varying(20)", maxLength: 20, nullable: false),
                    AlertOnCreation = table.Column<bool>(type: "boolean", nullable: false),
                    AlertOnModification = table.Column<bool>(type: "boolean", nullable: false),
                    AlertOnDeletion = table.Column<bool>(type: "boolean", nullable: false),
                    AlertOnRename = table.Column<bool>(type: "boolean", nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    UpdatedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    CreatedBy = table.Column<string>(type: "character varying(50)", maxLength: 50, nullable: true),
                    TargetAgents = table.Column<string>(type: "text", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_file_integrity_rules", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "threat_campaigns",
                columns: table => new
                {
                    Id = table.Column<string>(type: "text", nullable: false),
                    Name = table.Column<string>(type: "character varying(200)", maxLength: 200, nullable: false),
                    Description = table.Column<string>(type: "character varying(1000)", maxLength: 1000, nullable: true),
                    Actor = table.Column<string>(type: "character varying(100)", maxLength: 100, nullable: true),
                    Severity = table.Column<string>(type: "character varying(50)", maxLength: 50, nullable: false),
                    FirstDetected = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    LastDetected = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    IsActive = table.Column<bool>(type: "boolean", nullable: false),
                    TechniquesUsed = table.Column<string>(type: "text", nullable: true),
                    TargetedSectors = table.Column<string>(type: "text", nullable: true),
                    Geography = table.Column<string>(type: "text", nullable: true),
                    IndicatorCount = table.Column<int>(type: "integer", nullable: false),
                    MatchCount = table.Column<int>(type: "integer", nullable: false),
                    CreatedBy = table.Column<string>(type: "character varying(50)", maxLength: 50, nullable: true),
                    CreatedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    Metadata = table.Column<string>(type: "text", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_threat_campaigns", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "threat_enrichments",
                columns: table => new
                {
                    Id = table.Column<string>(type: "text", nullable: false),
                    IndicatorValue = table.Column<string>(type: "character varying(500)", maxLength: 500, nullable: false),
                    IndicatorType = table.Column<string>(type: "character varying(50)", maxLength: 50, nullable: false),
                    EnrichmentSource = table.Column<string>(type: "character varying(100)", maxLength: 100, nullable: false),
                    EnrichmentData = table.Column<string>(type: "text", nullable: false),
                    EnrichedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    ExpiresAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: true),
                    Status = table.Column<string>(type: "character varying(20)", maxLength: 20, nullable: false),
                    ErrorMessage = table.Column<string>(type: "text", nullable: true),
                    IsCached = table.Column<bool>(type: "boolean", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_threat_enrichments", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "threat_intelligence_feeds",
                columns: table => new
                {
                    Id = table.Column<string>(type: "text", nullable: false),
                    Name = table.Column<string>(type: "character varying(100)", maxLength: 100, nullable: false),
                    Description = table.Column<string>(type: "character varying(500)", maxLength: 500, nullable: false),
                    FeedType = table.Column<string>(type: "character varying(50)", maxLength: 50, nullable: false),
                    FeedUrl = table.Column<string>(type: "text", nullable: false),
                    ApiKey = table.Column<string>(type: "character varying(100)", maxLength: 100, nullable: false),
                    Username = table.Column<string>(type: "character varying(100)", maxLength: 100, nullable: false),
                    Password = table.Column<string>(type: "character varying(100)", maxLength: 100, nullable: false),
                    UpdateIntervalMinutes = table.Column<int>(type: "integer", nullable: false),
                    IsActive = table.Column<bool>(type: "boolean", nullable: false),
                    LastUpdated = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    CreatedBy = table.Column<string>(type: "character varying(50)", maxLength: 50, nullable: true),
                    TotalIndicators = table.Column<int>(type: "integer", nullable: false),
                    LastError = table.Column<string>(type: "text", nullable: true),
                    Priority = table.Column<string>(type: "character varying(20)", maxLength: 20, nullable: false),
                    Source = table.Column<string>(type: "character varying(50)", maxLength: 50, nullable: false),
                    EnableEnrichment = table.Column<bool>(type: "boolean", nullable: false),
                    Configuration = table.Column<string>(type: "text", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_threat_intelligence_feeds", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "threat_whitelists",
                columns: table => new
                {
                    Id = table.Column<string>(type: "text", nullable: false),
                    Type = table.Column<string>(type: "character varying(50)", maxLength: 50, nullable: false),
                    Value = table.Column<string>(type: "character varying(500)", maxLength: 500, nullable: false),
                    Reason = table.Column<string>(type: "character varying(500)", maxLength: 500, nullable: true),
                    CreatedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    CreatedBy = table.Column<string>(type: "character varying(50)", maxLength: 50, nullable: true),
                    IsActive = table.Column<bool>(type: "boolean", nullable: false),
                    ExpiresAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_threat_whitelists", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "threat_indicators",
                columns: table => new
                {
                    Id = table.Column<string>(type: "text", nullable: false),
                    Type = table.Column<string>(type: "character varying(50)", maxLength: 50, nullable: false),
                    Value = table.Column<string>(type: "character varying(500)", maxLength: 500, nullable: false),
                    Confidence = table.Column<string>(type: "character varying(20)", maxLength: 20, nullable: false),
                    Severity = table.Column<string>(type: "character varying(20)", maxLength: 20, nullable: false),
                    ThreatType = table.Column<string>(type: "character varying(200)", maxLength: 200, nullable: true),
                    MalwareFamily = table.Column<string>(type: "character varying(100)", maxLength: 100, nullable: true),
                    Description = table.Column<string>(type: "character varying(500)", maxLength: 500, nullable: true),
                    Tags = table.Column<string>(type: "text", nullable: true),
                    FirstSeen = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    LastSeen = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    ExpiresAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: true),
                    IsActive = table.Column<bool>(type: "boolean", nullable: false),
                    FeedId = table.Column<string>(type: "text", nullable: false),
                    Source = table.Column<string>(type: "character varying(100)", maxLength: 100, nullable: true),
                    Context = table.Column<string>(type: "text", nullable: true),
                    HitCount = table.Column<int>(type: "integer", nullable: false),
                    LastHit = table.Column<DateTime>(type: "timestamp with time zone", nullable: true),
                    LogEntryId = table.Column<string>(type: "text", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_threat_indicators", x => x.Id);
                    table.ForeignKey(
                        name: "FK_threat_indicators_threat_intelligence_feeds_FeedId",
                        column: x => x.FeedId,
                        principalTable: "threat_intelligence_feeds",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "threat_matches",
                columns: table => new
                {
                    Id = table.Column<string>(type: "text", nullable: false),
                    IndicatorId = table.Column<string>(type: "text", nullable: false),
                    LogEntryId = table.Column<string>(type: "text", nullable: false),
                    MatchedValue = table.Column<string>(type: "character varying(500)", maxLength: 500, nullable: false),
                    MatchedField = table.Column<string>(type: "character varying(100)", maxLength: 100, nullable: false),
                    Confidence = table.Column<string>(type: "character varying(20)", maxLength: 20, nullable: false),
                    Severity = table.Column<string>(type: "character varying(20)", maxLength: 20, nullable: false),
                    DetectedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    IsAcknowledged = table.Column<bool>(type: "boolean", nullable: false),
                    AcknowledgedBy = table.Column<string>(type: "character varying(50)", maxLength: 50, nullable: true),
                    AcknowledgedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: true),
                    Notes = table.Column<string>(type: "character varying(500)", maxLength: 500, nullable: true),
                    IsFalsePositive = table.Column<bool>(type: "boolean", nullable: false),
                    EnrichmentData = table.Column<string>(type: "text", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_threat_matches", x => x.Id);
                    table.ForeignKey(
                        name: "FK_threat_matches_log_entries_LogEntryId",
                        column: x => x.LogEntryId,
                        principalTable: "log_entries",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_threat_matches_threat_indicators_IndicatorId",
                        column: x => x.IndicatorId,
                        principalTable: "threat_indicators",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateIndex(
                name: "IX_AlertMetadata_AlertId",
                table: "AlertMetadata",
                column: "AlertId");

            migrationBuilder.CreateIndex(
                name: "IX_attack_techniques_IsActive",
                table: "attack_techniques",
                column: "IsActive");

            migrationBuilder.CreateIndex(
                name: "IX_attack_techniques_Platform",
                table: "attack_techniques",
                column: "Platform");

            migrationBuilder.CreateIndex(
                name: "IX_attack_techniques_Tactic",
                table: "attack_techniques",
                column: "Tactic");

            migrationBuilder.CreateIndex(
                name: "IX_attack_techniques_TechniqueId",
                table: "attack_techniques",
                column: "TechniqueId",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_CollectorConfigurations_AgentId",
                table: "CollectorConfigurations",
                column: "AgentId");

            migrationBuilder.CreateIndex(
                name: "IX_file_integrity_baselines_AgentId",
                table: "file_integrity_baselines",
                column: "AgentId");

            migrationBuilder.CreateIndex(
                name: "IX_file_integrity_baselines_AgentId_FilePath",
                table: "file_integrity_baselines",
                columns: new[] { "AgentId", "FilePath" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_file_integrity_baselines_BaselineCreatedAt",
                table: "file_integrity_baselines",
                column: "BaselineCreatedAt");

            migrationBuilder.CreateIndex(
                name: "IX_file_integrity_baselines_IsActive",
                table: "file_integrity_baselines",
                column: "IsActive");

            migrationBuilder.CreateIndex(
                name: "IX_file_integrity_events_AgentId",
                table: "file_integrity_events",
                column: "AgentId");

            migrationBuilder.CreateIndex(
                name: "IX_file_integrity_events_ChangeType",
                table: "file_integrity_events",
                column: "ChangeType");

            migrationBuilder.CreateIndex(
                name: "IX_file_integrity_events_DetectedAt",
                table: "file_integrity_events",
                column: "DetectedAt");

            migrationBuilder.CreateIndex(
                name: "IX_file_integrity_events_IsAcknowledged",
                table: "file_integrity_events",
                column: "IsAcknowledged");

            migrationBuilder.CreateIndex(
                name: "IX_file_integrity_events_Severity",
                table: "file_integrity_events",
                column: "Severity");

            migrationBuilder.CreateIndex(
                name: "IX_file_integrity_rules_CreatedAt",
                table: "file_integrity_rules",
                column: "CreatedAt");

            migrationBuilder.CreateIndex(
                name: "IX_file_integrity_rules_IsEnabled",
                table: "file_integrity_rules",
                column: "IsEnabled");

            migrationBuilder.CreateIndex(
                name: "IX_file_integrity_rules_Name",
                table: "file_integrity_rules",
                column: "Name",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_threat_campaigns_FirstDetected",
                table: "threat_campaigns",
                column: "FirstDetected");

            migrationBuilder.CreateIndex(
                name: "IX_threat_campaigns_IsActive",
                table: "threat_campaigns",
                column: "IsActive");

            migrationBuilder.CreateIndex(
                name: "IX_threat_campaigns_LastDetected",
                table: "threat_campaigns",
                column: "LastDetected");

            migrationBuilder.CreateIndex(
                name: "IX_threat_campaigns_Name",
                table: "threat_campaigns",
                column: "Name",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_threat_campaigns_Severity",
                table: "threat_campaigns",
                column: "Severity");

            migrationBuilder.CreateIndex(
                name: "IX_threat_enrichments_EnrichedAt",
                table: "threat_enrichments",
                column: "EnrichedAt");

            migrationBuilder.CreateIndex(
                name: "IX_threat_enrichments_IndicatorValue_IndicatorType_EnrichmentS~",
                table: "threat_enrichments",
                columns: new[] { "IndicatorValue", "IndicatorType", "EnrichmentSource" });

            migrationBuilder.CreateIndex(
                name: "IX_threat_enrichments_Status",
                table: "threat_enrichments",
                column: "Status");

            migrationBuilder.CreateIndex(
                name: "IX_threat_indicators_Confidence",
                table: "threat_indicators",
                column: "Confidence");

            migrationBuilder.CreateIndex(
                name: "IX_threat_indicators_FeedId",
                table: "threat_indicators",
                column: "FeedId");

            migrationBuilder.CreateIndex(
                name: "IX_threat_indicators_FirstSeen",
                table: "threat_indicators",
                column: "FirstSeen");

            migrationBuilder.CreateIndex(
                name: "IX_threat_indicators_IsActive",
                table: "threat_indicators",
                column: "IsActive");

            migrationBuilder.CreateIndex(
                name: "IX_threat_indicators_LastSeen",
                table: "threat_indicators",
                column: "LastSeen");

            migrationBuilder.CreateIndex(
                name: "IX_threat_indicators_Severity",
                table: "threat_indicators",
                column: "Severity");

            migrationBuilder.CreateIndex(
                name: "IX_threat_indicators_Type",
                table: "threat_indicators",
                column: "Type");

            migrationBuilder.CreateIndex(
                name: "IX_threat_indicators_Type_Value",
                table: "threat_indicators",
                columns: new[] { "Type", "Value" });

            migrationBuilder.CreateIndex(
                name: "IX_threat_intelligence_feeds_IsActive",
                table: "threat_intelligence_feeds",
                column: "IsActive");

            migrationBuilder.CreateIndex(
                name: "IX_threat_intelligence_feeds_LastUpdated",
                table: "threat_intelligence_feeds",
                column: "LastUpdated");

            migrationBuilder.CreateIndex(
                name: "IX_threat_intelligence_feeds_Name",
                table: "threat_intelligence_feeds",
                column: "Name",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_threat_intelligence_feeds_Priority",
                table: "threat_intelligence_feeds",
                column: "Priority");

            migrationBuilder.CreateIndex(
                name: "IX_threat_matches_DetectedAt",
                table: "threat_matches",
                column: "DetectedAt");

            migrationBuilder.CreateIndex(
                name: "IX_threat_matches_IndicatorId",
                table: "threat_matches",
                column: "IndicatorId");

            migrationBuilder.CreateIndex(
                name: "IX_threat_matches_IsAcknowledged",
                table: "threat_matches",
                column: "IsAcknowledged");

            migrationBuilder.CreateIndex(
                name: "IX_threat_matches_IsFalsePositive",
                table: "threat_matches",
                column: "IsFalsePositive");

            migrationBuilder.CreateIndex(
                name: "IX_threat_matches_LogEntryId",
                table: "threat_matches",
                column: "LogEntryId");

            migrationBuilder.CreateIndex(
                name: "IX_threat_matches_Severity",
                table: "threat_matches",
                column: "Severity");

            migrationBuilder.CreateIndex(
                name: "IX_threat_whitelists_CreatedAt",
                table: "threat_whitelists",
                column: "CreatedAt");

            migrationBuilder.CreateIndex(
                name: "IX_threat_whitelists_IsActive",
                table: "threat_whitelists",
                column: "IsActive");

            migrationBuilder.CreateIndex(
                name: "IX_threat_whitelists_Type_Value",
                table: "threat_whitelists",
                columns: new[] { "Type", "Value" },
                unique: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "AlertCorrelations");

            migrationBuilder.DropTable(
                name: "AlertMetadata");

            migrationBuilder.DropTable(
                name: "attack_techniques");

            migrationBuilder.DropTable(
                name: "CollectorConfigurations");

            migrationBuilder.DropTable(
                name: "file_integrity_baselines");

            migrationBuilder.DropTable(
                name: "file_integrity_events");

            migrationBuilder.DropTable(
                name: "file_integrity_rules");

            migrationBuilder.DropTable(
                name: "threat_campaigns");

            migrationBuilder.DropTable(
                name: "threat_enrichments");

            migrationBuilder.DropTable(
                name: "threat_matches");

            migrationBuilder.DropTable(
                name: "threat_whitelists");

            migrationBuilder.DropTable(
                name: "threat_indicators");

            migrationBuilder.DropTable(
                name: "threat_intelligence_feeds");

            migrationBuilder.DropPrimaryKey(
                name: "PK_AgentDeploymentTokens",
                table: "AgentDeploymentTokens");

            migrationBuilder.DropColumn(
                name: "Details",
                table: "log_entries");

            migrationBuilder.DropColumn(
                name: "Actions",
                table: "alert_rules");

            migrationBuilder.DropColumn(
                name: "AlertsGenerated",
                table: "alert_rules");

            migrationBuilder.DropColumn(
                name: "CollectorType",
                table: "alert_rules");

            migrationBuilder.DropColumn(
                name: "EvaluationFrequencyMinutes",
                table: "alert_rules");

            migrationBuilder.DropColumn(
                name: "ExecutionCount",
                table: "alert_rules");

            migrationBuilder.DropColumn(
                name: "LastExecuted",
                table: "alert_rules");

            migrationBuilder.DropColumn(
                name: "Priority",
                table: "alert_rules");

            migrationBuilder.DropColumn(
                name: "ThresholdCount",
                table: "alert_rules");

            migrationBuilder.DropColumn(
                name: "TimeWindowMinutes",
                table: "alert_rules");

            migrationBuilder.DropColumn(
                name: "UpdatedBy",
                table: "alert_rules");

            migrationBuilder.DropColumn(
                name: "AgentVersion",
                table: "agents");

            migrationBuilder.DropColumn(
                name: "DeploymentTokenId",
                table: "agents");

            migrationBuilder.DropColumn(
                name: "OsVersion",
                table: "agents");

            migrationBuilder.DropColumn(
                name: "Id",
                table: "AgentDeploymentTokens");

            migrationBuilder.DropColumn(
                name: "Configuration",
                table: "AgentDeploymentTokens");

            migrationBuilder.DropColumn(
                name: "CreatedBy",
                table: "AgentDeploymentTokens");

            migrationBuilder.DropColumn(
                name: "Description",
                table: "AgentDeploymentTokens");

            migrationBuilder.DropColumn(
                name: "IsActive",
                table: "AgentDeploymentTokens");

            migrationBuilder.DropColumn(
                name: "LastUsed",
                table: "AgentDeploymentTokens");

            migrationBuilder.DropColumn(
                name: "MaxUsage",
                table: "AgentDeploymentTokens");

            migrationBuilder.DropColumn(
                name: "Name",
                table: "AgentDeploymentTokens");

            migrationBuilder.DropColumn(
                name: "PlatformType",
                table: "AgentDeploymentTokens");

            migrationBuilder.DropColumn(
                name: "UsageCount",
                table: "AgentDeploymentTokens");

            migrationBuilder.DropColumn(
                name: "Configuration",
                table: "AgentConfigs");

            migrationBuilder.DropColumn(
                name: "LastUpdated",
                table: "AgentConfigs");

            migrationBuilder.DropColumn(
                name: "RequiresRestart",
                table: "AgentConfigs");

            migrationBuilder.RenameColumn(
                name: "Configuration",
                table: "alert_rules",
                newName: "Tags");

            migrationBuilder.AlterColumn<string>(
                name: "Description",
                table: "alert_rules",
                type: "character varying(500)",
                maxLength: 500,
                nullable: true,
                oldClrType: typeof(string),
                oldType: "character varying(1000)",
                oldMaxLength: 1000,
                oldNullable: true);

            migrationBuilder.AlterColumn<string>(
                name: "CreatedBy",
                table: "alert_rules",
                type: "text",
                nullable: false,
                defaultValue: "",
                oldClrType: typeof(string),
                oldType: "character varying(100)",
                oldMaxLength: 100,
                oldNullable: true);

            migrationBuilder.AlterColumn<DateTime>(
                name: "ExpiresAt",
                table: "AgentDeploymentTokens",
                type: "timestamp with time zone",
                nullable: false,
                defaultValue: new DateTime(1, 1, 1, 0, 0, 0, 0, DateTimeKind.Unspecified),
                oldClrType: typeof(DateTime),
                oldType: "timestamp with time zone",
                oldNullable: true);

            migrationBuilder.AddPrimaryKey(
                name: "PK_AgentDeploymentTokens",
                table: "AgentDeploymentTokens",
                column: "Token");
        }
    }
}
