using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace backend.Migrations
{
    /// <inheritdoc />
    public partial class EnsureAllTablesUpToDate : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "normalized_logs",
                columns: table => new
                {
                    Id = table.Column<string>(type: "text", nullable: false),
                    LogEntryId = table.Column<string>(type: "text", nullable: false),
                    Timestamp = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    AgentId = table.Column<string>(type: "text", nullable: true),
                    AgentName = table.Column<string>(type: "text", nullable: true),
                    HostName = table.Column<string>(type: "text", nullable: true),
                    HostIp = table.Column<string>(type: "text", nullable: true),
                    UserName = table.Column<string>(type: "text", nullable: true),
                    UserId = table.Column<string>(type: "text", nullable: true),
                    UserDomain = table.Column<string>(type: "text", nullable: true),
                    ProcessName = table.Column<string>(type: "text", nullable: true),
                    ProcessId = table.Column<int>(type: "integer", nullable: true),
                    ProcessPath = table.Column<string>(type: "text", nullable: true),
                    ProcessCommandLine = table.Column<string>(type: "text", nullable: true),
                    ProcessHash = table.Column<string>(type: "text", nullable: true),
                    ParentProcessName = table.Column<string>(type: "text", nullable: true),
                    ParentProcessId = table.Column<int>(type: "integer", nullable: true),
                    SourceIp = table.Column<string>(type: "text", nullable: true),
                    SourcePort = table.Column<int>(type: "integer", nullable: true),
                    DestinationIp = table.Column<string>(type: "text", nullable: true),
                    DestinationPort = table.Column<int>(type: "integer", nullable: true),
                    Protocol = table.Column<string>(type: "text", nullable: true),
                    EventAction = table.Column<string>(type: "text", nullable: true),
                    EventCategory = table.Column<string>(type: "text", nullable: true),
                    EventType = table.Column<string>(type: "text", nullable: true),
                    EventOutcome = table.Column<string>(type: "text", nullable: true),
                    EventCode = table.Column<string>(type: "text", nullable: true),
                    FilePath = table.Column<string>(type: "text", nullable: true),
                    FileName = table.Column<string>(type: "text", nullable: true),
                    FileHash = table.Column<string>(type: "text", nullable: true),
                    FileSize = table.Column<long>(type: "bigint", nullable: true),
                    SiemRuleId = table.Column<string>(type: "text", nullable: true),
                    SiemTechniqueId = table.Column<string>(type: "text", nullable: true),
                    SiemConfidence = table.Column<double>(type: "double precision", nullable: true),
                    SiemSeverity = table.Column<int>(type: "integer", nullable: true),
                    SiemCorrelationId = table.Column<string>(type: "text", nullable: true),
                    metadata_json = table.Column<string>(type: "text", nullable: true),
                    CreatedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_normalized_logs", x => x.Id);
                });
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "normalized_logs");
        }
    }
}
