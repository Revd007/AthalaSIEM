using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace backend.Migrations
{
    /// <inheritdoc />
    public partial class AddFIMTables : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "FIMConfigurations",
                columns: table => new
                {
                    Id = table.Column<string>(type: "text", nullable: false),
                    Name = table.Column<string>(type: "character varying(255)", maxLength: 255, nullable: false),
                    Description = table.Column<string>(type: "character varying(1000)", maxLength: 1000, nullable: false),
                    Enabled = table.Column<bool>(type: "boolean", nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    UpdatedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    CreatedBy = table.Column<string>(type: "character varying(255)", maxLength: 255, nullable: false),
                    RulesJson = table.Column<string>(type: "TEXT", nullable: false),
                    GlobalSettingsJson = table.Column<string>(type: "TEXT", nullable: false),
                    TargetAgentsJson = table.Column<string>(type: "TEXT", nullable: false),
                    SupportedOSJson = table.Column<string>(type: "TEXT", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_FIMConfigurations", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "FIMEvents",
                columns: table => new
                {
                    Id = table.Column<string>(type: "text", nullable: false),
                    RuleId = table.Column<string>(type: "character varying(255)", maxLength: 255, nullable: false),
                    RuleName = table.Column<string>(type: "character varying(255)", maxLength: 255, nullable: false),
                    AgentId = table.Column<string>(type: "character varying(255)", maxLength: 255, nullable: false),
                    Timestamp = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    FilePath = table.Column<string>(type: "character varying(1000)", maxLength: 1000, nullable: false),
                    EventType = table.Column<string>(type: "character varying(100)", maxLength: 100, nullable: false),
                    OldFilePath = table.Column<string>(type: "character varying(1000)", maxLength: 1000, nullable: false),
                    OldFileInfoJson = table.Column<string>(type: "TEXT", nullable: false),
                    NewFileInfoJson = table.Column<string>(type: "TEXT", nullable: false),
                    User = table.Column<string>(type: "character varying(255)", maxLength: 255, nullable: false),
                    Process = table.Column<string>(type: "character varying(255)", maxLength: 255, nullable: false),
                    ProcessId = table.Column<int>(type: "integer", nullable: true),
                    SecurityLevel = table.Column<string>(type: "character varying(50)", maxLength: 50, nullable: false),
                    MetadataJson = table.Column<string>(type: "TEXT", nullable: false),
                    AlertGenerated = table.Column<bool>(type: "boolean", nullable: false),
                    TagsJson = table.Column<string>(type: "TEXT", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_FIMEvents", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "FIMTemplates",
                columns: table => new
                {
                    Id = table.Column<string>(type: "text", nullable: false),
                    Name = table.Column<string>(type: "character varying(255)", maxLength: 255, nullable: false),
                    Description = table.Column<string>(type: "character varying(1000)", maxLength: 1000, nullable: false),
                    Category = table.Column<string>(type: "character varying(100)", maxLength: 100, nullable: false),
                    TemplateRulesJson = table.Column<string>(type: "TEXT", nullable: false),
                    SupportedOSJson = table.Column<string>(type: "TEXT", nullable: false),
                    VariablesJson = table.Column<string>(type: "TEXT", nullable: false),
                    IsBuiltIn = table.Column<bool>(type: "boolean", nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    UpdatedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    CreatedBy = table.Column<string>(type: "character varying(255)", maxLength: 255, nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_FIMTemplates", x => x.Id);
                });
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "FIMConfigurations");

            migrationBuilder.DropTable(
                name: "FIMEvents");

            migrationBuilder.DropTable(
                name: "FIMTemplates");
        }
    }
}
