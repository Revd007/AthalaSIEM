using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace backend.Migrations
{
    /// <inheritdoc />
    public partial class FixPendingModelChanges : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "user_security_settings",
                columns: table => new
                {
                    UserId = table.Column<string>(type: "text", nullable: false),
                    MaxConcurrentSessions = table.Column<int>(type: "integer", nullable: false),
                    SessionTimeoutMinutes = table.Column<int>(type: "integer", nullable: false),
                    RequireReauthForSensitive = table.Column<bool>(type: "boolean", nullable: false),
                    RestrictLoginByIP = table.Column<bool>(type: "boolean", nullable: false),
                    AllowedIPAddresses = table.Column<string>(type: "TEXT", nullable: true),
                    RestrictLoginByTime = table.Column<bool>(type: "boolean", nullable: false),
                    AllowedTimeWindows = table.Column<string>(type: "TEXT", nullable: true),
                    MaxFailedLoginAttempts = table.Column<int>(type: "integer", nullable: false),
                    LockoutDurationMinutes = table.Column<int>(type: "integer", nullable: false),
                    EnablePasswordExpiration = table.Column<bool>(type: "boolean", nullable: false),
                    PasswordExpirationDays = table.Column<int>(type: "integer", nullable: false),
                    PreventPasswordReuse = table.Column<bool>(type: "boolean", nullable: false),
                    PasswordHistoryCount = table.Column<int>(type: "integer", nullable: false),
                    RequireStrongPassword = table.Column<bool>(type: "boolean", nullable: false),
                    MinPasswordLength = table.Column<int>(type: "integer", nullable: false),
                    RequireUppercase = table.Column<bool>(type: "boolean", nullable: false),
                    RequireLowercase = table.Column<bool>(type: "boolean", nullable: false),
                    RequireDigit = table.Column<bool>(type: "boolean", nullable: false),
                    RequireSpecialChar = table.Column<bool>(type: "boolean", nullable: false),
                    LogAllLoginAttempts = table.Column<bool>(type: "boolean", nullable: false),
                    EmailSecurityNotifications = table.Column<bool>(type: "boolean", nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    UpdatedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_user_security_settings", x => x.UserId);
                    table.ForeignKey(
                        name: "FK_user_security_settings_users_UserId",
                        column: x => x.UserId,
                        principalTable: "users",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "user_security_settings");
        }
    }
}
