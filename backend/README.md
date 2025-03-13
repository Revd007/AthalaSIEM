# ATHALA SIEM Backend

This document outlines the clean code principles applied to the ATHALA SIEM Backend project.

## Clean Code Principles Applied

### Separation of Concerns
- **Controllers**: Handle HTTP requests and responses, input validation, and route management.
- **Services**: Contain business logic and orchestrate operations between different components.
- **Data Access**: Managed through Entity Framework Core with a clean repository pattern.
- **DTOs**: Used for data transfer between layers, preventing domain model exposure.
- **Models**: Represent the domain entities with proper validation and relationships.

### Dependency Injection
- Constructor injection used throughout the application for loose coupling.
- Services registered in a centralized location using extension methods.
- Interfaces defined for all services to enable testability and flexibility.

### SOLID Principles
- **Single Responsibility**: Each class has one reason to change.
- **Open/Closed**: Classes are open for extension but closed for modification.
- **Liskov Substitution**: Derived classes can substitute base classes without affecting functionality.
- **Interface Segregation**: Specific interfaces are better than general-purpose ones.
- **Dependency Inversion**: High-level modules depend on abstractions, not concrete implementations.

### Error Handling
- Consistent error responses across all controllers.
- Proper exception handling with meaningful error messages.
- Logging of exceptions with appropriate context information.
- Use of status codes that accurately reflect the nature of errors.

### Async/Await Pattern
- Asynchronous programming used throughout for scalability.
- Proper use of Task-based asynchronous pattern.
- Cancellation tokens passed where appropriate.

### Security Best Practices
- Input validation on all endpoints.
- JWT authentication with proper token validation.
- Role-based authorization for protected endpoints.
- API keys for agent authentication.
- HTTPS enforcement.

### Logging
- Structured logging with appropriate context.
- Different log levels (Debug, Information, Warning, Error) used appropriately.
- Sensitive information not logged.

### Code Organization
- Consistent naming conventions.
- Clear folder structure.
- XML documentation on public APIs.
- Clean separation between different components.

## Project Structure

- **Controllers/**: API endpoints for agents, logs, and authentication.
- **Services/**: Business logic implementation.
- **Data/**: Database context and configurations.
- **Models/**: Domain entities.
- **DTOs/**: Data transfer objects.
- **Migrations/**: Database migrations.
- **Program.cs**: Application entry point and configuration.

## API Endpoints

### Agents
- `POST /api/agents/register`: Register a new agent.
- `GET /api/agents`: Get all agents (requires Admin role).
- `PUT /api/agents/{agentId}/configure`: Configure an agent (requires Admin role).
- `POST /api/agents/{agentId}/heartbeat`: Receive a heartbeat from an agent.

### Logs
- `POST /api/logs/ingest`: Ingest a single log entry.
- `POST /api/logs/batch/{agentId}`: Receive a batch of logs from an agent.
- `GET /api/logs/agent/{agentId}`: Get logs for a specific agent (requires Analyst role).

## Authentication and Authorization

The application uses JWT (JSON Web Tokens) for authentication and role-based authorization:
- Admin role: Full access to all endpoints.
- Analyst role: Access to read-only endpoints.

## Getting Started

1. **Requirements**:
   - .NET 7.0 or later
   - SQL Server or PostgreSQL

2. **Update Connection String**:
   - Open `appsettings.json`
   - Update the `DefaultConnection` string with your database details

3. **Run Migrations**:
   ```
   dotnet ef database update
   ```

4. **Start the Application**:
   ```
   dotnet run
   ```

5. **Access Swagger UI**:
   - Navigate to `https://localhost:5315/swagger` in your browser 