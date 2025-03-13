# Athala SIEM - Security Information and Event Management System

## Overview
Athala SIEM is a comprehensive Security Information and Event Management system designed to collect, analyze, and visualize security events and logs from multiple sources. The system consists of three main components: a Backend API server, a Frontend web application, and Agent software that can be deployed on various systems to collect and forward logs.

![Athala SIEM Architecture](docs/images/architecture.png)

## System Architecture
Athala SIEM follows a distributed architecture:
- **Backend**: .NET Core API server that processes and stores security events
- **Frontend**: Next.js-based web application providing a user interface for monitoring and analysis  
- **Agent**: .NET Core application that collects logs from systems and forwards them to the Backend

## Components

### Backend (API Server)
The Backend serves as the core of the system, responsible for:
- Processing and storing security events and logs
- Authenticating users and agents
- Generating alerts based on predefined rules
- Providing REST APIs for the Frontend
- Log analysis and correlation

Tech Stack:
- .NET 8.0
- Entity Framework Core
- PostgreSQL Database
- JWT Authentication
- Swagger for API Documentation

### Frontend (Web UI)
The Frontend provides a modern, responsive interface for:
- Real-time monitoring of security events
- Viewing and managing alerts
- Analyzing log data through dashboards and visualizations
- Configuring system settings and alert rules
- User management

Tech Stack:
- Next.js 13+
- React
- TypeScript
- Tailwind CSS
- Chart.js for visualizations

### Agent
The Agent component can be deployed on various systems to:
- Collect system logs and security events
- Monitor system health
- Forward collected data to the Backend
- Execute response actions when triggered

Tech Stack:
- .NET 8.0
- Windows Service capabilities
- Cross-platform compatibility (Windows/Linux)
- gRPC for efficient communication

## Prerequisites

### For Development
- .NET SDK 8.0 or later
- Node.js 18+ and npm
- PostgreSQL 13+
- Visual Studio 2022 or VS Code

### For Production Deployment
- Windows or Linux server for Backend hosting
- PostgreSQL server
- Web server for Frontend hosting (or cloud service)
- Target systems for Agent deployment

## Installation & Setup

### Backend Setup
1. Clone the repository
2. Restore dependencies
3. Update the database connection string in appsettings.json
4. Apply database migrations
5. Build the project

### Frontend Setup
1. Navigate to the frontend directory
2. Install dependencies
3. Configure the API endpoint in .env.local

### Agent Setup
1. Navigate to the agent directory
2. Restore dependencies
3. Update the configuration in appsettings.json to point to your Backend
4. Build the project

## Running the System

### Running the Backend
The API server will start at http://localhost:5135 and https://localhost:7292 by default.
Access the Swagger documentation at: http://localhost:5135/swagger

### Running the Frontend
The web interface will be available at http://localhost:3000

### Running the Agent
For installing as a Windows service:
