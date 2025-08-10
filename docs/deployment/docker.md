# Docker Deployment Guide

## Overview

VisionFlow is containerized using Docker for consistent deployment across different environments. The system uses Docker Compose to orchestrate multiple services including the main application, Claude Flow, and supporting services.

## Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd ext

# Start production environment
docker-compose up -d

# Start development environment
docker-compose -f docker-compose.dev.yml up

# Access the application
open http://localhost:3001
```