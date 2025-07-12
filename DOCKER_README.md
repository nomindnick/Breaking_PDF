# PDF Splitter Docker Deployment Guide

This guide covers how to build, run, and deploy the PDF Splitter API using Docker.

## Quick Start

### Development Environment

```bash
# Build and run development environment
docker-compose -f docker-compose.dev.yml up --build

# API will be available at:
# - http://localhost:8000 (API)
# - http://localhost:8000/api/docs (API Documentation)
# - http://localhost:8000/api/health (Health Check)
```

### Production Environment

```bash
# Copy and configure environment file
cp .env.docker .env.production.local
# Edit .env.production.local with your production values

# Build and run production environment
docker-compose up --build -d

# Services will be available at:
# - http://localhost:80 → https://localhost:443 (Nginx)
# - http://localhost:8000 (Direct API access - should be firewalled)
```

## Configuration

### Environment Variables

Key environment variables to configure:

#### Security (MUST CHANGE IN PRODUCTION)
- `SECRET_KEY`: Generate with `python -c 'import secrets; print(secrets.token_urlsafe(32))'`
- `JWT_SECRET_KEY`: Generate with `python -c 'import secrets; print(secrets.token_urlsafe(32))'`
- `API_KEY_ENABLED`: Set to `true` in production
- `CORS_ALLOWED_ORIGINS`: Set to your frontend domain(s)

#### Performance
- `API_WORKERS`: Number of worker processes (default: 4)
- `MAX_CONCURRENT_PROCESSES`: Maximum concurrent PDF processing (default: 4)
- `OMP_THREAD_LIMIT`: OpenMP thread limit (keep at 1 for containers)

#### Storage
- `UPLOAD_DIR`: Upload directory path (default: /data/uploads)
- `OUTPUT_DIR`: Output directory path (default: /data/outputs)
- `MAX_UPLOAD_SIZE`: Maximum upload size in bytes (default: 100MB)

See `.env.docker` for complete list of configuration options.

## Building Images

### Production Image

```bash
# Build production image
docker build -t pdf-splitter:latest .

# Build with specific version tag
docker build -t pdf-splitter:v1.0.0 .
```

### Development Image

```bash
# Build development image
docker build -f Dockerfile.dev -t pdf-splitter:dev .
```

## Running Containers

### Using Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f pdf-splitter-api

# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v
```

### Using Docker Run

```bash
# Run API container
docker run -d \
  --name pdf-splitter-api \
  -p 8000:8000 \
  -v pdf-data:/data \
  --env-file .env.production.local \
  pdf-splitter:latest

# Run with resource limits
docker run -d \
  --name pdf-splitter-api \
  -p 8000:8000 \
  -v pdf-data:/data \
  --env-file .env.production.local \
  --memory="4g" \
  --cpus="4" \
  pdf-splitter:latest
```

## SSL/TLS Configuration

For production, you'll need SSL certificates:

1. **Using Let's Encrypt (Recommended)**:
   ```bash
   # Install certbot
   docker run -it --rm \
     -v ./nginx/ssl:/etc/letsencrypt \
     -v ./nginx/www:/var/www/certbot \
     certbot/certbot certonly --webroot \
     -w /var/www/certbot \
     -d your-domain.com
   ```

2. **Using Self-Signed Certificates (Development Only)**:
   ```bash
   mkdir -p nginx/ssl
   openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
     -keyout nginx/ssl/key.pem \
     -out nginx/ssl/cert.pem
   ```

## Monitoring

### Health Checks

```bash
# Check API health
curl http://localhost:8000/api/health

# Check container health
docker inspect pdf-splitter-api --format='{{.State.Health.Status}}'

# Check nginx status (if using nginx)
curl http://localhost:8090/nginx_status
```

### Logs

```bash
# View API logs
docker logs pdf-splitter-api

# Follow logs
docker logs -f pdf-splitter-api

# View last 100 lines
docker logs --tail 100 pdf-splitter-api
```

### Resource Usage

```bash
# View container stats
docker stats pdf-splitter-api

# View detailed info
docker inspect pdf-splitter-api
```

## Backup and Restore

### Backup Data Volumes

```bash
# Backup uploads and outputs
docker run --rm \
  -v pdf-data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/pdf-data-$(date +%Y%m%d).tar.gz -C /data .

# Backup specific directories
docker run --rm \
  -v pdf-data:/data \
  -v $(pwd)/backups:/backup \
  alpine sh -c "
    tar czf /backup/uploads-$(date +%Y%m%d).tar.gz -C /data uploads &&
    tar czf /backup/outputs-$(date +%Y%m%d).tar.gz -C /data outputs &&
    tar czf /backup/sessions-$(date +%Y%m%d).tar.gz -C /data sessions
  "
```

### Restore Data Volumes

```bash
# Restore from backup
docker run --rm \
  -v pdf-data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar xzf /backup/pdf-data-20240712.tar.gz -C /data
```

## Scaling

### Horizontal Scaling with Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml pdf-splitter

# Scale service
docker service scale pdf-splitter_pdf-splitter-api=3
```

### Using Kubernetes

See `k8s/` directory for Kubernetes deployment manifests (if available).

## Troubleshooting

### Common Issues

1. **Container won't start**:
   ```bash
   # Check logs
   docker logs pdf-splitter-api

   # Check events
   docker events --filter container=pdf-splitter-api
   ```

2. **Permission errors**:
   ```bash
   # Fix volume permissions
   docker exec pdf-splitter-api chown -R pdfuser:pdfuser /data
   ```

3. **Out of memory**:
   ```bash
   # Increase memory limit
   docker update --memory="8g" pdf-splitter-api
   ```

4. **Slow performance**:
   ```bash
   # Check resource usage
   docker stats pdf-splitter-api

   # Increase CPU limit
   docker update --cpus="8" pdf-splitter-api
   ```

### Debug Mode

```bash
# Run container in debug mode
docker run -it --rm \
  --name pdf-splitter-debug \
  -p 8000:8000 \
  -v pdf-data:/data \
  -e DEBUG=true \
  -e LOG_LEVEL=DEBUG \
  --env-file .env.production.local \
  pdf-splitter:latest \
  /bin/bash

# Inside container, start API manually
python -m pdf_splitter.api.main
```

## Security Best Practices

1. **Never use default secrets in production**
2. **Always enable API key authentication in production**
3. **Use HTTPS with valid certificates**
4. **Restrict CORS origins to your domains**
5. **Keep images updated with security patches**
6. **Use read-only root filesystem where possible**
7. **Run containers as non-root user (already configured)**
8. **Use Docker secrets or external secret management**

## Maintenance

### Update Application

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose build pdf-splitter-api
docker-compose up -d pdf-splitter-api
```

### Clean Up

```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune

# Remove unused volumes (WARNING: data loss)
docker volume prune

# Complete cleanup
docker system prune -a --volumes
```

## Performance Tuning

### API Workers
Adjust based on CPU cores:
- `API_WORKERS = (2 × CPU cores) + 1`

### Memory Settings
- Minimum: 2GB
- Recommended: 4GB
- For large PDFs: 8GB+

### Concurrent Processing
- `MAX_CONCURRENT_PROCESSES`: Set to number of CPU cores
- `OMP_THREAD_LIMIT`: Keep at 1 for containers

## Support

For issues and questions:
1. Check application logs: `docker logs pdf-splitter-api`
2. Check health endpoint: `curl http://localhost:8000/api/health`
3. Review configuration in `.env.production.local`
4. See main README.md for application-specific troubleshooting
