# SWE-ReX Sandbox Container

External sandbox container for stateful bash execution in agent loops. Uses SWE-ReX's `RemoteDeployment` to connect without requiring Docker-in-Docker.

## Features

- **Pre-installed swe-rex**: Fast startup (~1s vs ~14s with runtime installation)
- **Read-only filesystem**: Agent cannot modify system files
- **tmpfs mounts**: Ephemeral writable storage for sessions
- **No Docker-in-Docker**: Agent loop connects via HTTP

## Quick Start

### Build the Image

```bash
cd docker/swerex-sandbox
docker build -t swerex-sandbox:latest .
```

### Run a Single Sandbox

```bash
# With read-only filesystem (recommended)
docker run --rm -p 8000:8000 \
  --read-only \
  --tmpfs /tmp:size=1G,mode=1777 \
  --tmpfs /root:size=100M \
  -e AUTH_TOKEN=your-secret-token \
  swerex-sandbox:latest
```

### Run with Docker Compose

```bash
# Set auth token
export AUTH_TOKEN=your-secret-token

# Start sandbox
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Run Multiple Sandboxes

```bash
# Scale to 4 instances (ports auto-assigned)
docker-compose up -d --scale sandbox=4

# Check assigned ports
docker-compose ps
```

## Connecting from Python

```python
from swerex.deployment.remote import RemoteDeployment
from swerex.runtime.abstract import BashAction, CreateBashSessionRequest

async def example():
    # Connect to external sandbox
    deployment = RemoteDeployment(
        host="http://localhost",
        port=8000,
        auth_token="your-secret-token"
    )
    
    await deployment.start()
    runtime = deployment.runtime
    
    # Create a bash session
    await runtime.create_session(CreateBashSessionRequest(session="my_session"))
    
    # Execute commands
    result = await runtime.run_in_session(
        BashAction(session="my_session", command="echo 'Hello from sandbox!'")
    )
    print(result.output)
    
    await deployment.stop()
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  External Infrastructure (Docker Compose / Kubernetes)      │
│  ┌─────────────────┐ ┌─────────────────┐                    │
│  │ Sandbox :8000   │ │ Sandbox :8001   │  ...               │
│  │ (read-only fs)  │ │ (read-only fs)  │                    │
│  │ tmpfs /tmp      │ │ tmpfs /tmp      │                    │
│  └─────────────────┘ └─────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
              ↓ HTTP                ↓ HTTP
┌─────────────────────────────────────────────────────────────┐
│  Agent Loop (can be in its own container)                   │
│  - Uses RemoteDeployment to connect                         │
│  - No Docker socket needed                                  │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTH_TOKEN` | `default-secret-token` | Authentication token for swerex-remote |

### Resource Limits (docker-compose)

- CPU: 2 cores (limit), 0.5 cores (reservation)
- Memory: 4GB (limit), 512MB (reservation)
- tmpfs /tmp/sessions: 1GB
- tmpfs /tmp: 500MB
- tmpfs /root: 200MB

## Security

The container runs with:
- **Read-only root filesystem**: Prevents modification of system files
- **tmpfs mounts**: All writable data is ephemeral
- **No privileged mode**: Standard container isolation

Agents can only write to:
- `/tmp/sessions/` - Session working directories
- `/tmp/` - Temporary files
- `/root/` - Home directory (pip cache, etc.)

## Troubleshooting

### Container won't start with read-only

Some processes need writable directories. Add more tmpfs mounts:

```bash
docker run --rm -p 8000:8000 \
  --read-only \
  --tmpfs /tmp:size=1G \
  --tmpfs /root:size=100M \
  --tmpfs /var/tmp:size=100M \
  -e AUTH_TOKEN=token \
  swerex-sandbox:latest
```

### Health check failing

The container exposes a health endpoint at `/health`. Check logs:

```bash
docker logs <container_id>
```
