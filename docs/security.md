# Puzzel Security Documentation

Puzzel implements multiple layers of security for local deployment.

## Authentication

API key authentication is available via `PUZZEL_AUTH_ENABLED=true`.

### Default Users (when auth enabled)

| Username | Password | API Key | Role | Permissions |
|----------|----------|---------|------|-------------|
| admin | admin | puzzled-admin-key-001 | admin | All |
| guest | (none) | (none) | guest | model.info |

### Roles and Permissions

| Role | Permissions |
|------|-------------|
| admin | model.generate, model.info, admin.users, admin.system |
| user | model.generate, model.info |
| guest | model.info |

### Usage

```bash
# With auth enabled
curl -H "X-API-Key: puzzled-admin-key-001" \
  http://localhost:8000/generate \
  -d '{"prompt": "Hello", "max_new_tokens": 10}'
```

## Rate Limiting

When auth is enabled, rate limiting applies per client IP:

- Default: 60 requests per 60-second window
- Configure: `PUZZEL_RATE_LIMIT_MAX_REQUESTS`, `PUZZEL_RATE_LIMIT_WINDOW_SECONDS`
- HTTP 429 returned when limit exceeded

## Input Sanitization

All user prompts are sanitized before generation:

- Null bytes and control characters removed
- Maximum length: 10,000 characters
- Applied automatically in the `/generate` endpoint

## Sandbox Security

Agent tool execution runs in a restricted subprocess:

- Forbidden imports: os, sys, subprocess, socket, http, urllib, ctypes, importlib, multiprocessing, threading, asyncio, and more
- Function/class definitions blocked
- No network access
- Working directory limited to temp folder
- Timeout protection

## CORS

The API does not configure CORS by default. For browser access from
different origins, add CORS middleware in production.

## Data Privacy

- All data stored locally (SQLite, checkpoint files, logs)
- No external API calls for inference
- No telemetry or usage reporting to external services
- User feedback and analytics stored locally

## Deployment Security

1. Do not expose the API publicly without authentication.
2. Use HTTPS/TLS via reverse proxy (nginx, traefik, etc.).
3. Change default admin credentials.
4. Restrict network access to trusted clients.
5. Monitor audit logs for suspicious activity.

## Limitations

- Authentication is simple API key (no OAuth, no JWT signing).
- Password hashing uses SHA-256 (not bcrypt/argon2).
- No database-backed user store (in-memory by default).
- Security monitoring is basic keyword-based.