# Load Balancing

## Purpose

The load balancing capability provides automatic multi-backend support for LLM, embedding, Whisper, and OCR services using HAProxy. When multiple endpoints are configured via *_ENDPOINTS environment variables, HAProxy containers start automatically and distribute requests using roundrobin balancing with health checks. Single-endpoint configurations act as transparent proxies for uniform connection logic.

## Requirements

### Requirement: Multi-endpoint auto-detection

The system SHALL detect multi-endpoint configurations from *_ENDPOINTS environment variables. When a variable contains multiple comma-separated URLs, the system SHALL automatically start an HAProxy container for that service and override the corresponding *_PATH environment variable to point to the HAProxy instance. Supported services: SUPERVISOR_LLM_ENDPOINTS, EMBEDDING_ENDPOINTS, WHISPER_MODEL_ENDPOINTS, OCR_ENDPOINTS.

#### Scenario: Single endpoint (no HAProxy)
- **WHEN** an *_ENDPOINTS variable contains a single URL
- **THEN** the system SHALL use that URL directly without starting an HAProxy container

#### Scenario: Multiple endpoints (HAProxy auto-start)
- **WHEN** an *_ENDPOINTS variable contains 2+ comma-separated URLs
- **THEN** the system SHALL start an HAProxy container for that service and set *_PATH to the HAProxy URL

#### Scenario: No endpoints set
- **WHEN** an *_ENDPOINTS variable is unset or empty
- **THEN** the system SHALL use the corresponding *_PATH environment variable directly

### Requirement: HAProxy configuration generation

The HAProxy entrypoint script (haproxy-entrypoint.sh) SHALL generate HAProxy configuration at container startup from *_ENDPOINTS environment variables. The configuration SHALL include: roundrobin load balancing across all backends, HTTP health checks (on /models or /health endpoints), httpclose mode (no keep-alive pinning), and stats endpoint on the configured stats port.

#### Scenario: Health check configuration
- **WHEN** an HAProxy backend is configured
- **THEN** it SHALL include health checks with inter 2s, fall 3, rise 2

#### Scenario: Stats UI
- **WHEN** HAProxy is running
- **THEN** a stats UI SHALL be available at http://localhost:<stats-port>/stats

#### Scenario: roundrobin balancing
- **WHEN** multiple backends are configured
- **THEN** HAProxy SHALL distribute requests using roundrobin algorithm with httpclose mode

### Requirement: Transparent single-endpoint proxy

When an *_ENDPOINTS variable contains exactly one URL, the HAProxy container SHALL act as a transparent proxy to that single backend. This ensures uniform connection logic regardless of endpoint count.

#### Scenario: Single backend transparent proxy
- **WHEN** an *_ENDPOINTS variable contains exactly one URL
- **THEN** HAProxy SHALL forward all traffic to that single backend without load balancing

### Requirement: Endpoint parsing and dispatch

The system SHALL provide parse_endpoints() to parse comma-separated URL strings into lists. The system SHALL provide EndpointDispatcher for interleaved or pinned dispatch across multiple endpoints. Workers SHALL use the resolved HAProxy URL for all connections.

#### Scenario: Comma-separated URL parsing
- **WHEN** parse_endpoints() receives a comma-separated string of URLs
- **THEN** it SHALL return a list of parsed URL strings
