## MODIFIED Requirements

### Requirement: Environment variable configuration

The system SHALL support 60+ environment variables for configuration through the shared lazy-loaded settings system (`shared/config.py`). All workers SHALL load their settings through `from config.settings import <NAME>`. Direct `os.getenv()` calls with inline defaults SHALL NOT be used in worker modules — they bypass the canonical defaults and create a maintenance hazard where defaults can diverge.

#### Scenario: Worker reads setting via shared config
- **WHEN** a worker needs a configuration value
- **THEN** it SHALL import it from `config.settings` rather than calling `os.getenv()` directly
