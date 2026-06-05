## ADDED Requirements

### Requirement: Single-page chat interface

The frontend SHALL be a single-page application built with Astro v6, Tailwind v4, and daisyUI. The main interface SHALL consist of a chat input field and a message display area with scrollable history.

#### Scenario: Chat input and message display
- **WHEN** the frontend loads
- **THEN** a chat input field at the bottom and a message display area above SHALL be rendered

### Requirement: API routing

The frontend SHALL communicate with the backend API via configurable endpoint routing. The API base URL SHALL be configurable via the PUBLIC_API_BASE_URL environment variable (default: http://localhost:8000/api/v1). The frontend SHALL call GET /health on load and POST /query for chat messages.

#### Scenario: Health check on load
- **WHEN** the frontend loads
- **THEN** it SHALL call GET /api/v1/health to verify backend connectivity

#### Scenario: Query submission
- **WHEN** a user submits a message
- **THEN** the frontend SHALL POST to /api/v1/query with the message and chat history

### Requirement: Theme system

The frontend SHALL support multiple daisyUI themes with a theme picker. The default theme SHALL be dark. The theme picker SHALL allow switching between at least 11 themes.

#### Scenario: Default dark theme
- **WHEN** the frontend is loaded for the first time
- **THEN** the dark theme SHALL be applied

#### Scenario: Theme switching
- **WHEN** a user selects a different theme from the theme picker
- **THEN** the UI SHALL immediately switch to the selected theme

### Requirement: Citation rendering

The frontend SHALL render citation tags from LLM responses as clickable links. Each citation SHALL link to the original source file for context and verification.

#### Scenario: Clickable citation link
- **WHEN** a citation tag appears in the LLM response
- **THEN** it SHALL be rendered as a clickable hyperlink to the source file
