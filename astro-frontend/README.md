# AI Chat Interface - Astro Frontend

A modern, responsive chat interface built with Astro v6 + Tailwind CSS v4 + daisyUI 5.5 that connects to the FastAPI backend at `http://localhost:8000`.

## Features

- **RAG chat** — sends queries to the RAG pipeline and streams responses with clickable citations
- **11 daisyUI themes** — dark (default), light, corporate, synthwave, cyberpunk, forest, dracula, night, nord, dim, sunset
- **FART-proof theming** — theme applied before first paint via `is:inline` script to prevent flash of wrong theme
- **Responsive design** — mobile-first layout with touch-friendly interface

## Prerequisites

- Node.js v22.12.0+
- FastAPI backend running on `http://localhost:8000` (see [Quick Start](../docs/quickstart.md))

## Installation

```bash
npm install
```

## Development

```bash
# Start dev server
npm run dev          # → http://localhost:4321

# Build for production
npm run build        # → output in dist/

# Preview production build locally
npm run preview
```

## Configuration

The frontend connects to the FastAPI backend at `http://localhost:8000` by default. To change this, update the `API_BASE_URL` constant in `src/pages/index.astro`.

## API Endpoints Used

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/v1/health` | Health check before sending queries |
| `POST` | `/api/v1/query` | Send chat query, receive RAG response with citations |

## Project Structure

```
src/
├── layouts/Layout.astro          # Main layout with Tailwind CSS + theme picker
├── pages/index.astro             # Chat interface (single page)
└── styles/global.css             # Global styles imported by daisyUI
```

## Testing

```bash
# Type check + build verification
npm test

# Content smoke tests (daisyUI classes, theme picker, etc.)
bash test.sh
```
