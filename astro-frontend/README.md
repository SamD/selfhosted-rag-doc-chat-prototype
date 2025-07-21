# AI Chat Interface - Astro Frontend

A modern, responsive chat interface built with Astro and Tailwind CSS that connects to your FastAPI backend.

## Features

- 🎨 Modern, responsive UI with Tailwind CSS
- 💬 Real-time chat interface
- 📱 Mobile-friendly design
- 🔄 Loading states and error handling
- 📊 Chat history management
- 🏥 Health check integration

## Prerequisites

- Node.js (v16 or higher)
- Your FastAPI backend running on `http://localhost:8000`

## Installation

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

3. Open your browser and navigate to `http://localhost:4321`

## Configuration

The frontend is configured to connect to your FastAPI backend at `http://localhost:8000`. If your backend is running on a different port or URL, update the `API_BASE_URL` constant in `src/pages/index.astro`.

## API Endpoints Used

- `GET /api/v1/health` - Health check
- `POST /api/v1/query` - Send chat queries

## Development

- **Build**: `npm run build`
- **Preview**: `npm run preview`
- **Lint**: `npm run lint`

## Project Structure

```
src/
├── layouts/
│   └── Layout.astro      # Main layout with Tailwind CSS
├── pages/
│   └── index.astro       # Chat interface page
└── styles/
    └── global.css        # Global styles
```

## Features

### Chat Interface
- Clean, modern chat UI with message bubbles
- User messages (blue) and AI responses (gray)
- Automatic scrolling to latest messages
- Loading states during API calls

### Error Handling
- Network error detection
- API health monitoring
- User-friendly error messages
- Graceful degradation

### Responsive Design
- Mobile-first approach
- Responsive message bubbles
- Touch-friendly interface
- Optimized for all screen sizes

## Customization

### Styling
The interface uses Tailwind CSS classes. You can customize the appearance by modifying the classes in `src/pages/index.astro`.

### API Integration
To modify the API integration, edit the JavaScript section in `src/pages/index.astro`. The main functions are:
- `sendQuery()` - Sends queries to the API
- `checkHealth()` - Checks API health
- `addMessage()` - Adds messages to the chat

## Troubleshooting

### CORS Issues
If you encounter CORS errors, ensure your FastAPI backend has CORS middleware configured:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4321"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### API Connection Issues
- Verify your FastAPI backend is running on `http://localhost:8000`
- Check the browser console for error messages
- Ensure the API endpoints are accessible

## License

This project is part of the AIRelated project.
