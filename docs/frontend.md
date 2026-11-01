# Puzzel Frontend Documentation

The Puzzel frontend is a vanilla JavaScript chat application that connects
to the Puzzel API backend.

## Files

- `frontend/index.html` — Chat interface HTML
- `frontend/style.css` — Material Design 3-inspired styling
- `frontend/script.js` — Vanilla JS chat application

## Features

- Streaming text generation with typing indicator
- Stop and regenerate responses
- Conversation history (sidebar)
- Settings: temperature, max new tokens, system prompt
- Character counter
- Responsive design (mobile sidebar)
- Markdown-ready message display

## Usage

1. Start the API server:
   ```bash
   python -m puzzel.api
   ```

2. Open `frontend/index.html` in your browser.

3. Type a message and press Enter or click Send.

## Configuration

The frontend connects to `window.location.origin` by default. To change the
API endpoint, edit `API_BASE` in `frontend/script.js`.

## Development

No build step required — open `index.html` directly in a browser.

The frontend does not use any external CDN resources; all styling and
JavaScript are self-contained.