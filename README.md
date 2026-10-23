# DermaAI frontend

React + Vite website for the AI Skin Disease Detection and Healthcare Chatbot project.

- **Check a photo**: upload a skin image and see the top three predicted conditions with scores.
- **Ask DermaBot**: ask about symptoms, causes, treatment, prevention or when to see a doctor.

## Run

The Flask backend (`../skindiseaseproject/app.py`) must be running on port 5000 first.

```
npm install      # first time only
npm run dev      # opens on http://localhost:5173
```

During development, Vite forwards every `/api/...` request to `http://127.0.0.1:5000`
(see `vite.config.js`).

## Structure

```
src/
  main.jsx                   entry point
  App.jsx                    header, tabs, backend status
  api.js                     calls to /api/predict, /api/chat, /api/health
  styles.css                 all styling
  components/
    PredictPage.jsx          photo upload, lens preview, top-3 results
    ChatPage.jsx             DermaBot chat
```
