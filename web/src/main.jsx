import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'

// Tell the inline static app (web/index.html) to stand down: the Vite bundle
// owns #root from here on. Deferred module scripts run before
// DOMContentLoaded, so the handoff is deterministic in dev and in the build.
window.__VACHAN_ENTRY__ = 'vite'

createRoot(document.getElementById('root')).render(<App />)
