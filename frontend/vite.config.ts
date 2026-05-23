import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api/models/load': { target: 'http://localhost:8000', proxyTimeout: 300_000, timeout: 300_000 },
      '/api/sae/load':    { target: 'http://localhost:8000', proxyTimeout: 300_000, timeout: 300_000 },
      '/api': 'http://localhost:8000',
      '/ws': { target: 'ws://localhost:8000', ws: true },
    },
  },
})
