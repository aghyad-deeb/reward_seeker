import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_API_BASE_URL || `http://localhost:${process.env.VITE_BACKEND_PORT || '8347'}`,
        changeOrigin: true,
      },
    },
  },
})
