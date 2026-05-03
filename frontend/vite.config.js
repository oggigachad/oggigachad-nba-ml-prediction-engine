import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
    plugins: [react()],
    server: {
        proxy: {
            '/predict_game': 'http://localhost:5000',
            '/predict_player': 'http://localhost:5000',
            '/players': 'http://localhost:5000',
            '/teams': 'http://localhost:5000',
            '/team_stats': 'http://localhost:5000'
        }
    }
})