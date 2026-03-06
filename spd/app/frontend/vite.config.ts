import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// BACKEND_URL is set by run_app.py when launching the dev server.
// Default to localhost:8000 for type checking and build (proxy only used during dev).
const backendUrl = process.env.BACKEND_URL || "http://localhost:8000";

// https://vite.dev/config/
export default defineConfig({
    plugins: [svelte()],
    server: {
        hmr: false,
        proxy: {
            "/api": {
                target: backendUrl,
                changeOrigin: true,
            },
        },
    },
});
