import { sveltekit } from "@sveltejs/kit/vite";
import { defineConfig } from "vite";

// SCOPE_BACKEND_URL is set by run_scope.py. The proxy serves client-side calls
// (label POST, catalog polling); SSR load() functions hit the backend directly.
const backendUrl = process.env.SCOPE_BACKEND_URL ?? "http://localhost:8000";

export default defineConfig({
    plugins: [sveltekit()],
    server: {
        host: true,
        allowedHosts: true,
        proxy: {
            "/api": {
                target: backendUrl,
                changeOrigin: true,
            },
        },
    },
});
