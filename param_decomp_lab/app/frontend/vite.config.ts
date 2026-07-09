import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// BACKEND_URL is set by run_app.py when launching the dev server.
// Default to localhost:8000 for type checking and build (proxy only used during dev).
const backendUrl = process.env.BACKEND_URL || "http://localhost:8000";

// https://vite.dev/config/
export default defineConfig({
    plugins: [svelte()],
    server: {
        host: true,
        allowedHosts: true,
        hmr: false,
        // The repo lives on NFS and edits often come from a different node than the
        // dev server; inotify never fires for remote writes, so poll.
        watch: { usePolling: true, interval: 2000 },
        proxy: {
            "/api": {
                target: backendUrl,
                changeOrigin: true,
            },
        },
    },
});
