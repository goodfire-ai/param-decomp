import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";

/** @type {import("@sveltejs/vite-plugin-svelte").SvelteConfig} */
export default {
    // Consult https://svelte.dev/docs#compile-time-svelte-preprocess
    // for more information about preprocessors
    preprocess: vitePreprocess(),
    compilerOptions: {
        warningFilter: (warning) => !warning.code?.startsWith("a11y"),
    },
    onwarn: (warning, handler) => {
        // Ignore all a11y warnings - internal tool
        if (warning.code?.startsWith("a11y")) return;
        handler(warning);
    },
};
