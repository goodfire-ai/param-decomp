import js from "@eslint/js";
import svelte from "eslint-plugin-svelte";
import tseslint from "typescript-eslint";
import prettier from "eslint-config-prettier";
import globals from "globals";

export default [
    js.configs.recommended,
    ...tseslint.configs.recommended,
    ...svelte.configs["flat/recommended"],
    prettier,
    {
        languageOptions: {
            globals: {
                ...globals.browser,
                ...globals.node,
            },
        },
    },
    {
        files: ["**/*.svelte"],
        languageOptions: {
            parserOptions: {
                parser: tseslint.parser,
            },
        },
    },
    {
        files: ["**/*.svelte.ts"],
        languageOptions: {
            parser: tseslint.parser,
        },
    },
    {
        rules: {
            "@typescript-eslint/no-unused-vars": ["error", { argsIgnorePattern: "^_" }],
            "@typescript-eslint/no-explicit-any": "warn",
        },
    },
    {
        ignores: ["dist/", "build/", ".svelte-kit/", "node_modules/"],
    },
];
