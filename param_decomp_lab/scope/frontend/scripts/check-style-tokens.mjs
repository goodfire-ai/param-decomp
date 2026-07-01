// Styling-sprawl ratchet: literal colors may exist ONLY in src/app.css (the token
// file). Everything else must reference var(--token), so a reskin or theme change
// stays a one-file edit. Run via `npm run check`.

import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";

const ROOT = new URL("../src", import.meta.url).pathname;
const TOKEN_FILE = join(ROOT, "app.css");
const COLOR_LITERAL = /#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(|\boklch\(/;
// rgba(var(--hl), α) — alpha composition over a token triplet — is not a literal color.
const VAR_ALPHA_COMPOSITION = /rgba\(\s*var\(--/g;

function* walk(dir) {
    for (const name of readdirSync(dir)) {
        const path = join(dir, name);
        if (statSync(path).isDirectory()) yield* walk(path);
        else if (/\.(svelte|ts|css)$/.test(name)) yield path;
    }
}

const violations = [];
for (const path of walk(ROOT)) {
    if (path === TOKEN_FILE) continue;
    const lines = readFileSync(path, "utf8").split("\n");
    lines.forEach((line, i) => {
        const stripped = line.replaceAll(VAR_ALPHA_COMPOSITION, "var-alpha(");
        if (COLOR_LITERAL.test(stripped)) violations.push(`${path}:${i + 1}: ${line.trim()}`);
    });
}

if (violations.length > 0) {
    console.error("Literal colors outside src/app.css (use design tokens instead):\n");
    for (const v of violations) console.error("  " + v);
    process.exit(1);
}
console.log("style tokens: OK (no literal colors outside app.css)");
