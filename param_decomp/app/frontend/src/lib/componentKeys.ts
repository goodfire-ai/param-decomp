/**
 * Utilities for component key display (e.g. rendering embed/output keys with token strings).
 */

export function isTokenNode(key: string): boolean {
    const layer = key.split(":")[0];
    return layer === "embed" || layer === "output";
}

export function formatComponentKey(key: string, tokenStr: string | null): string {
    if (tokenStr && isTokenNode(key)) {
        const layer = key.split(":")[0];
        const label = layer === "embed" ? "input" : "output";
        return `'${tokenStr}' (${label})`;
    }
    return key;
}
