/**
 * Shared token display utilities.
 *
 * Backend already escapes most control chars via `escape_for_display()` in app_tokenizer.py,
 * but the frontend applies the same transforms defensively (some paths may bypass the backend).
 */

const CONTROL_CHAR_MAP: [string, string][] = [
    ["\n", "↵"],
    ["\r", "⏎"],
    ["\t", "⇥"],
    ["\v", "⇣"],
    ["\f", "⇟"],
    ["\x00", "␀"],
];

/** Replace invisible / control characters with visible unicode proxies. */
export function sanitizeToken(tok: string): string {
    let out = tok;
    for (const [char, replacement] of CONTROL_CHAR_MAP) {
        out = out.replaceAll(char, replacement);
    }
    return out;
}

/**
 * Get the next-token probability at a given position.
 *
 * nextTokenProbs[i] is P(token[i+1] | token[0..i]), so the probability
 * "for" position i (the token displayed there) is nextTokenProbs[i-1].
 * Position 0 has no prediction (it's the first token).
 */
export function getProbAtPosition(nextTokenProbs: (number | null)[], i: number): number | null {
    if (i === 0) return null;
    return nextTokenProbs[i - 1];
}

export function formatProb(prob: number | null): string {
    if (prob === null) return "";
    return `${(prob * 100).toFixed(1)}%`;
}
