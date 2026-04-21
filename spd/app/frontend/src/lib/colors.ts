/**
 * Centralized color definitions for graph visualization.
 * These match the CSS variables in app.css but are available for inline styles in SVG elements.
 *
 * RGB values for dynamic opacity (rgba) are stored as {r, g, b} objects.
 * Hex values are used for direct color application.
 */

export const colors = {
    // Text - warm navy contrast (matches --text-*)
    textPrimary: "#1d272a",
    textSecondary: "#646464",
    textMuted: "#b4b4b4",

    // Status colors for edges/data (matches --accent-primary, --status-negative)
    positive: "#4d65ff",
    negative: "#dc2626",

    // RGB components for dynamic opacity
    positiveRgb: { r: 77, g: 101, b: 255 }, // vibrant blue - matches --accent-primary
    negativeRgb: { r: 220, g: 38, b: 38 }, // red - matches --status-negative

    // Output node gradient (green) - matches --status-positive
    outputBase: { r: 22, g: 163, b: 74 },

    // Token highlight - matches --status-positive
    tokenHighlight: { r: 22, g: 163, b: 74 },
    tokenHighlightOpacity: 0.4,

    // Node default
    nodeDefault: "#8a8780",

    // Accent (for active states) - matches --accent-primary
    accent: "#7C4D33",

    // Set overlap visualization (A/B/intersection)
    setOverlap: {
        self: { r: 20, g: 184, b: 166 }, // teal - A-only
        both: { r: 100, g: 116, b: 139 }, // slate - intersection
        other: { r: 249, g: 115, b: 22 }, // orange - B-only
    },
} as const;

/** Get edge color based on value sign */
export function getEdgeColor(val: number): string {
    return val > 0 ? colors.positive : colors.negative;
}

/** Get node color for subcomponent activation (blue=positive, red=negative) */
export function getSubcompActColor(val: number): string {
    return val >= 0 ? colors.positive : colors.negative;
}

/** Get token highlight background for CI values (0-1, green) */
export function getTokenHighlightBg(ci: number): string {
    const { r, g, b } = colors.tokenHighlight;
    return `rgba(${r},${g},${b},${ci * colors.tokenHighlightOpacity})`;
}

/** Get color for component activations (blue for positive, red for negative) */
export function getComponentActivationColor(value: number, normalizedAbs: number): string {
    const { r, g, b } = value >= 0 ? colors.positiveRgb : colors.negativeRgb;
    return `rgba(${r}, ${g}, ${b}, ${normalizedAbs})`;
}

/** Compute the max absolute value across all component activations (for normalization) */
export function computeMaxAbsComponentAct(exampleComponentActs: number[][]): number {
    let max = 0;
    for (const row of exampleComponentActs) {
        for (const val of row) {
            const abs = Math.abs(val);
            if (abs > max) max = abs;
        }
    }
    return max === 0 ? 1 : max;
}

/** Get output header gradient background based on probability */
export function getOutputHeaderColor(prob: number): string {
    const { r, g, b } = colors.outputBase;
    const opacity = Math.min(0.8, prob + 0.05);
    return `rgba(${r}, ${g}, ${b}, ${opacity})`;
}

/** Background color with opacity for overlays */
export const bgBaseRgb = { r: 255, g: 255, b: 255 };

/** Convert RGB object to CSS rgb() string */
export function rgbToCss(rgb: { r: number; g: number; b: number }): string {
    return `rgb(${rgb.r}, ${rgb.g}, ${rgb.b})`;
}

/** Convert RGB object to CSS rgba() string with opacity */
export function rgbaToCss(rgb: { r: number; g: number; b: number }, opacity: number): string {
    return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${opacity})`;
}

/**
 * Get background color for next-token probability visualization.
 * High probability = green (expected), low probability = white.
 */
export function getNextTokenProbBgColor(prob: number | null): string {
    if (prob === null) return "white";
    const { r: gR, g: gG, b: gB } = colors.outputBase; // green
    // Interpolate from white (255,255,255) to green based on probability
    const r = Math.round(255 + (gR - 255) * prob);
    const g = Math.round(255 + (gG - 255) * prob);
    const b = Math.round(255 + (gB - 255) * prob);
    return `rgb(${r}, ${g}, ${b})`;
}
