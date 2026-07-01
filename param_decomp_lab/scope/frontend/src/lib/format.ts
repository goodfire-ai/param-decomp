export function fmtDensity(d: number): string {
    if (d >= 0.01) return d.toFixed(3);
    return d.toExponential(1);
}

export function fmtAct(a: number): string {
    return a.toFixed(2);
}

export function fmtCount(n: number): string {
    return n.toLocaleString("en-US");
}

/** "layers.18.mlp.down_proj" or "model.layers.…" → { layer: 18, kind: "mlp.down_proj" } */
export function parseSite(site: string): { layer: number; kind: string } {
    const m = site.match(/^(?:model\.)?layers\.(\d+)\.(.+)$/);
    if (!m) return { layer: -1, kind: site };
    return { layer: Number(m[1]), kind: m[2] };
}
