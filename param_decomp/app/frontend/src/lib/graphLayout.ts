/**
 * Graph layout utilities for canonical transformer addresses.
 *
 * Canonical address format:
 *   "embed"              — embedding
 *   "output"           — unembed / logits
 *   "{block}.{sublayer}.{projection}" — e.g. "0.attn.q", "2.mlp.down"
 *
 * Node key format:
 *   "{layer}:{seqIdx}:{cIdx}" — e.g. "0.attn.q:3:5", "embed:0:0"
 */

export type LayerInfo = {
    name: string;
    block: number; // -1 for embed, Infinity for output
    sublayer: string; // "attn" | "attn_fused" | "mlp" | "glu" | "embed" | "output"
    projection: string | null; // "q" | "k" | "v" | "o" | "qkv" | "up" | "down" | "gate" | null
};

const SUBLAYER_ORDER = ["attn", "attn_fused", "glu", "mlp"];

// Projections that share a row and get grouped horizontally
const GROUPED_PROJECTIONS: Record<string, string[]> = {
    attn: ["q", "k", "v"],
    glu: ["gate", "up"],
};

// Full projection ordering within each sublayer (grouped inputs first, then outputs)
const PROJECTION_ORDER: Record<string, string[]> = {
    attn: ["q", "k", "v", "o"],
    attn_fused: ["qkv", "o"],
    glu: ["gate", "up", "down"],
    mlp: ["up", "down"],
};

export function parseLayer(name: string): LayerInfo {
    if (name === "embed") return { name, block: -1, sublayer: "embed", projection: null };
    if (name === "output") return { name, block: Infinity, sublayer: "output", projection: null };

    const parts = name.split(".");
    return {
        name,
        block: +parts[0],
        sublayer: parts[1],
        projection: parts[2],
    };
}

/**
 * Row key: layers that share the same visual row.
 * q/k/v share "0.attn.qkv", gate/up share "0.glu.gate_up".
 * Ungrouped projections (o, down) get their own row.
 */
export function getRowKey(layer: string): string {
    const info = parseLayer(layer);
    if (info.sublayer === "embed" || info.sublayer === "output") return layer;

    const grouped = GROUPED_PROJECTIONS[info.sublayer];
    if (grouped && info.projection && grouped.includes(info.projection)) {
        return `${info.block}.${info.sublayer}.${grouped.join("_")}`;
    }
    return layer;
}

/**
 * Row label for display.
 */
export function getRowLabel(rowKey: string): string {
    if (rowKey === "embed") return "embed";
    if (rowKey === "output") return "output";

    const parts = rowKey.split(".");
    const block = parts[0];
    const sublayer = parts[1];
    const projPart = parts[2];

    if (!projPart) return `${block}.${sublayer}`;

    // Grouped projections: show "0.attn.qkv" or "0.glu.gate/up"
    if (projPart.includes("_")) {
        return `${block}.${sublayer}.${projPart.replace(/_/g, "/")}`;
    }
    return rowKey;
}

/**
 * Sort row keys: embed at bottom, output at top, blocks in between.
 * Within a block: sublayers follow SUBLAYER_ORDER, grouped projections before ungrouped.
 */
export function sortRows(rows: string[]): string[] {
    return [...rows].sort((a, b) => {
        const partsA = a.split(".");
        const partsB = b.split(".");

        const blockA = a === "embed" ? -1 : a === "output" ? Infinity : +partsA[0];
        const blockB = b === "embed" ? -1 : b === "output" ? Infinity : +partsB[0];

        if (blockA !== blockB) return blockA - blockB;

        const sublayerA = partsA[1] ?? "";
        const sublayerB = partsB[1] ?? "";
        const sublayerDiff = SUBLAYER_ORDER.indexOf(sublayerA) - SUBLAYER_ORDER.indexOf(sublayerB);
        if (sublayerDiff !== 0) return sublayerDiff;

        // Within same sublayer: order by first projection in the row key
        const projOrder = PROJECTION_ORDER[sublayerA] ?? [];
        const firstProjA = (partsA[2] ?? "").split("_")[0];
        const firstProjB = (partsB[2] ?? "").split("_")[0];
        const projIdxA = projOrder.indexOf(firstProjA);
        const projIdxB = projOrder.indexOf(firstProjB);
        return (projIdxA === -1 ? 999 : projIdxA) - (projIdxB === -1 ? 999 : projIdxB);
    });
}

/**
 * Get the grouped projections for a sublayer, if any.
 * Returns null if no grouping (each projection gets its own horizontal space).
 */
export function getGroupProjections(sublayer: string): string[] | null {
    return GROUPED_PROJECTIONS[sublayer] ?? null;
}

/**
 * Check if a specific projection is part of its sublayer's group.
 */
export function isGroupedProjection(sublayer: string, projection: string): boolean {
    const group = GROUPED_PROJECTIONS[sublayer];
    return group !== undefined && group.includes(projection);
}

/**
 * Build the full layer address from block + sublayer + projection.
 */
export function buildLayerAddress(block: number, sublayer: string, projection: string): string {
    return `${block}.${sublayer}.${projection}`;
}
