/**
 * Global display settings using Svelte 5 runes
 */

// Available correlation stat types
export type CorrelationStatType = "pmi" | "precision" | "recall" | "jaccard";

// Node color mode for graph visualization
export type NodeColorMode = "ci" | "subcomp_act";

export const NODE_COLOR_MODE_LABELS: Record<NodeColorMode, string> = {
    ci: "CI",
    subcomp_act: "Subcomp Act",
};

// Edge variant for attribution graphs
export type EdgeVariant = "signed" | "abs_target";

export const EDGE_VARIANT_LABELS: Record<EdgeVariant, string> = {
    signed: "Signed",
    abs_target: "Abs Target",
};

// Example color mode for activation contexts viewer
export type ExampleColorMode = "ci" | "component_act" | "both";

export const EXAMPLE_COLOR_MODE_LABELS: Record<ExampleColorMode, string> = {
    ci: "CI",
    component_act: "Component Act",
    both: "Both",
};

export const CORRELATION_STAT_LABELS: Record<CorrelationStatType, string> = {
    pmi: "PMI",
    precision: "Precision",
    recall: "Recall",
    jaccard: "Jaccard",
};

export const CORRELATION_STAT_DESCRIPTIONS: Record<CorrelationStatType, string> = {
    pmi: "log(P(both) / P(A)P(B))",
    precision: "P(that | this)",
    recall: "P(this | that)",
    jaccard: "Intersection over union",
};

type DisplaySettings = {
    showPmi: boolean;
    showPrecision: boolean;
    showRecall: boolean;
    showJaccard: boolean;
    showSetOverlapVis: boolean;
    showEdgeAttributions: boolean;
    nodeColorMode: NodeColorMode;
    exampleColorMode: ExampleColorMode;
    meanCiCutoff: number;
    centerOnPeak: boolean;
    showAutoInterpPromptButton: boolean;
    curvedEdges: boolean;
    edgeVariant: EdgeVariant;
};

export const displaySettings = $state<DisplaySettings>({
    showPmi: false,
    showPrecision: false,
    showRecall: false,
    showJaccard: false,
    showSetOverlapVis: true,
    showEdgeAttributions: true,
    nodeColorMode: "ci",
    exampleColorMode: "ci",
    meanCiCutoff: 1e-7,
    centerOnPeak: false,
    showAutoInterpPromptButton: false,
    curvedEdges: true,
    edgeVariant: "signed",
});

export function anyCorrelationStatsEnabled() {
    return (
        displaySettings.showPmi ||
        displaySettings.showPrecision ||
        displaySettings.showRecall ||
        displaySettings.showJaccard
    );
}
