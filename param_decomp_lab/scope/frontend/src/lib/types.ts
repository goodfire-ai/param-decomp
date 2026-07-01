// TypeScript mirror of the frozen scope API contract (see scope/README.md).

export type SubrunStatus = "present" | "in_flight";

export interface Subrun {
    subrun_id: string;
    status: SubrunStatus;
    n_batches: number;
    progress: number;
}

export interface Site {
    site: string;
    n_components: number;
    n_labeled: number;
    subruns: Subrun[];
}

export interface Run {
    run_id: string;
    sites: Site[];
}

export interface Catalog {
    runs: Run[];
}

export type SortKey = "density" | "max_act" | "unlabeled_first";

export interface ComponentRow {
    idx: number;
    density: number;
    max_act: number;
    label: string | null;
}

export interface ComponentListing {
    total: number;
    page: number;
    items: ComponentRow[];
}

export interface ComponentLabel {
    text: string;
    model: string;
    cost_usd: number;
    created_at: string;
}

export interface ActivationExample {
    tokens: string[];
    acts: number[];
    cis: number[];
    max_act: number;
}

export interface CurvePoint {
    rank: number;
    idx: number;
    mean_ci: number;
}

export interface SiteCurve {
    total: number;
    points: CurvePoint[];
}

export interface ComponentDetail {
    idx: number;
    rank: number;
    prev_idx: number | null;
    next_idx: number | null;
    density: number;
    max_act: number;
    mean_ci: number;
    label: ComponentLabel | null;
    input_pmi: [string, number][];
    output_pmi: [string, number][];
    examples: ActivationExample[];
}
