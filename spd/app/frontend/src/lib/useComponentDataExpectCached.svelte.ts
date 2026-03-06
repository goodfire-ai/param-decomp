/**
 * Hook for lazily loading component data with small initial limits.
 *
 * Fetches activation contexts (10), correlations (10), and token stats (10)
 * in parallel for fast initial render, then background-fetches full activation
 * examples (200). Dataset attributions and interpretation detail are on-demand.
 */

import { getContext, untrack } from "svelte";
import type { Loadable } from ".";
import {
    ApiError,
    getActivationContextDetail,
    getComponentAttributions,
    getComponentCorrelations,
    getComponentTokenStats,
    getGraphInterpComponentDetail,
    getInterpretationDetail,
    requestComponentInterpretation,
} from "./api";
import type { AllMetricAttributions, GraphInterpComponentDetail, InterpretationDetail } from "./api";
import type {
    SubcomponentCorrelationsResponse,
    SubcomponentActivationContexts,
    TokenStatsResponse,
} from "./promptAttributionsTypes";
import { RUN_KEY, type InterpretationBackendState, type RunContext } from "./useRun.svelte";

const DATASET_ATTRIBUTIONS_TOP_K = 20;
/** Fetch more activation examples in background after initial cached load */
const ACTIVATION_EXAMPLES_FULL_LIMIT = 200;

export type { AllMetricAttributions as DatasetAttributions };

export type ComponentCoords = { layer: string; cIdx: number };

export function useComponentDataExpectCached() {
    const runState = getContext<RunContext>(RUN_KEY);

    let componentDetail = $state<Loadable<SubcomponentActivationContexts>>({ status: "uninitialized" });
    let correlations = $state<Loadable<SubcomponentCorrelationsResponse | null>>({ status: "uninitialized" });
    let tokenStats = $state<Loadable<TokenStatsResponse | null>>({ status: "uninitialized" });
    let datasetAttributions = $state<Loadable<AllMetricAttributions | null>>({ status: "uninitialized" });
    let interpretationDetail = $state<Loadable<InterpretationDetail | null>>({ status: "uninitialized" });
    let graphInterpDetail = $state<Loadable<GraphInterpComponentDetail | null>>({ status: "uninitialized" });

    let currentCoords = $state<ComponentCoords | null>(null);
    let requestId = 0;

    /** Fetch full activation examples in background (overwrites cached data when complete). */
    function startBackgroundFetch(
        layer: string,
        cIdx: number,
        cachedDetail: SubcomponentActivationContexts,
        isStale: () => boolean,
    ) {
        getActivationContextDetail(layer, cIdx, ACTIVATION_EXAMPLES_FULL_LIMIT)
            .then((data) => {
                if (isStale()) return;
                if (data.example_tokens.length > cachedDetail.example_tokens.length) {
                    componentDetail = { status: "loaded", data };
                }
            })
            .catch((error) => {
                if (isStale()) return;
                componentDetail = { status: "error", error };
            });
    }

    /** Start on-demand fetches (dataset attributions, interpretation detail). */
    function startOnDemandFetches(layer: string, cIdx: number, isStale: () => boolean) {
        // Skip fetch entirely if dataset attributions not available for this run
        if (runState.datasetAttributionsAvailable) {
            datasetAttributions = { status: "loading" };
            getComponentAttributions(layer, cIdx, DATASET_ATTRIBUTIONS_TOP_K)
                .then((data) => {
                    if (isStale()) return;
                    datasetAttributions = { status: "loaded", data };
                })
                .catch((error) => {
                    if (isStale()) return;
                    if (error instanceof ApiError && error.status === 404) {
                        datasetAttributions = { status: "loaded", data: null };
                    } else {
                        datasetAttributions = { status: "error", error };
                    }
                });
        } else {
            datasetAttributions = { status: "loaded", data: null };
        }

        const interpState = untrack(() => runState.getInterpretation(`${layer}:${cIdx}`));
        if (interpState.status === "loaded" && interpState.data.status !== "none") {
            interpretationDetail = { status: "loading" };
            getInterpretationDetail(layer, cIdx)
                .then((data) => {
                    if (isStale()) return;
                    interpretationDetail = { status: "loaded", data };
                })
                .catch((error) => {
                    if (isStale()) return;
                    interpretationDetail = { status: "error", error };
                });
        } else {
            interpretationDetail = { status: "loaded", data: null };
        }

        // Fetch graph interp detail
        if (runState.graphInterpAvailable) {
            graphInterpDetail = { status: "loading" };
            getGraphInterpComponentDetail(layer, cIdx)
                .then((data) => {
                    if (isStale()) return;
                    graphInterpDetail = { status: "loaded", data };
                })
                .catch((error) => {
                    if (isStale()) return;
                    if (error instanceof ApiError && error.status === 404) {
                        graphInterpDetail = { status: "loaded", data: null };
                    } else {
                        graphInterpDetail = { status: "error", error };
                    }
                });
        } else {
            graphInterpDetail = { status: "loaded", data: null };
        }
    }

    function load(layer: string, cIdx: number) {
        currentCoords = { layer, cIdx };
        const thisRequestId = ++requestId;

        const isStale = () => requestId !== thisRequestId;

        componentDetail = { status: "loading" };
        correlations = { status: "loading" };
        tokenStats = { status: "loading" };

        Promise.all([
            getActivationContextDetail(layer, cIdx, 10),
            getComponentCorrelations(layer, cIdx, 10).catch(() => null),
            getComponentTokenStats(layer, cIdx, 10).catch(() => null),
        ])
            .then(([detail, corr, stats]) => {
                if (isStale()) return;
                componentDetail = { status: "loaded", data: detail };
                correlations = { status: "loaded", data: corr };
                tokenStats = { status: "loaded", data: stats };
                startBackgroundFetch(layer, cIdx, detail, isStale);
            })
            .catch((error) => {
                if (isStale()) return;
                componentDetail = { status: "error", error };
                correlations = { status: "error", error };
                tokenStats = { status: "error", error };
            });

        startOnDemandFetches(layer, cIdx, isStale);
    }

    function reset() {
        requestId++;
        currentCoords = null;
        componentDetail = { status: "uninitialized" };
        correlations = { status: "uninitialized" };
        tokenStats = { status: "uninitialized" };
        datasetAttributions = { status: "uninitialized" };
        interpretationDetail = { status: "uninitialized" };
        graphInterpDetail = { status: "uninitialized" };
    }

    // Interpretation is derived from the global cache
    const interpretation = $derived.by((): Loadable<InterpretationBackendState> => {
        if (!currentCoords) return { status: "uninitialized" };
        return runState.getInterpretation(`${currentCoords.layer}:${currentCoords.cIdx}`);
    });

    async function generateInterpretation() {
        if (!currentCoords) return;

        const { layer, cIdx } = currentCoords;
        const componentKey = `${layer}:${cIdx}`;

        try {
            runState.setInterpretation(componentKey, { status: "generating" });
            const result = await requestComponentInterpretation(layer, cIdx);
            runState.setInterpretation(componentKey, { status: "generated", data: result });

            // Fetch the detail now that it exists
            try {
                const detail = await getInterpretationDetail(layer, cIdx);
                interpretationDetail = { status: "loaded", data: detail };
            } catch (detailError) {
                interpretationDetail = { status: "error", error: detailError };
            }
        } catch (e) {
            runState.setInterpretation(componentKey, {
                status: "generation-error",
                error: e instanceof Error ? e.message : String(e),
            });
        }
    }

    return {
        get componentDetail() {
            return componentDetail;
        },
        get correlations() {
            return correlations;
        },
        get tokenStats() {
            return tokenStats;
        },
        get datasetAttributions() {
            return datasetAttributions;
        },
        get interpretation() {
            return interpretation;
        },
        get interpretationDetail() {
            return interpretationDetail;
        },
        get graphInterpDetail() {
            return graphInterpDetail;
        },
        load,
        reset,
        generateInterpretation,
    };
}
