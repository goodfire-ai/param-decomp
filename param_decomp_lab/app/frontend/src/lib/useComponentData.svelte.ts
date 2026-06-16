import { getContext, untrack } from "svelte";
import type { Loadable } from ".";
import {
    ApiError,
    getComponentCorrelations,
    getComponentTokenStats,
    getInterpretationDetail,
    requestComponentInterpretation,
} from "./api";
import type { InterpretationDetail } from "./api";
import type {
    SubcomponentCorrelationsResponse,
    SubcomponentActivationContexts,
    TokenStatsResponse,
} from "./promptAttributionsTypes";
import { RUN_KEY, type InterpretationBackendState, type RunContext } from "./useRun.svelte";

/** Correlations are paginated in the UI, so fetch more */
const CORRELATIONS_TOP_K = 100;
/** Token stats are paginated in the UI */
const TOKEN_STATS_TOP_K = 200;

export type ComponentCoords = { layer: string; cIdx: number };

/**
 * Hook for loading component data via network requests.
 *
 * Call `load(layer, cIdx)` explicitly when you want to fetch data.
 * Interpretation headline is derived from the global runState cache.
 * Interpretation detail (reasoning + prompt) is fetched on-demand.
 *
 * For graph tooltips (smaller initial limits + background fetch), use useComponentDataExpectCached.
 */
export function useComponentData() {
    const runState = getContext<RunContext>(RUN_KEY);

    let componentDetail = $state<Loadable<SubcomponentActivationContexts>>({ status: "uninitialized" });
    // null inside Loadable means "no data for this component" (404)
    let correlations = $state<Loadable<SubcomponentCorrelationsResponse | null>>({ status: "uninitialized" });
    let tokenStats = $state<Loadable<TokenStatsResponse | null>>({ status: "uninitialized" });

    let interpretationDetail = $state<Loadable<InterpretationDetail | null>>({ status: "uninitialized" });

    // Current coords being loaded/displayed (for interpretation lookup)
    let currentCoords = $state<ComponentCoords | null>(null);

    // Request counter for handling stale responses
    let requestId = 0;

    /**
     * Load all data for the given component.
     * Call this from event handlers or on mount.
     */
    function load(layer: string, cIdx: number) {
        currentCoords = { layer, cIdx };
        const thisRequestId = ++requestId;

        // Set loading states
        componentDetail = { status: "loading" };
        correlations = { status: "loading" };
        tokenStats = { status: "loading" };
        interpretationDetail = { status: "loading" };

        // Helper to check if this request is still current
        const isStale = () => requestId !== thisRequestId;

        // Fetch component detail (cached in runState after first call)
        runState
            .getActivationContextDetail(layer, cIdx)
            .then((data) => {
                if (isStale()) return;
                componentDetail = { status: "loaded", data };
            })
            .catch((error) => {
                if (isStale()) return;
                componentDetail = { status: "error", error };
            });

        // Fetch correlations (404 = no data for this component)
        getComponentCorrelations(layer, cIdx, CORRELATIONS_TOP_K)
            .then((data) => {
                if (isStale()) return;
                correlations = { status: "loaded", data };
            })
            .catch((error) => {
                if (isStale()) return;
                if (error instanceof ApiError && error.status === 404) {
                    correlations = { status: "loaded", data: null };
                } else {
                    correlations = { status: "error", error };
                }
            });

        // Fetch token stats (404 = no data for this component)
        getComponentTokenStats(layer, cIdx, TOKEN_STATS_TOP_K)
            .then((data) => {
                if (isStale()) return;
                tokenStats = { status: "loaded", data };
            })
            .catch((error) => {
                if (isStale()) return;
                if (error instanceof ApiError && error.status === 404) {
                    tokenStats = { status: "loaded", data: null };
                } else {
                    tokenStats = { status: "error", error };
                }
            });

        const interpState = untrack(() => runState.getInterpretation(`${layer}:${cIdx}`));
        if (interpState.status === "loaded" && interpState.data.status !== "none") {
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
    }

    /**
     * Reset all state to uninitialized.
     */
    function reset() {
        requestId++; // Invalidate any in-flight requests
        currentCoords = null;
        componentDetail = { status: "uninitialized" };
        correlations = { status: "uninitialized" };
        tokenStats = { status: "uninitialized" };
        interpretationDetail = { status: "uninitialized" };
    }

    // Interpretation is derived from the global cache - reactive to both coords and cache
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

            // Fetch the detail (reasoning + prompt) now that it exists
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
        get interpretation() {
            return interpretation;
        },
        get interpretationDetail() {
            return interpretationDetail;
        },
        load,
        reset,
        generateInterpretation,
    };
}
