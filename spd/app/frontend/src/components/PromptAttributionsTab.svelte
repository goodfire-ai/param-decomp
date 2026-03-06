<script lang="ts">
    import * as api from "../lib/api";
    import ProbColoredTokens from "./ProbColoredTokens.svelte";
    import {
        type GraphData,
        type HoveredNode,
        type PinnedNode,
        type PromptPreview,
        type EdgeData,
        getActiveEdges,
    } from "../lib/promptAttributionsTypes";
    import { displaySettings } from "../lib/displaySettings.svelte";
    import ComponentNodeCard from "./prompt-attr/ComponentNodeCard.svelte";
    import ComputeProgressOverlay from "./prompt-attr/ComputeProgressOverlay.svelte";
    import GraphTabs from "./prompt-attr/GraphTabs.svelte";
    import InterventionsView from "./prompt-attr/InterventionsView.svelte";
    import OptimizationParams from "./prompt-attr/OptimizationParams.svelte";
    import OptimizationSettings from "./prompt-attr/OptimizationSettings.svelte";
    import PromptTabs from "./prompt-attr/PromptTabs.svelte";
    import StagedNodesPanel from "./prompt-attr/StagedNodesPanel.svelte";
    import {
        defaultDraftState,
        defaultOptimizeConfig,
        isOptimizeConfigValid,
        validateOptimizeConfig,
        type DraftState,
        type GraphComputeState,
        type OptimizeConfigDraft,
        type PromptCard,
        type StoredGraph,
        type TabViewState,
        type ViewSettings,
    } from "./prompt-attr/types";
    import { buildInterventionState, type BakedRun, type InterventionState } from "../lib/interventionTypes";
    import { SvelteSet } from "svelte/reactivity";
    import ViewControls from "./prompt-attr/ViewControls.svelte";
    import ViewTabs from "./prompt-attr/ViewTabs.svelte";
    import PromptAttributionsGraph from "./PromptAttributionsGraph.svelte";

    /** Generate a display label for a graph based on its type */
    function getGraphLabel(data: GraphData): string {
        switch (data.graphType) {
            case "standard":
                return "Standard";
            case "optimized":
                return data.optimization ? `Optimized (${data.optimization.steps} steps)` : "Optimized";
            case "manual":
                return `Manual (${Object.keys(data.nodeCiVals).length} components)`;
        }
    }

    type Props = {
        prompts: PromptPreview[];
    };

    let { prompts: initialPrompts }: Props = $props();

    // Local copy of prompts that can be modified (e.g. after deletion)
    let prompts = $state(initialPrompts);

    // Prompt cards state
    let promptCards = $state<PromptCard[]>([]);

    // Prompt deletion confirmation
    let confirmingDeleteId = $state<number | null>(null);

    // Tab view state - discriminated union makes invalid states unrepresentable
    let tabView = $state<TabViewState>({ view: "draft", draft: defaultDraftState() });

    // Timer for debounced tokenization (not part of view state since it's internal)
    let tokenizeDebounceTimer: ReturnType<typeof setTimeout> | null = null;

    // Graph computation state
    let graphCompute = $state<GraphComputeState>({ status: "idle" });

    // Intervention loading state
    let runningIntervention = $state(false);
    let generatingSubgraph = $state(false);

    // Refetching state (for CI threshold/normalize changes) - tracks which graph is being refetched
    let refetchingGraphId = $state<number | null>(null);

    // Intervention state - transient UI state for intervention versions, keyed by graph ID
    let interventionStates = $state<Record<number, InterventionState>>({});

    // Helper to get or create intervention state for a graph
    function getInterventionState(graphId: number, graph: StoredGraph): InterventionState {
        if (!interventionStates[graphId]) {
            interventionStates[graphId] = buildInterventionState(graph.interventionRuns);
        }
        return interventionStates[graphId];
    }

    // Derived: intervention state for the active graph
    const activeInterventionState = $derived.by(() => {
        if (!activeGraph) return null;
        return interventionStates[activeGraph.id] ?? null;
    });

    // Default view settings for new graphs
    const defaultViewSettings: ViewSettings = {
        topK: 1000,
        componentGap: 8,
        layerGap: 40,
        normalizeEdges: "layer",
        ciThreshold: 0,
    };

    // Edge count is derived from the graph rendering, not stored per-graph
    let filteredEdgeCount = $state<number | null>(null);
    let hideUnpinnedEdges = $state(false);
    let hideNodeCard = $state(false);

    // Hovered node from graph (lifted up for side panel rendering)
    let hoveredNode = $state<HoveredNode | null>(null);

    // Resizable split for node detail panel
    let detailPanelWidth = $state(500);

    function handleResizeStart(e: MouseEvent) {
        e.preventDefault();
        const startX = e.clientX;
        const startWidth = detailPanelWidth;

        function onMouseMove(e: MouseEvent) {
            detailPanelWidth = Math.max(250, Math.min(900, startWidth + (startX - e.clientX)));
        }

        function onMouseUp() {
            window.removeEventListener("mousemove", onMouseMove);
            window.removeEventListener("mouseup", onMouseUp);
        }

        window.addEventListener("mousemove", onMouseMove);
        window.addEventListener("mouseup", onMouseUp);
    }

    // Sticky: last hovered component persists in the side panel
    type ComponentNodeInfo = {
        layer: string;
        cIdx: number;
        seqIdx: number;
        ciVal: number | null;
        subcompAct: number | null;
        token: string;
    };
    let stickyComponentNode = $state<ComponentNodeInfo | null>(null);

    $effect(() => {
        if (!hoveredNode || !activeGraph) return;
        if (hoveredNode.layer === "embed" || hoveredNode.layer === "output") return;
        const key = `${hoveredNode.layer}:${hoveredNode.seqIdx}:${hoveredNode.cIdx}`;
        stickyComponentNode = {
            layer: hoveredNode.layer,
            cIdx: hoveredNode.cIdx,
            seqIdx: hoveredNode.seqIdx,
            ciVal: activeGraph.data.nodeCiVals[key] ?? null,
            subcompAct: activeGraph.data.nodeSubcompActs[key] ?? null,
            token: activeCard!.tokens[hoveredNode.seqIdx],
        };
    });

    // Pinned nodes for attributions graph
    let pinnedNodes = $state<PinnedNode[]>([]);

    function handlePinnedNodesChange(nodes: PinnedNode[]) {
        pinnedNodes = nodes;
    }

    // NOTE: Token selection is handled entirely by TokenDropdown, which provides the exact
    // token ID. We don't re-tokenize text because the same string (e.g. "art") can map to
    // different tokens depending on context (continuation "##art" vs word-initial " art").
    // The dropdown's onSelect callback sets labelTokenId directly.

    // Derived state from tabView
    const activeCardId = $derived(tabView.view === "card" ? tabView.cardId : null);
    const activeCard = $derived(
        activeCardId !== null ? (promptCards.find((c) => c.id === activeCardId) ?? null) : null,
    );
    const activeGraph = $derived.by(() => {
        if (!activeCard) return null;
        return activeCard.graphs.find((g) => g.id === activeCard.activeGraphId) ?? null;
    });

    // Active edge variant (derived from display settings and graph data availability)
    const activeEdgeSet = $derived(activeGraph ? getActiveEdges(activeGraph.data, displaySettings.edgeVariant) : null);
    const activeEdgesBySource = $derived(activeEdgeSet?.bySource ?? new Map<string, EdgeData[]>());
    const activeEdgesByTarget = $derived(activeEdgeSet?.byTarget ?? new Map<string, EdgeData[]>());

    // Check if a standard graph already exists for the active card
    const hasStandardGraph = $derived(activeCard?.graphs.some((g) => g.data.graphType === "standard") ?? false);

    // Check if compute button should be enabled (config must be valid for optimized graphs)
    const canCompute = $derived.by(() => {
        if (!activeCard) return false;
        const needsOptimized = hasStandardGraph || activeCard.useOptimized;
        if (!needsOptimized) return true; // Standard graph mode - always valid
        return isOptimizeConfigValid(activeCard.newGraphConfig);
    });

    // Helper to update draft state (only valid when in draft view)
    function updateDraft(partial: Partial<DraftState>) {
        if (tabView.view !== "draft") return;
        tabView = { view: "draft", draft: { ...tabView.draft, ...partial } };
    }

    async function addPromptCard(
        promptId: number,
        tokens: string[],
        tokenIds: number[],
        nextTokenProbs: (number | null)[],
        isCustom: boolean,
    ) {
        tabView = { view: "loading" };
        try {
            await addPromptCardInner(promptId, tokens, tokenIds, nextTokenProbs, isCustom);
            tabView = { view: "card", cardId: promptId };
        } catch (error) {
            tabView = { view: "error", error: String(error) };
        }
    }

    async function addPromptCardInner(
        promptId: number,
        tokens: string[],
        tokenIds: number[],
        nextTokenProbs: (number | null)[],
        isCustom: boolean,
    ) {
        // Fetch stored graphs for this prompt
        const storedGraphs = await api.getGraphs(
            promptId,
            defaultViewSettings.normalizeEdges,
            defaultViewSettings.ciThreshold,
        );
        const graphs: StoredGraph[] = await Promise.all(
            storedGraphs.map(async (data) => {
                // Load intervention runs for this graph
                const runs = await api.getInterventionRuns(data.id);

                const sg: StoredGraph = {
                    id: data.id,
                    label: getGraphLabel(data),
                    data,
                    viewSettings: { ...defaultViewSettings },
                    interventionRuns: runs,
                };

                // Initialize intervention state for this graph
                getInterventionState(data.id, sg);

                return sg;
            }),
        );

        const newCard: PromptCard = {
            id: promptId,
            tokens,
            tokenIds,
            nextTokenProbs,
            isCustom,
            graphs,
            activeGraphId: graphs.length > 0 ? graphs[0].id : null,
            activeView: "graph",
            newGraphConfig: defaultOptimizeConfig(tokens.length),
            useOptimized: false,
        };
        promptCards = [...promptCards, newCard];
    }

    async function handleDeletePrompt(promptId: number) {
        await api.deletePrompt(promptId);
        prompts = prompts.filter((p) => p.id !== promptId);
        promptCards = promptCards.filter((c) => c.id !== promptId);
        confirmingDeleteId = null;
        // If we were viewing the deleted prompt, go back to draft
        if (activeCardId === promptId) {
            if (promptCards.length > 0) {
                tabView = { view: "card", cardId: promptCards[promptCards.length - 1].id };
            } else {
                tabView = { view: "draft", draft: defaultDraftState() };
            }
        }
    }

    function handleSelectPrompt(prompt: PromptPreview) {
        // If prompt is already open as a card, just focus it
        const existingCard = promptCards.find((c) => c.id === prompt.id);
        if (existingCard) {
            tabView = { view: "card", cardId: prompt.id };
            return;
        }
        addPromptCard(prompt.id, prompt.tokens, prompt.token_ids, prompt.next_token_probs, false);
    }

    // Create a new prompt from draft text and add as card
    async function handleAddFromDraft() {
        if (tabView.view !== "draft") return;
        const draftText = tabView.draft.text;
        if (!draftText.trim()) return;

        updateDraft({ isAdding: true });
        try {
            const prompt = await api.createCustomPrompt(draftText);
            // If prompt already exists (returned existing ID), just focus it
            const existingCard = promptCards.find((c) => c.id === prompt.id);
            if (existingCard) {
                tabView = { view: "card", cardId: prompt.id };
                return;
            }
            await addPromptCard(prompt.id, prompt.tokens, prompt.token_ids, prompt.next_token_probs, true);
            // addPromptCard sets tabView to card on success
        } catch (error) {
            // If addPromptCard failed, we're already in error state
            // If createCustomPrompt failed, stay in draft and show error
            if (tabView.view === "draft") {
                updateDraft({ isAdding: false });
            }
            throw error;
        }
    }

    function handleStartNewDraft() {
        tabView = { view: "draft", draft: defaultDraftState() };
    }

    function handleDraftTextChange(text: string) {
        if (tabView.view !== "draft") return;
        updateDraft({ text });

        // Debounced tokenization for preview
        if (tokenizeDebounceTimer) clearTimeout(tokenizeDebounceTimer);
        if (!text.trim()) {
            updateDraft({ tokenPreview: { status: "uninitialized" } });
            return;
        }
        updateDraft({ tokenPreview: { status: "loading" } });
        tokenizeDebounceTimer = setTimeout(async () => {
            try {
                const result = await api.tokenizeText(text);
                updateDraft({
                    tokenPreview: {
                        status: "loaded",
                        data: { tokens: result.tokens, next_token_probs: result.next_token_probs },
                    },
                });
            } catch (e) {
                updateDraft({
                    tokenPreview: {
                        status: "error",
                        error: e instanceof Error ? e.message : String(e),
                    },
                });
            }
        }, 150);
    }

    function handleDraftKeydown(e: KeyboardEvent) {
        if (e.key === "Enter" && e.metaKey) {
            e.preventDefault();
            handleAddFromDraft();
        }
    }

    function handleCloseCard(cardId: number) {
        promptCards = promptCards.filter((c) => c.id !== cardId);
        if (activeCardId === cardId) {
            // Switch to another card or back to draft
            if (promptCards.length > 0) {
                tabView = { view: "card", cardId: promptCards[promptCards.length - 1].id };
            } else {
                tabView = { view: "draft", draft: defaultDraftState() };
            }
        }
    }

    function handleSelectCard(cardId: number) {
        tabView = { view: "card", cardId };
    }

    function handleDismissError() {
        // Go back to draft on error dismissal
        tabView = { view: "draft", draft: defaultDraftState() };
    }

    function handleSelectGraph(graphId: number) {
        if (!activeCard) return;
        promptCards = promptCards.map((card) =>
            card.id === activeCard.id ? { ...card, activeGraphId: graphId } : card,
        );
    }

    function handleCloseGraph(graphId: number) {
        if (!activeCard) return;
        promptCards = promptCards.map((card) => {
            if (card.id !== activeCard.id) return card;
            const newGraphs = card.graphs.filter((g) => g.id !== graphId);
            return {
                ...card,
                graphs: newGraphs,
                activeGraphId:
                    card.activeGraphId === graphId
                        ? newGraphs.length > 0
                            ? newGraphs[newGraphs.length - 1].id
                            : null
                        : card.activeGraphId,
            };
        });
    }

    function handleUseOptimizedChange(useOptimized: boolean) {
        if (!activeCard) return;
        promptCards = promptCards.map((card) => (card.id === activeCard.id ? { ...card, useOptimized } : card));
    }

    function handleOptimizeConfigChange(newConfig: OptimizeConfigDraft) {
        if (!activeCard) return;
        promptCards = promptCards.map((card) =>
            card.id === activeCard.id ? { ...card, newGraphConfig: newConfig } : card,
        );
    }

    function handleEnterNewGraphMode() {
        if (!activeCard) return;
        promptCards = promptCards.map((card) => (card.id === activeCard.id ? { ...card, activeGraphId: null } : card));
    }

    // Switch between graph and interventions view
    function handleViewChange(view: "graph" | "interventions") {
        if (!activeCard) return;
        promptCards = promptCards.map((card) => (card.id === activeCard.id ? { ...card, activeView: view } : card));
    }

    // Update draft selection for the active graph
    function handleDraftSelectionChange(selection: Set<string>) {
        if (!activeGraph) return;
        const state = interventionStates[activeGraph.id];
        if (!state) throw new Error("No intervention state for active graph");
        const activeRun = state.runs[state.activeIndex];
        if (activeRun.kind !== "draft") throw new Error("Can only change selection on draft runs");
        activeRun.selectedNodes = new SvelteSet(selection);
        interventionStates = { ...interventionStates };
    }

    // Forward a draft run: call API, replace draft with baked
    async function handleForwardDraft(advPgd: { n_steps: number; step_size: number }) {
        if (!activeCard || !activeGraph) return;
        const state = interventionStates[activeGraph.id];
        if (!state) throw new Error("No intervention state for active graph");
        const activeRun = state.runs[state.activeIndex];
        if (activeRun.kind !== "draft") throw new Error("Can only forward draft runs");

        runningIntervention = true;
        try {
            const selectedNodes = Array.from(activeRun.selectedNodes);
            const baseRun = state.runs[0];
            if (baseRun.kind !== "baked") throw new Error("First run must be baked base run");
            const nodesToAblate = Array.from(baseRun.selectedNodes).filter((n) => !activeRun.selectedNodes.has(n));

            const run = await api.runAndSaveIntervention({
                graph_id: activeGraph.id,
                selected_nodes: selectedNodes,
                nodes_to_ablate: nodesToAblate.length > 0 ? nodesToAblate : undefined,
                top_k: 10,
                adv_pgd: advPgd,
            });

            // Replace the draft with a baked run
            const baked: BakedRun = {
                kind: "baked",
                id: run.id,
                selectedNodes: new Set(run.selected_nodes),
                result: run.result,
                createdAt: run.created_at,
            };
            state.runs[state.activeIndex] = baked;

            // Also update the persisted list on StoredGraph
            promptCards = promptCards.map((card) => {
                if (card.id !== activeCard.id) return card;
                return {
                    ...card,
                    graphs: card.graphs.map((g) =>
                        g.id === activeGraph.id ? { ...g, interventionRuns: [...g.interventionRuns, run] } : g,
                    ),
                };
            });

            interventionStates = { ...interventionStates };
        } finally {
            runningIntervention = false;
        }
    }

    // Select a version by index
    function handleSelectVersion(index: number) {
        if (!activeGraph) return;
        const state = interventionStates[activeGraph.id];
        if (!state) throw new Error("No intervention state for active graph");
        if (index < 0 || index >= state.runs.length) throw new Error(`Invalid version index: ${index}`);
        state.activeIndex = index;
        interventionStates = { ...interventionStates };
    }

    // Clone the active run into a new draft
    function handleCloneRun() {
        if (!activeGraph) return;
        const state = interventionStates[activeGraph.id];
        if (!state) throw new Error("No intervention state for active graph");

        const activeRun = state.runs[state.activeIndex];
        if (activeRun.kind !== "baked") throw new Error("Can only clone baked runs");

        const draft = {
            kind: "draft" as const,
            parentId: activeRun.id,
            selectedNodes: new SvelteSet(activeRun.selectedNodes),
        };
        state.runs.push(draft);
        state.activeIndex = state.runs.length - 1;
        interventionStates = { ...interventionStates };
    }

    // Delete a baked intervention run
    async function handleDeleteRun(runId: number) {
        if (!activeCard || !activeGraph) return;

        await api.deleteInterventionRun(runId);

        // Remove from persisted state
        promptCards = promptCards.map((card) => {
            if (card.id !== activeCard.id) return card;
            return {
                ...card,
                graphs: card.graphs.map((g) => {
                    if (g.id !== activeGraph.id) return g;
                    return { ...g, interventionRuns: g.interventionRuns.filter((r) => r.id !== runId) };
                }),
            };
        });

        // Remove from intervention state and fix activeIndex
        const state = interventionStates[activeGraph.id];
        if (state) {
            const idx = state.runs.findIndex((r) => r.kind === "baked" && r.id === runId);
            if (idx >= 0) {
                state.runs.splice(idx, 1);
                if (state.activeIndex >= state.runs.length) {
                    state.activeIndex = 0;
                } else if (state.activeIndex === idx) {
                    state.activeIndex = 0;
                } else if (state.activeIndex > idx) {
                    state.activeIndex--;
                }
            }
            interventionStates = { ...interventionStates };
        }
    }

    // Fork handlers commented out — functionality disabled for now
    // async function handleForkRun(runId: number, tokenReplacements: [number, number][]) { ... }
    // async function handleDeleteFork(forkId: number) { ... }

    async function handleGenerateGraphFromSelection() {
        if (!activeCard || !activeGraph) return;
        const state = interventionStates[activeGraph.id];
        if (!state) throw new Error("No intervention state");
        const activeRun = state.runs[state.activeIndex];
        if (activeRun.kind !== "baked" && activeRun.kind !== "draft")
            throw new Error("Can only generate subgraph from baked or draft runs");
        const selection = activeRun.selectedNodes;

        if (selection.size === 0) throw new Error("handleGenerateGraphFromSelection called with empty selection");

        const cardId = activeCard.id;
        const includedNodes = Array.from(selection);

        generatingSubgraph = true;
        graphCompute = {
            status: "computing",
            cardId,
            ciSnapshot: null,
            progress: {
                stages: [{ name: "Computing attribution graph from selection", progress: 0 }],
                currentStage: 0,
            },
        };

        try {
            const data = await api.computeGraphStream(
                {
                    promptId: cardId,
                    normalize: activeGraph.viewSettings.normalizeEdges,
                    ciThreshold: activeGraph.viewSettings.ciThreshold,
                    includedNodes,
                },
                (progress) => {
                    if (graphCompute.status === "computing") {
                        graphCompute.progress.stages[0].progress = progress.current / progress.total;
                    }
                },
            );

            const runs = await api.getInterventionRuns(data.id);
            const newGraph: StoredGraph = {
                id: data.id,
                label: getGraphLabel(data),
                data,
                viewSettings: { ...activeGraph.viewSettings },
                interventionRuns: runs,
            };
            getInterventionState(data.id, newGraph);

            promptCards = promptCards.map((card) => {
                if (card.id !== cardId) return card;

                // Check if graph with this ID already exists (get-or-create semantics from backend)
                const existingGraph = card.graphs.find((g) => g.id === data.id);
                if (existingGraph) {
                    // Graph already exists, just select it
                    return {
                        ...card,
                        activeGraphId: data.id,
                        activeView: "graph",
                    };
                }

                // Add new graph
                return {
                    ...card,
                    graphs: [...card.graphs, newGraph],
                    activeGraphId: data.id,
                    activeView: "graph",
                };
            });

            graphCompute = { status: "idle" };
        } catch (error) {
            graphCompute = { status: "idle" };
            alert(String(error));
        } finally {
            generatingSubgraph = false;
        }
    }

    async function computeGraphForCard() {
        if (!activeCard || !activeCard.tokenIds || graphCompute.status === "computing") return;

        const draftConfig = activeCard.newGraphConfig;
        // If a standard graph exists, always use optimized (no point computing another standard)
        const isOptimized = hasStandardGraph || activeCard.useOptimized;
        const cardId = activeCard.id;

        // Validate config (button should be disabled if invalid, so this is a safety check)
        const validConfig = validateOptimizeConfig(draftConfig);
        if (isOptimized && !validConfig) {
            throw new Error("Invalid config: CE loss requires a target token");
        }

        const initialProgress = isOptimized
            ? {
                  stages: [
                      { name: "Optimizing", progress: 0 },
                      { name: "Computing attribution graph", progress: 0 },
                  ],
                  currentStage: 0,
              }
            : {
                  stages: [{ name: "Computing attribution graph", progress: 0 }],
                  currentStage: 0,
              };

        graphCompute = { status: "computing", cardId, ciSnapshot: null, progress: initialProgress };

        try {
            let data: GraphData;

            if (isOptimized) {
                // validConfig is guaranteed non-null here due to early return above
                const optConfig = validConfig!;

                const params: api.ComputeGraphOptimizedParams = {
                    promptId: cardId,
                    normalize: defaultViewSettings.normalizeEdges,
                    impMinCoeff: optConfig.impMinCoeff,
                    steps: optConfig.steps,
                    pnorm: optConfig.pnorm,
                    beta: optConfig.beta,
                    ciThreshold: defaultViewSettings.ciThreshold,
                    maskType: optConfig.maskType,
                    lossType: optConfig.loss.type,
                    lossCoeff: optConfig.loss.coeff,
                    lossPosition: optConfig.loss.position,
                    labelToken:
                        optConfig.loss.type === "ce" || optConfig.loss.type === "logit"
                            ? optConfig.loss.labelTokenId
                            : undefined,
                    advPgdNSteps:
                        optConfig.advPgdNSteps !== null && optConfig.advPgdStepSize !== null
                            ? optConfig.advPgdNSteps
                            : undefined,
                    advPgdStepSize:
                        optConfig.advPgdNSteps !== null && optConfig.advPgdStepSize !== null
                            ? optConfig.advPgdStepSize
                            : undefined,
                };

                data = await api.computeGraphOptimizedStream(
                    params,
                    (progress) => {
                        if (graphCompute.status !== "computing") return;
                        if (progress.stage === "graph") {
                            graphCompute.progress.currentStage = 1;
                            graphCompute.progress.stages[1].progress = progress.current / progress.total;
                        } else {
                            graphCompute.progress.stages[0].progress = progress.current / progress.total;
                        }
                    },
                    (snapshot) => {
                        if (graphCompute.status !== "computing") return;
                        graphCompute.ciSnapshot = snapshot;
                    },
                );
            } else {
                const params: api.ComputeGraphParams = {
                    promptId: cardId,
                    normalize: defaultViewSettings.normalizeEdges,
                    ciThreshold: defaultViewSettings.ciThreshold,
                };
                data = await api.computeGraphStream(params, (progress) => {
                    if (graphCompute.status !== "computing") return;
                    graphCompute.progress.stages[0].progress = progress.current / progress.total;
                });
            }

            const runs = await api.getInterventionRuns(data.id);
            const newGraph: StoredGraph = {
                id: data.id,
                label: getGraphLabel(data),
                data,
                viewSettings: { ...defaultViewSettings },
                interventionRuns: runs,
            };
            getInterventionState(data.id, newGraph);

            promptCards = promptCards.map((card) => {
                if (card.id !== cardId) return card;

                // Check if graph with this ID already exists (defensive check)
                const existingGraph = card.graphs.find((g) => g.id === data.id);
                if (existingGraph) {
                    return { ...card, activeGraphId: data.id };
                }

                return {
                    ...card,
                    graphs: [...card.graphs, newGraph],
                    activeGraphId: data.id,
                };
            });

            graphCompute = { status: "idle" };
        } catch (error) {
            graphCompute = { status: "error", error: String(error) };
        }
    }

    async function computeBatchGraphsForCard(impMinCoeffs: number[]) {
        if (!activeCard || !activeCard.tokenIds || graphCompute.status === "computing") return;

        const draftConfig = activeCard.newGraphConfig;
        const cardId = activeCard.id;

        const validConfig = validateOptimizeConfig(draftConfig);
        if (!validConfig) {
            throw new Error("Invalid config: CE loss requires a target token");
        }

        graphCompute = {
            status: "computing",
            cardId,
            ciSnapshot: null,
            progress: {
                stages: [
                    { name: `Optimizing (${impMinCoeffs.length} coefficients)`, progress: 0 },
                    { name: "Computing attribution graphs", progress: 0 },
                ],
                currentStage: 0,
            },
        };

        try {
            const params: api.ComputeGraphOptimizedBatchParams = {
                promptId: cardId,
                normalize: defaultViewSettings.normalizeEdges,
                impMinCoeffs,
                steps: validConfig.steps,
                pnorm: validConfig.pnorm,
                beta: validConfig.beta,
                ciThreshold: defaultViewSettings.ciThreshold,
                maskType: validConfig.maskType,
                lossType: validConfig.loss.type,
                lossCoeff: validConfig.loss.coeff,
                lossPosition: validConfig.loss.position,
                labelToken:
                    validConfig.loss.type === "ce" || validConfig.loss.type === "logit"
                        ? validConfig.loss.labelTokenId
                        : undefined,
                advPgdNSteps:
                    validConfig.advPgdNSteps !== null && validConfig.advPgdStepSize !== null
                        ? validConfig.advPgdNSteps
                        : undefined,
                advPgdStepSize:
                    validConfig.advPgdNSteps !== null && validConfig.advPgdStepSize !== null
                        ? validConfig.advPgdStepSize
                        : undefined,
            };

            const graphDataList = await api.computeGraphOptimizedBatchStream(
                params,
                (progress) => {
                    if (graphCompute.status !== "computing") return;
                    if (progress.stage === "graph") {
                        graphCompute.progress.currentStage = 1;
                        graphCompute.progress.stages[1].progress = progress.current / progress.total;
                    } else {
                        graphCompute.progress.stages[0].progress = progress.current / progress.total;
                    }
                },
                (snapshot) => {
                    if (graphCompute.status !== "computing") return;
                    graphCompute.ciSnapshot = snapshot;
                },
            );

            const newGraphs: StoredGraph[] = [];
            for (const data of graphDataList) {
                const runs = await api.getInterventionRuns(data.id);
                const newGraph: StoredGraph = {
                    id: data.id,
                    label: getGraphLabel(data),
                    data,
                    viewSettings: { ...defaultViewSettings },
                    interventionRuns: runs,
                };
                getInterventionState(data.id, newGraph);
                newGraphs.push(newGraph);
            }

            if (newGraphs.length !== 5) throw new Error(`Expected 5 batch graphs, got ${newGraphs.length}`);

            promptCards = promptCards.map((card) => {
                if (card.id !== cardId) return card;
                const existingIds = new Set(card.graphs.map((g) => g.id));
                const toAdd = newGraphs.filter((g) => !existingIds.has(g.id));
                return {
                    ...card,
                    graphs: [...card.graphs, ...toAdd],
                    activeGraphId: newGraphs[2].id,
                };
            });

            graphCompute = { status: "idle" };
        } catch (error) {
            graphCompute = { status: "error", error: String(error) };
        }
    }

    // Refetch graph data when normalize or ciThreshold changes (these affect server-side filtering)
    async function refetchActiveGraphData() {
        if (!activeCard || !activeGraph) return;

        const { normalizeEdges, ciThreshold } = activeGraph.viewSettings;
        refetchingGraphId = activeGraph.id;
        try {
            const storedGraphs = await api.getGraphs(activeCard.id, normalizeEdges, ciThreshold);
            const matchingData = storedGraphs.find((g) => g.id === activeGraph.id);

            if (!matchingData) {
                throw new Error("Could not find matching graph data after refetch");
            }

            // Update graph data
            promptCards = promptCards.map((card) => {
                if (card.id !== activeCard.id) return card;
                return {
                    ...card,
                    graphs: card.graphs.map((g) => (g.id !== activeGraph.id ? g : { ...g, data: matchingData })),
                };
            });

            // Intervention state stays as-is — base run's selectedNodes are from the persisted run
        } finally {
            refetchingGraphId = null;
        }
    }

    function updateActiveGraphViewSettings(partial: Partial<ViewSettings>) {
        if (!activeCard || !activeGraph) return;

        promptCards = promptCards.map((card) => {
            if (card.id !== activeCard.id) return card;
            return {
                ...card,
                graphs: card.graphs.map((g) => {
                    if (g.id !== activeGraph.id) return g;
                    return {
                        ...g,
                        viewSettings: { ...g.viewSettings, ...partial },
                    };
                }),
            };
        });
    }

    async function handleNormalizeChange(value: api.NormalizeType) {
        updateActiveGraphViewSettings({ normalizeEdges: value });
        await refetchActiveGraphData();
    }

    async function handleCiThresholdChange(value: number) {
        updateActiveGraphViewSettings({ ciThreshold: value });
        await refetchActiveGraphData();
    }

    function handleTopKChange(value: number) {
        updateActiveGraphViewSettings({ topK: value });
    }

    function handleComponentGapChange(value: number) {
        updateActiveGraphViewSettings({ componentGap: value });
    }

    function handleLayerGapChange(value: number) {
        updateActiveGraphViewSettings({ layerGap: value });
    }
</script>

<div class="prompt-attributions-tab">
    <div class="main-content">
        <div class="graph-container">
            <div class="card-tabs-row">
                <PromptTabs
                    cards={promptCards}
                    {tabView}
                    onSelectCard={handleSelectCard}
                    onCloseCard={handleCloseCard}
                    onSelectDraft={handleStartNewDraft}
                    onAddClick={handleStartNewDraft}
                />
            </div>

            <div class="card-content">
                <div class="card-content-main">
                    {#if tabView.view === "draft"}
                        {@const draft = tabView.draft}
                        <!-- New prompt staging area -->
                        <div class="draft-staging">
                            <div class="draft-main">
                                <div class="draft-input-section">
                                    <label class="draft-label">Enter prompt text</label>
                                    <textarea
                                        class="draft-textarea"
                                        placeholder="Type your prompt here... (Cmd+Enter to add)"
                                        value={draft.text}
                                        oninput={(e) => handleDraftTextChange(e.currentTarget.value)}
                                        onkeydown={handleDraftKeydown}
                                        rows={2}
                                    ></textarea>
                                    {#if draft.tokenPreview.status === "loading"}
                                        <div class="token-preview-row loading">Tokenizing...</div>
                                    {:else if draft.tokenPreview.status === "error"}
                                        <div class="token-preview-row error">{draft.tokenPreview.error}</div>
                                    {:else if draft.tokenPreview.status === "loaded" && draft.tokenPreview.data.tokens.length > 0}
                                        {@const { tokens, next_token_probs } = draft.tokenPreview.data}
                                        <div class="token-preview-row">
                                            <ProbColoredTokens {tokens} nextTokenProbs={next_token_probs} />
                                            <span class="token-count">{tokens.length} tokens</span>
                                        </div>
                                    {/if}
                                    <button
                                        class="btn-add-prompt"
                                        onclick={handleAddFromDraft}
                                        disabled={!draft.text.trim() || draft.isAdding}
                                    >
                                        {draft.isAdding ? "Adding..." : "Add Prompt"}
                                    </button>
                                </div>

                                {#if prompts.length > 0}
                                    <div class="existing-prompts-section">
                                        <label class="draft-label">Or select existing ({prompts.length})</label>
                                        <div class="prompt-list">
                                            {#each prompts as prompt (prompt.id)}
                                                {#if confirmingDeleteId === prompt.id}
                                                    <div class="prompt-item confirm-delete">
                                                        <span class="confirm-text">Delete prompt #{prompt.id}?</span>
                                                        <button
                                                            class="confirm-yes"
                                                            onclick={() => handleDeletePrompt(prompt.id)}>Yes</button
                                                        >
                                                        <button
                                                            class="confirm-no"
                                                            onclick={() => (confirmingDeleteId = null)}>No</button
                                                        >
                                                    </div>
                                                {:else}
                                                    <div class="prompt-item-row">
                                                        <button
                                                            class="prompt-item"
                                                            onclick={() => handleSelectPrompt(prompt)}
                                                        >
                                                            <span class="prompt-id">#{prompt.id}</span>
                                                            <span class="prompt-text">{prompt.preview}</span>
                                                        </button>
                                                        <button
                                                            class="btn-delete-prompt"
                                                            title="Delete prompt"
                                                            onclick={() => (confirmingDeleteId = prompt.id)}
                                                            >&times;</button
                                                        >
                                                    </div>
                                                {/if}
                                            {/each}
                                        </div>
                                    </div>
                                {/if}
                            </div>
                        </div>
                    {:else if activeCard}
                        <!-- Level 1: Tokens -->
                        <div class="prompt-tokens">
                            <ProbColoredTokens tokens={activeCard.tokens} nextTokenProbs={activeCard.nextTokenProbs} />
                        </div>

                        <!-- Level 2: Graph tabs -->
                        <GraphTabs
                            graphs={activeCard.graphs}
                            activeGraphId={activeCard.activeGraphId}
                            onSelectGraph={handleSelectGraph}
                            onCloseGraph={handleCloseGraph}
                            onNewGraph={handleEnterNewGraphMode}
                        />

                        {#if activeGraph}
                            <!-- Optimization params (if optimized graph) -->
                            {#if activeGraph.data.optimization}
                                <OptimizationParams
                                    optimization={activeGraph.data.optimization}
                                    tokens={activeCard.tokens}
                                />
                            {/if}

                            <!-- Level 3: View tabs -->
                            <div>
                                <ViewTabs
                                    activeView={activeCard.activeView}
                                    versionCount={activeInterventionState?.runs.length ?? 0}
                                    onViewChange={handleViewChange}
                                />

                                {#if activeCard.activeView === "graph"}
                                    <div class="graph-area">
                                        <ViewControls
                                            topK={activeGraph.viewSettings.topK}
                                            componentGap={activeGraph.viewSettings.componentGap}
                                            layerGap={activeGraph.viewSettings.layerGap}
                                            {filteredEdgeCount}
                                            normalizeEdges={activeGraph.viewSettings.normalizeEdges}
                                            ciThreshold={refetchingGraphId === activeGraph.id
                                                ? { status: "loading" }
                                                : { status: "loaded", data: activeGraph.viewSettings.ciThreshold }}
                                            {hideUnpinnedEdges}
                                            {hideNodeCard}
                                            onTopKChange={handleTopKChange}
                                            onComponentGapChange={handleComponentGapChange}
                                            onLayerGapChange={handleLayerGapChange}
                                            onNormalizeChange={handleNormalizeChange}
                                            onCiThresholdChange={handleCiThresholdChange}
                                            onHideUnpinnedEdgesChange={(v) => (hideUnpinnedEdges = v)}
                                            onHideNodeCardChange={(v) => (hideNodeCard = v)}
                                        />
                                        <div class="graph-info">
                                            <span class="l0-info"
                                                ><strong>L0:</strong>
                                                {activeGraph.data.l0_total.toFixed(0)} active at ci threshold {activeGraph
                                                    .viewSettings.ciThreshold}</span
                                            >
                                            {#if pinnedNodes.length > 0}
                                                <span class="pinned-count">{pinnedNodes.length} pinned</span>
                                            {/if}
                                        </div>
                                        {#key activeGraph.id}
                                            <PromptAttributionsGraph
                                                data={activeGraph.data}
                                                tokenIds={activeCard.tokenIds}
                                                topK={activeGraph.viewSettings.topK}
                                                componentGap={activeGraph.viewSettings.componentGap}
                                                layerGap={activeGraph.viewSettings.layerGap}
                                                {hideUnpinnedEdges}
                                                hideNodeCard={true}
                                                stagedNodes={pinnedNodes}
                                                onStagedNodesChange={handlePinnedNodesChange}
                                                onEdgeCountChange={(count) => (filteredEdgeCount = count)}
                                                onHoveredNodeChange={(node) => (hoveredNode = node)}
                                            />
                                        {/key}
                                    </div>
                                    <StagedNodesPanel
                                        stagedNodes={pinnedNodes}
                                        outputProbs={activeGraph.data.outputProbs}
                                        nodeCiVals={activeGraph.data.nodeCiVals}
                                        nodeSubcompActs={activeGraph.data.nodeSubcompActs}
                                        tokens={activeCard.tokens}
                                        edgesBySource={activeEdgesBySource}
                                        edgesByTarget={activeEdgesByTarget}
                                        onStagedNodesChange={handlePinnedNodesChange}
                                    />
                                {:else if activeInterventionState}
                                    <InterventionsView
                                        graph={activeGraph}
                                        interventionState={activeInterventionState}
                                        tokens={activeCard.tokens}
                                        topK={activeGraph.viewSettings.topK}
                                        componentGap={activeGraph.viewSettings.componentGap}
                                        layerGap={activeGraph.viewSettings.layerGap}
                                        normalizeEdges={activeGraph.viewSettings.normalizeEdges}
                                        ciThreshold={refetchingGraphId === activeGraph.id
                                            ? { status: "loading" }
                                            : { status: "loaded", data: activeGraph.viewSettings.ciThreshold }}
                                        {hideUnpinnedEdges}
                                        hideNodeCard={true}
                                        onTopKChange={handleTopKChange}
                                        onComponentGapChange={handleComponentGapChange}
                                        onLayerGapChange={handleLayerGapChange}
                                        onNormalizeChange={handleNormalizeChange}
                                        onCiThresholdChange={handleCiThresholdChange}
                                        onHideUnpinnedEdgesChange={(v) => (hideUnpinnedEdges = v)}
                                        onHideNodeCardChange={(v) => (hideNodeCard = v)}
                                        {runningIntervention}
                                        {generatingSubgraph}
                                        onSelectionChange={handleDraftSelectionChange}
                                        onForwardDraft={handleForwardDraft}
                                        onCloneRun={handleCloneRun}
                                        onSelectVersion={handleSelectVersion}
                                        onDeleteRun={handleDeleteRun}
                                        onGenerateGraphFromSelection={handleGenerateGraphFromSelection}
                                        onHoveredNodeChange={(node) => (hoveredNode = node)}
                                    />
                                {/if}
                            </div>
                        {:else}
                            <!-- No graph yet -->
                            {#if graphCompute.status === "error"}
                                <div class="error-banner">
                                    {graphCompute.error}
                                    <button onclick={() => (graphCompute = { status: "idle" })}>Dismiss</button>
                                    <button onclick={() => computeGraphForCard()} disabled={!canCompute}>Retry</button>
                                </div>
                            {/if}

                            <div
                                class="graph-area"
                                class:loading={graphCompute.status === "computing" &&
                                    graphCompute.cardId === activeCard.id}
                            >
                                {#if graphCompute.status === "computing" && graphCompute.cardId === activeCard.id}
                                    <ComputeProgressOverlay
                                        state={graphCompute.progress}
                                        ciSnapshot={graphCompute.ciSnapshot}
                                    />
                                {:else}
                                    <div class="empty-state">
                                        <div class="compute-controls">
                                            {#if !hasStandardGraph}
                                                <label class="optimize-checkbox">
                                                    <input
                                                        type="checkbox"
                                                        checked={activeCard.useOptimized}
                                                        onchange={(e) =>
                                                            handleUseOptimizedChange(e.currentTarget.checked)}
                                                    />
                                                    <span>Optimize</span>
                                                </label>
                                            {/if}
                                            {#if hasStandardGraph || activeCard.useOptimized}
                                                <OptimizationSettings
                                                    config={activeCard.newGraphConfig}
                                                    tokens={activeCard.tokens}
                                                    nextTokenProbs={activeCard.nextTokenProbs}
                                                    onChange={handleOptimizeConfigChange}
                                                    cardId={activeCard.id}
                                                />
                                            {/if}
                                            <div class="compute-buttons">
                                                <button
                                                    class="btn-compute-center"
                                                    onclick={() => computeGraphForCard()}
                                                    disabled={!canCompute}
                                                >
                                                    Compute
                                                </button>
                                                {#if hasStandardGraph || activeCard.useOptimized}
                                                    <button
                                                        class="btn-compute-batch"
                                                        onclick={() => {
                                                            const base = activeCard.newGraphConfig.impMinCoeff;
                                                            const coeffs = [
                                                                base * 0.1,
                                                                base * 0.3,
                                                                base,
                                                                base * 3,
                                                                base * 10,
                                                            ];
                                                            computeBatchGraphsForCard(coeffs);
                                                        }}
                                                        disabled={!canCompute}
                                                        title="Compute 5 graphs at 0.1x, 0.3x, 1x, 3x, 10x of current sparsity coefficient"
                                                    >
                                                        Batch (5x)
                                                    </button>
                                                {/if}
                                            </div>
                                        </div>
                                    </div>
                                {/if}
                            </div>
                        {/if}
                    {:else if tabView.view === "loading"}
                        <div class="empty-state">
                            <p>Loading prompt...</p>
                        </div>
                    {:else if tabView.view === "error"}
                        <div class="empty-state">
                            <p class="error-text">Error loading prompt: {tabView.error}</p>
                            <button onclick={handleDismissError}>Dismiss</button>
                        </div>
                    {/if}
                </div>

                {#if !hideNodeCard && stickyComponentNode && activeGraph}
                    <!-- svelte-ignore a11y_no_static_element_interactions -->
                    <div class="resize-handle" onmousedown={handleResizeStart}></div>
                    <div class="node-detail-panel" style:width="{detailPanelWidth}px">
                        {#key `${stickyComponentNode.layer}:${stickyComponentNode.cIdx}`}
                            <ComponentNodeCard
                                layer={stickyComponentNode.layer}
                                cIdx={stickyComponentNode.cIdx}
                                seqIdx={stickyComponentNode.seqIdx}
                                ciVal={stickyComponentNode.ciVal}
                                subcompAct={stickyComponentNode.subcompAct}
                                token={stickyComponentNode.token}
                                edgesBySource={activeEdgesBySource}
                                edgesByTarget={activeEdgesByTarget}
                                tokens={activeCard?.tokens ?? []}
                                outputProbs={activeGraph.data.outputProbs}
                                onPinComponent={(layer, cIdx, seqIdx) => {
                                    handlePinnedNodesChange([
                                        ...pinnedNodes.filter(
                                            (p) => !(p.layer === layer && p.seqIdx === seqIdx && p.cIdx === cIdx),
                                        ),
                                        { layer, seqIdx, cIdx },
                                    ]);
                                }}
                            />
                        {/key}
                    </div>
                {/if}
            </div>
        </div>
    </div>
</div>

<style>
    .prompt-attributions-tab {
        display: flex;
        flex: 1;
        min-height: 0;
        background: var(--bg-base);
    }

    .main-content {
        flex: 1;
        gap: var(--space-4);
        display: flex;
        flex-direction: column;
        min-width: 0;
        padding: var(--space-6);
    }

    .graph-container {
        flex: 1;
        display: flex;
        flex-direction: column;
        min-height: 0;
    }

    .card-tabs-row {
        display: flex;
        align-items: center;
        margin-bottom: var(--space-2);
        background: var(--bg-elevated);
        position: relative;
    }

    .card-content {
        flex: 1;
        display: flex;
        min-height: 0;
        min-width: 0;
        padding: var(--space-4);
        border: 1px solid var(--border-default);
        background: var(--bg-inset);
    }

    .card-content-main {
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
        min-height: 0;
        min-width: 0;
        overflow: auto;
    }

    .resize-handle {
        width: 6px;
        cursor: col-resize;
        background: transparent;
        flex-shrink: 0;
        position: relative;
    }

    .resize-handle:hover,
    .resize-handle:active {
        background: var(--accent-primary-dim);
    }

    .node-detail-panel {
        flex-shrink: 0;
        overflow-y: auto;
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        padding: var(--space-3);
    }

    .prompt-tokens {
        display: flex;
        flex-wrap: wrap;
        gap: 1px;
        align-items: center;
    }

    .graph-info {
        display: flex;
        align-items: center;
        gap: var(--space-4);
        padding: var(--space-2) var(--space-3);
        background: var(--bg-surface);
        border-bottom: 1px solid var(--border-default);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-muted);
    }

    .graph-info strong {
        color: var(--text-secondary);
        font-weight: 500;
    }

    .graph-info .pinned-count {
        color: var(--accent-primary);
    }

    .graph-area {
        display: flex;
        flex-direction: column;
        position: relative;
        min-height: 400px;
        border: 1px solid var(--border-default);
        overflow: hidden;
    }

    .graph-area.loading {
        opacity: 0.5;
    }

    .empty-state {
        display: flex;
        flex: 1;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: var(--text-muted);
        text-align: center;
        padding: var(--space-4);
        font-family: var(--font-sans);
    }

    .empty-state p {
        margin: var(--space-1) 0;
        font-size: var(--text-base);
    }

    .empty-state .error-text {
        color: var(--status-negative-bright);
    }

    .btn-compute-center {
        padding: var(--space-2) var(--space-4);
        background: var(--bg-elevated);
        border: 1px dashed var(--accent-primary-dim);
        font-size: var(--text-base);
        font-family: var(--font-mono);
        font-weight: 500;
        color: var(--accent-primary);
        cursor: pointer;
    }

    .btn-compute-center:hover {
        background: var(--bg-inset);
        border-style: solid;
        border-color: var(--accent-primary);
    }

    .compute-buttons {
        display: flex;
        gap: var(--space-2);
        justify-content: center;
    }

    .btn-compute-batch {
        padding: var(--space-2) var(--space-3);
        background: var(--bg-elevated);
        border: 1px dashed var(--border-default);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-secondary);
        cursor: pointer;
    }

    .btn-compute-batch:hover {
        background: var(--bg-inset);
        border-style: solid;
        border-color: var(--accent-primary-dim);
        color: var(--accent-primary);
    }

    .compute-controls {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: var(--space-3);
    }

    .optimize-checkbox {
        display: flex;
        align-items: center;
        gap: var(--space-1);
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        cursor: pointer;
    }

    .optimize-checkbox:hover {
        color: var(--text-primary);
    }

    .optimize-checkbox input {
        cursor: pointer;
    }

    .error-banner {
        padding: var(--space-2) var(--space-3);
        background: var(--bg-surface);
        border: 1px solid var(--status-negative);
        border-radius: var(--radius-sm);
        display: flex;
        align-items: center;
        gap: var(--space-3);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--status-negative-bright);
    }

    .error-banner button {
        margin-left: auto;
        padding: var(--space-1) var(--space-2);
        background: var(--status-negative);
        color: white;
        border: none;
    }

    .error-banner button:hover {
        background: var(--status-negative-bright);
    }

    /* Draft staging area styles */
    .draft-staging {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: var(--space-6);
    }

    .draft-main {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: var(--space-6);
        max-width: 900px;
        width: 100%;
        align-items: start;
    }

    .draft-input-section {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .draft-label {
        font-size: var(--text-sm);
        font-weight: 500;
        color: var(--text-secondary);
    }

    .draft-textarea {
        width: 100%;
        padding: var(--space-3);
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        resize: vertical;
        min-height: 120px;
    }

    .draft-textarea:focus {
        outline: none;
        border-color: var(--accent-primary);
    }

    .draft-textarea::placeholder {
        color: var(--text-muted);
    }

    .btn-add-prompt {
        align-self: flex-start;
        padding: var(--space-1) var(--space-3);
        background: var(--accent-primary);
        border: none;
        color: white;
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        font-weight: 500;
        cursor: pointer;
    }

    .btn-add-prompt:hover:not(:disabled) {
        background: var(--accent-primary-bright);
    }

    .btn-add-prompt:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    .token-preview-row {
        display: flex;
        flex-wrap: wrap;
        gap: 1px;
        align-items: center;
    }

    .token-preview-row.loading {
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        color: var(--text-muted);
    }

    .token-preview-row.error {
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        color: var(--status-negative);
    }

    .token-preview-row .token-count {
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        color: var(--text-muted);
        margin-left: var(--space-2);
    }

    .existing-prompts-section {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .prompt-list {
        display: flex;
        flex-direction: column;
        max-height: 400px;
        overflow-y: auto;
        background: var(--bg-inset);
        border: 1px solid var(--border-default);
    }

    .prompt-item-row {
        display: flex;
        align-items: stretch;
        border-bottom: 1px solid var(--border-subtle);
    }

    .prompt-item-row:last-child {
        border-bottom: none;
    }

    .prompt-item {
        flex: 1;
        padding: var(--space-2) var(--space-3);
        background: transparent;
        border: none;
        cursor: pointer;
        text-align: left;
        display: flex;
        gap: var(--space-2);
        align-items: baseline;
        color: var(--text-primary);
        min-width: 0;
    }

    .prompt-item:hover {
        background: var(--bg-surface);
    }

    .btn-delete-prompt {
        padding: 0 var(--space-2);
        background: transparent;
        border: none;
        color: var(--text-muted);
        font-size: var(--text-base);
        cursor: pointer;
        flex-shrink: 0;
    }

    .btn-delete-prompt:hover {
        color: var(--status-negative-bright);
        background: var(--bg-surface);
    }

    .confirm-delete {
        border-bottom: 1px solid var(--border-subtle);
        padding: var(--space-2) var(--space-3);
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .confirm-delete:last-child {
        border-bottom: none;
    }

    .confirm-text {
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-secondary);
        flex: 1;
    }

    .confirm-yes,
    .confirm-no {
        padding: var(--space-1) var(--space-2);
        border: none;
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        cursor: pointer;
    }

    .confirm-yes {
        background: var(--status-negative);
        color: white;
    }

    .confirm-yes:hover {
        background: var(--status-negative-bright);
    }

    .confirm-no {
        background: var(--bg-elevated);
        color: var(--text-secondary);
        border: 1px solid var(--border-default);
    }

    .confirm-no:hover {
        background: var(--bg-surface);
    }

    .prompt-id {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-muted);
        flex-shrink: 0;
    }

    .prompt-text {
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        color: var(--text-primary);
    }
</style>
