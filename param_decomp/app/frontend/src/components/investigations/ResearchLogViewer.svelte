<script lang="ts">
    /**
     * Research log viewer that renders markdown with inline graph artifacts.
     *
     * Supports the ```spd:graph syntax:
     * ```spd:graph
     * artifact: graph_001
     * ```
     *
     * This block is replaced with an interactive ArtifactGraph component.
     */

    import { marked } from "marked";
    import type { GraphArtifact } from "../../lib/api/investigations";
    import ArtifactGraph from "./ArtifactGraph.svelte";

    type Props = {
        markdown: string;
        artifacts: Record<string, GraphArtifact>;
        artifactsLoading?: boolean;
    };

    let { markdown, artifacts, artifactsLoading = false }: Props = $props();

    // Content block type: either rendered markdown HTML or a graph artifact
    type ContentBlock = { type: "html"; content: string } | { type: "graph"; artifactId: string };

    // Parse markdown and extract spd:graph blocks
    const contentBlocks = $derived.by(() => {
        const blocks: ContentBlock[] = [];

        // Pattern to find ```spd:graph blocks with artifact reference
        const spdGraphPattern = /```spd:graph\s*\n\s*artifact:\s*(\S+)\s*\n```/g;

        // Split markdown by spd:graph blocks
        let lastIndex = 0;
        let match;

        while ((match = spdGraphPattern.exec(markdown)) !== null) {
            // Add markdown before this block
            if (match.index > lastIndex) {
                const mdContent = markdown.slice(lastIndex, match.index);
                if (mdContent.trim()) {
                    blocks.push({ type: "html", content: marked.parse(mdContent) as string });
                }
            }

            // Add graph block
            const artifactId = match[1];
            blocks.push({ type: "graph", artifactId });

            lastIndex = match.index + match[0].length;
        }

        // Add remaining markdown after last block
        if (lastIndex < markdown.length) {
            const mdContent = markdown.slice(lastIndex);
            if (mdContent.trim()) {
                blocks.push({ type: "html", content: marked.parse(mdContent) as string });
            }
        }

        // If no blocks parsed at all (no spd:graph), parse everything as markdown
        if (blocks.length === 0 && markdown.trim()) {
            blocks.push({ type: "html", content: marked.parse(markdown) as string });
        }

        return blocks;
    });
</script>

<div class="research-log-viewer">
    {#each contentBlocks as block, i (i)}
        {#if block.type === "html"}
            <!-- eslint-disable-next-line svelte/no-at-html-tags -->
            <div class="markdown-content">{@html block.content}</div>
        {:else if block.type === "graph"}
            {@const artifact = artifacts[block.artifactId]}
            {#if artifact}
                <ArtifactGraph data={artifact.data} caption={artifact.caption ?? undefined} />
            {:else if artifactsLoading}
                <div class="artifact-loading">
                    Loading graph: <code>{block.artifactId}</code>...
                </div>
            {:else}
                <div class="artifact-missing">
                    Graph artifact not found: <code>{block.artifactId}</code>
                </div>
            {/if}
        {/if}
    {/each}
</div>

<style>
    .research-log-viewer {
        font-size: var(--text-sm);
        color: var(--text-primary);
        line-height: 1.6;
    }

    .markdown-content :global(h1),
    .markdown-content :global(h2),
    .markdown-content :global(h3),
    .markdown-content :global(h4) {
        margin-top: var(--space-4);
        margin-bottom: var(--space-2);
        font-weight: 600;
        color: var(--text-primary);
    }

    .markdown-content :global(h1) {
        font-size: var(--text-xl);
        border-bottom: 1px solid var(--border-default);
        padding-bottom: var(--space-2);
    }

    .markdown-content :global(h2) {
        font-size: var(--text-lg);
    }

    .markdown-content :global(h3) {
        font-size: var(--text-base);
    }

    .markdown-content :global(p) {
        margin: var(--space-2) 0;
    }

    .markdown-content :global(code) {
        background: var(--bg-inset);
        padding: var(--space-0) var(--space-1);
        border-radius: var(--radius-sm);
        font-family: var(--font-mono);
        font-size: 0.9em;
    }

    .markdown-content :global(pre) {
        background: var(--bg-inset);
        padding: var(--space-3);
        border-radius: var(--radius-md);
        overflow-x: auto;
        margin: var(--space-3) 0;
    }

    .markdown-content :global(pre code) {
        background: none;
        padding: 0;
    }

    .markdown-content :global(ul),
    .markdown-content :global(ol) {
        margin: var(--space-2) 0;
        padding-left: var(--space-4);
    }

    .markdown-content :global(li) {
        margin: var(--space-1) 0;
    }

    .markdown-content :global(blockquote) {
        border-left: 3px solid var(--border-default);
        padding-left: var(--space-3);
        margin: var(--space-3) 0;
        color: var(--text-secondary);
    }

    .markdown-content :global(a) {
        color: var(--accent-primary);
        text-decoration: none;
    }

    .markdown-content :global(a:hover) {
        text-decoration: underline;
    }

    .markdown-content :global(table) {
        border-collapse: collapse;
        width: 100%;
        margin: var(--space-3) 0;
    }

    .markdown-content :global(th),
    .markdown-content :global(td) {
        border: 1px solid var(--border-default);
        padding: var(--space-2);
        text-align: left;
    }

    .markdown-content :global(th) {
        background: var(--bg-elevated);
        font-weight: 600;
    }

    .artifact-loading {
        padding: var(--space-3);
        background: var(--bg-inset);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        color: var(--text-muted);
        margin: var(--space-3) 0;
    }

    .artifact-loading code {
        background: var(--bg-elevated);
        padding: var(--space-0) var(--space-1);
        border-radius: var(--radius-sm);
    }

    .artifact-missing {
        padding: var(--space-3);
        background: var(--accent-red-dim);
        border: 1px solid var(--accent-red);
        border-radius: var(--radius-md);
        color: var(--accent-red);
        margin: var(--space-3) 0;
    }

    .artifact-missing code {
        background: rgba(255, 255, 255, 0.1);
        padding: var(--space-0) var(--space-1);
        border-radius: var(--radius-sm);
    }
</style>
