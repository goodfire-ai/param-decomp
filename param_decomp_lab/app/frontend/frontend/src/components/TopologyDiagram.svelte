<script lang="ts">
    import type { TopologyInfo } from "../lib/api/pretrainInfo";

    type Props = {
        topology: TopologyInfo;
    };

    let { topology }: Props = $props();
</script>

<div class="topology">
    <div class="layer special">embed</div>
    {#each topology.block_structure as block (block.index)}
        <div class="block">
            <span class="block-label">{block.index}</span>
            <div class="sublayers">
                <div class="sublayer attn">
                    <span class="sublayer-label">{block.attn_type === "fused" ? "attn_fused" : "attn"}</span>
                    <div class="projections">
                        {#each block.attn_projections as proj (proj)}
                            <span class="proj">{proj}</span>
                        {/each}
                    </div>
                </div>
                <div class="sublayer ffn">
                    <span class="sublayer-label">{block.ffn_type}</span>
                    <div class="projections">
                        {#each block.ffn_projections as proj (proj)}
                            <span class="proj">{proj}</span>
                        {/each}
                    </div>
                </div>
            </div>
        </div>
    {/each}
    <div class="layer special">output</div>
</div>

<style>
    .topology {
        display: flex;
        flex-direction: column;
        gap: 4px;
        font-family: var(--font-mono);
        font-size: 11px;
    }

    .layer.special {
        padding: 3px 8px;
        background: var(--bg-elevated, #1a1a2e);
        border: 1px solid var(--border-default);
        border-radius: 4px;
        color: var(--text-muted);
        text-align: center;
        width: fit-content;
        align-self: center;
    }

    .block {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 4px 0;
    }

    .block-label {
        color: var(--text-muted);
        min-width: 16px;
        text-align: right;
        flex-shrink: 0;
    }

    .sublayers {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
    }

    .sublayer {
        display: flex;
        align-items: center;
        gap: 4px;
    }

    .sublayer-label {
        color: var(--text-muted);
        font-size: 10px;
    }

    .projections {
        display: flex;
        gap: 2px;
    }

    .proj {
        padding: 1px 5px;
        border-radius: 3px;
        color: var(--text-primary);
        font-size: 10px;
        line-height: 1.4;
    }

    .attn .proj {
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    .ffn .proj {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
</style>
