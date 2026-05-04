### WandB run links {toc: WandB run links}

<label id="app:wandb-links"/>

The decomposition studied throughout this paper is the run

- **Main VPD decomposition:** <a href="https://wandb.ai/goodfire/spd/runs/s-55ea3f9b" target="_blank">goodfire/spd/runs/s-55ea3f9b</a> (<a href="https://wandb.ai/goodfire/spd/runs/s-55ea3f9b/files/final_config.yaml" target="_blank">config</a>, <a href="https://wandb.ai/goodfire/spd/runs/s-55ea3f9b/files/model_400000.pt" target="_blank">checkpoint</a>, <a href="https://wandb.ai/goodfire/spd/runs/s-55ea3f9b" target="_blank">run logs</a>).

It decomposes the target model

- **Target language model:** <a href="https://wandb.ai/goodfire/spd/runs/t-9d2b8f02" target="_blank">goodfire/spd/runs/t-9d2b8f02</a> (<a href="https://wandb.ai/goodfire/spd/runs/t-9d2b8f02/files/final_config.yaml" target="_blank">config</a>, <a href="https://wandb.ai/goodfire/spd/runs/t-9d2b8f02/files/model_step_99999.pt" target="_blank">checkpoint</a>, <a href="https://wandb.ai/goodfire/spd/runs/t-9d2b8f02" target="_blank">run logs</a>).

All other runs reported in the paper are listed below by the figure or table they support. Sweeps link to the WandB project workspace; individual VPD runs link to the run page.

| Used in | What | WandB link |
|---|---|---|
| <ref>fig:pareto-mse</ref>; <ref>fig:splitting-heatmap</ref> | PLT/CLT, local-MSE objective, $k \in \{8, 16, 32, 64\}$ | <a href="https://wandb.ai/mats-sprint/pile_local_sweep_jose" target="_blank">dict_4k</a>, <a href="https://wandb.ai/mats-sprint/pile_local_sweep_jose_32k" target="_blank">dict_32k</a> |
| <ref>fig:pareto-e2e</ref> | PLT/CLT, end-to-end KL objective, $k \in \{8, 16, 32, 64\}$, three training modes (`cascading` = error-propagating, `parallel` = clean-input, `independent` = single-layer) | <a href="https://wandb.ai/mats-sprint/pile_e2e_sweep_jose" target="_blank">dict_4k</a>, <a href="https://wandb.ai/mats-sprint/pile_e2e_sweep_jose_32k" target="_blank">dict_32k</a> |
| <ref>tab:seed-mmcs</ref> — PLT/CLT seed runs | 5 seeds $\times$ {PLT local-MSE, PLT e2e-independent, CLT local-MSE, CLT e2e-parallel}, $k = 16$, 4k dict | <a href="https://wandb.ai/mats-sprint/pile_multiseed_jose2" target="_blank">pile_multiseed_jose2</a> |
| <ref>tab:seed-mmcs</ref> — VPD seed runs | 5 VPD seed runs (otherwise identical to the main decomposition) | <a href="https://wandb.ai/goodfire/spd?nw=n9l0amrrudc" target="_blank">VPD seeds workspace</a> |
| <ref>tab:seed-mmcs</ref> — hidden-activation aux-loss VPD | VPD trained with an auxiliary stochastic-forward-pass hidden-activation MSE loss | <a href="https://wandb.ai/goodfire/spd/runs/s-aa4fec0a" target="_blank">goodfire/spd/runs/s-aa4fec0a</a> |
| <ref>fig:feature_splitting</ref>; <ref>fig:splitting-heatmap</ref> | VPD capacity sweep ($0.5\times$, $1\times$, $2\times$, $4\times$ subcomponents); $1\times$ is the main run above. The other three runs use registry experiments `pile_4L_fs_C_0.5x`, `pile_4L_fs_C_2x`, `pile_4L_fs_C_4x` (configs in `spd/experiments/lm/`), logged to the `goodfire/spd` project with run-name prefixes `C_0.5x_`, `C_2x_`, `C_4x_` | _[VPD capacity sweep link TBD]_ |
| <ref>fig:adv-vs-no-adv</ref> | No-adversarial-loss control run; otherwise identical training configuration to the main decomposition | _[no-adv control run link TBD]_ |

All activation-based comparison runs target the same `t-9d2b8f02` model and use $\text{LR} = 3 \times 10^{-4}$, batch size $4096$, sequence length $512$, $500$M tokens of the Pile, with BatchTopK activation.
