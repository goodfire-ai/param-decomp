

# **Jose Target Model Training Runs**

All runs target jose's LlamaSimpleMLP (goodfire/spd/runs/t-9d2b8f02), trained on the Pile dataset (500M tokens), using BatchTopK activation function.

## **Local MSE Training**

Training mode: per-layer MSE reconstruction loss (no end-to-end KL).

### **4k dict ([pile\_local\_sweep\_jose](https://wandb.ai/mats-sprint/pile_local_sweep_jose))**

| Run | Type | Dict Size | Top-k | Status | Artifacts |
| ----- | ----- | ----- | ----- | ----- | ----- |
| [tc\_k8](https://wandb.ai/mats-sprint/pile_local_sweep_jose/runs/1fgdzkza) | Transcoder | 4096 | 8 | finished | 4 (per layer) |
| [tc\_k16](https://wandb.ai/mats-sprint/pile_local_sweep_jose/runs/4ziu27fn) | Transcoder | 4096 | 16 | finished | 4 (per layer) |
| [tc\_k32](https://wandb.ai/mats-sprint/pile_local_sweep_jose/runs/r7jmo7tn) | Transcoder | 4096 | 32 | finished | 4 (per layer) |
| [tc\_k64](https://wandb.ai/mats-sprint/pile_local_sweep_jose/runs/ms2gzfro) | Transcoder | 4096 | 64 | finished | 4 (per layer) |
| [clt\_k8](https://wandb.ai/mats-sprint/pile_local_sweep_jose/runs/kn2wny4z) | CLT | 4096 | 8 | finished | 1 |
| [clt\_k16](https://wandb.ai/mats-sprint/pile_local_sweep_jose/runs/77sgz1pe) | CLT | 4096 | 16 | finished | 1 |
| [clt\_k32](https://wandb.ai/mats-sprint/pile_local_sweep_jose/runs/hxyj1pdx) | CLT | 4096 | 32 | finished | 1 |
| [clt\_k64](https://wandb.ai/mats-sprint/pile_local_sweep_jose/runs/8g87bvon) | CLT | 4096 | 64 | finished | 1 |

### **32k dict ([pile\_local\_sweep\_jose\_32k](https://wandb.ai/mats-sprint/pile_local_sweep_jose_32k))**

| Run | Type | Dict Size | Top-k | Status | Artifacts |
| ----- | ----- | ----- | ----- | ----- | ----- |
| [tc\_k8](https://wandb.ai/mats-sprint/pile_local_sweep_jose_32k/runs/bu4on2g8) | Transcoder | 32768 | 8 | finished | 4 (per layer) |
| [tc\_k16](https://wandb.ai/mats-sprint/pile_local_sweep_jose_32k/runs/c4o8i98k) | Transcoder | 32768 | 16 | finished | 4 (per layer) |
| [tc\_k32](https://wandb.ai/mats-sprint/pile_local_sweep_jose_32k/runs/73cidp53) | Transcoder | 32768 | 32 | finished | 4 (per layer) |
| [tc\_k64](https://wandb.ai/mats-sprint/pile_local_sweep_jose_32k/runs/94vjrdni) | Transcoder | 32768 | 64 | running | 0 |
| [clt\_k8](https://wandb.ai/mats-sprint/pile_local_sweep_jose_32k/runs/ty1ofzrw) | CLT | 32768 | 8 | finished | 1 |
| [clt\_k16](https://wandb.ai/mats-sprint/pile_local_sweep_jose_32k/runs/j20m9hzr) | CLT | 32768 | 16 | finished | 1 |
| [clt\_k32](https://wandb.ai/mats-sprint/pile_local_sweep_jose_32k/runs/tzsnndn7) | CLT | 32768 | 32 | finished | 1 |
| [clt\_k64](https://wandb.ai/mats-sprint/pile_local_sweep_jose_32k/runs/ywqw69cj) | CLT | 32768 | 64 | finished | 1 |

## **End-to-End KL Training**

Training mode: end-to-end KL divergence on logits. Three TC modes (cascading, parallel, independent) and two CLT modes (cascading, parallel).

* cascading: each layer's reconstruction feeds into the next layer's encoder  
* parallel: all layers encode from clean activations, all MLPs patched simultaneously  
* independent: each layer gets its own KL loss (only its MLP replaced), sequential backward

### **4k dict ([pile\_e2e\_sweep\_jose](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose))**

| Run | Type | Mode | Dict Size | Top-k | Status | Artifacts |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| [tc\_cascading\_k8](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose/runs/wbz3ud8u) | Transcoder | cascading | 4096 | 8 | finished | 4 |
| [tc\_cascading\_k16](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose/runs/bq2n1t9m) | Transcoder | cascading | 4096 | 16 | finished | 4 |
| [tc\_cascading\_k32](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose/runs/7qblz2fn) | Transcoder | cascading | 4096 | 32 | finished | 4 |
| [tc\_cascading\_k64](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose/runs/v4gyqbrd) | Transcoder | cascading | 4096 | 64 | finished | 4 |
| [tc\_parallel\_k8](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose/runs/u8bz4xov) | Transcoder | parallel | 4096 | 8 | finished | 4 |
| [tc\_parallel\_k16](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose/runs/8sq5enhf) | Transcoder | parallel | 4096 | 16 | finished | 4 |
| [tc\_parallel\_k32](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose/runs/6d6yme3j) | Transcoder | parallel | 4096 | 32 | finished | 4 |
| [tc\_parallel\_k64](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose/runs/cgbuoclg) | Transcoder | parallel | 4096 | 64 | finished | 4 |
| [tc\_independent\_k8](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose/runs/4dl1b5zq) | Transcoder | independent | 4096 | 8 | finished | 4 |
| [tc\_independent\_k16](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose/runs/rjhk4uat) | Transcoder | independent | 4096 | 16 | finished | 4 |
| [tc\_independent\_k32](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose/runs/9v7q1zc6) | Transcoder | independent | 4096 | 32 | finished | 4 |
| [tc\_independent\_k64](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose/runs/n53887yb) | Transcoder | independent | 4096 | 64 | finished | 4 |
| [clt\_cascading\_k8](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose/runs/2t7c7oml) | CLT | cascading | 4096 | 8 | finished | 1 |
| [clt\_cascading\_k16](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose/runs/0lvptasy) | CLT | cascading | 4096 | 16 | finished | 1 |
| [clt\_cascading\_k32](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose/runs/75w3puee) | CLT | cascading | 4096 | 32 | finished | 1 |
| [clt\_cascading\_k64](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose/runs/g65tcjl6) | CLT | cascading | 4096 | 64 | finished | 1 |
| [clt\_parallel\_k8](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose/runs/p8c3mqql) | CLT | parallel | 4096 | 8 | finished | 1 |
| [clt\_parallel\_k16](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose/runs/k88ilqfu) | CLT | parallel | 4096 | 16 | finished | 1 |
| [clt\_parallel\_k32](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose/runs/oiikseki) | CLT | parallel | 4096 | 32 | finished | 1 |
| [clt\_parallel\_k64](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose/runs/fz4b8yx0) | CLT | parallel | 4096 | 64 | finished | 1 |

### **32k dict ([pile\_e2e\_sweep\_jose\_32k](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose_32k))**

| Run | Type | Mode | Dict Size | Top-k | Status | Artifacts |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| [tc\_cascading\_k8](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose_32k/runs/3fqbtuq7) | Transcoder | cascading | 32768 | 8 | finished | 4 |
| [tc\_cascading\_k16](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose_32k/runs/i8ij10e5) | Transcoder | cascading | 32768 | 16 | finished | 4 |
| [tc\_cascading\_k32](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose_32k/runs/8f42smnn) | Transcoder | cascading | 32768 | 32 | running | 0 |
| [tc\_cascading\_k64](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose_32k/runs/od93aebm) | Transcoder | cascading | 32768 | 64 | running | 0 |
| [tc\_parallel\_k8](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose_32k/runs/1vrbgs47) | Transcoder | parallel | 32768 | 8 | finished | 4 |
| [tc\_parallel\_k16](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose_32k/runs/p1q596pr) | Transcoder | parallel | 32768 | 16 | finished | 4 |
| [tc\_parallel\_k32](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose_32k/runs/g5cleafb) | Transcoder | parallel | 32768 | 32 | finished | 4 |
| [tc\_parallel\_k64](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose_32k/runs/nb8gppuy) | Transcoder | parallel | 32768 | 64 | running | 0 |
| [tc\_independent\_k8](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose_32k/runs/l8asugzq) | Transcoder | independent | 32768 | 8 | finished | 4 |
| [tc\_independent\_k16](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose_32k/runs/1c2mu991) | Transcoder | independent | 32768 | 16 | finished | 4 |
| [tc\_independent\_k32](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose_32k/runs/dr5q43yq) | Transcoder | independent | 32768 | 32 | finished | 4 |
| [tc\_independent\_k64](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose_32k/runs/9oq96slq) | Transcoder | independent | 32768 | 64 | running | 0 |
| [clt\_cascading\_k8](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose_32k/runs/2uzec2cd) | CLT | cascading | 32768 | 8 | finished | 1 |
| [clt\_cascading\_k16](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose_32k/runs/cmgaj26o) | CLT | cascading | 32768 | 16 | running | 0 |
| [clt\_cascading\_k32](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose_32k/runs/wdab0s15) | CLT | cascading | 32768 | 32 | running | 0 |
| [clt\_parallel\_k8](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose_32k/runs/f7yjyafh) | CLT | parallel | 32768 | 8 | finished | 1 |
| [clt\_parallel\_k16](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose_32k/runs/e9pih3sr) | CLT | parallel | 32768 | 16 | finished | 1 |
| [clt\_parallel\_k32](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose_32k/runs/a05z1kta) | CLT | parallel | 32768 | 32 | running | 0 |
| [clt\_parallel\_k64](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose_32k/runs/juotwt3p) | CLT | parallel | 32768 | 64 | running | 0 |

##   **Multi-Seed Training (Stability Analysis)**

Training mode: 5 seeds × 4 modes, k=16, 4k dict. For comparison with SPD component stability.

### [**pile\_multiseed\_jose2**](https://wandb.ai/mats-sprint/pile_multiseed_jose2)

Transcoders (local MSE)

| Run | Type | Dict Size | Top-k | Seed | Status |
| ----- | ----- | ----- | ----- | ----- | ----- |
| [local\_mse\_k16\_seed0](https://wandb.ai/mats-sprint/pile_multiseed_jose2/runs/mrwfwu8f) | Transcoder | 4096 | 16 | 0 | finished |
| [local\_mse\_k16\_seed1](https://wandb.ai/mats-sprint/pile_multiseed_jose2/runs/g79l9mdy) | Transcoder | 4096 | 16 | 1 | finished |
| [local\_mse\_k16\_seed2](https://wandb.ai/mats-sprint/pile_multiseed_jose2/runs/4aye18d8) | Transcoder | 4096 | 16 | 2 | finished |
| [local\_mse\_k16\_seed3](https://wandb.ai/mats-sprint/pile_multiseed_jose2/runs/jos97f9i) | Transcoder | 4096 | 16 | 3 | finished |
| [local\_mse\_k16\_seed4](https://wandb.ai/mats-sprint/pile_multiseed_jose2/runs/qy13k51o) | Transcoder | 4096 | 16 | 4 | finished |

Transcoders (e2e independent)

| Run | Type | Dict Size | Top-k | Seed | Status |
| ----- | ----- | ----- | ----- | ----- | ----- |
| [e2e\_independent\_k16\_seed0](https://wandb.ai/mats-sprint/pile_multiseed_jose2/runs/z9567vuy) | Transcoder | 4096 | 16 | 0 | finished |
| [e2e\_independent\_k16\_seed1](https://wandb.ai/mats-sprint/pile_multiseed_jose2/runs/kxpwcbdx) | Transcoder | 4096 | 16 | 1 | finished |
| [e2e\_independent\_k16\_seed2](https://wandb.ai/mats-sprint/pile_multiseed_jose2/runs/abat1svk) | Transcoder | 4096 | 16 | 2 | finished |
| [e2e\_independent\_k16\_seed3](https://wandb.ai/mats-sprint/pile_multiseed_jose2/runs/uuboaw36) | Transcoder | 4096 | 16 | 3 | finished |
| [e2e\_independent\_k16\_seed4](https://wandb.ai/mats-sprint/pile_multiseed_jose2/runs/xbusz9qi) | Transcoder | 4096 | 16 | 4 | finished |

CLTs (local MSE)

| Run | Type | Dict Size | Top-k | Seed | Status |
| ----- | ----- | ----- | ----- | ----- | ----- |
| [clt\_local\_mse\_k16\_seed0](https://wandb.ai/mats-sprint/pile_multiseed_jose2/runs/17sbdath) | CLT | 4096 | 16 | 0 | finished |
| [clt\_local\_mse\_k16\_seed1](https://wandb.ai/mats-sprint/pile_multiseed_jose2/runs/xmzt89pl) | CLT | 4096 | 16 | 1 | finished |
| [clt\_local\_mse\_k16\_seed2](https://wandb.ai/mats-sprint/pile_multiseed_jose2/runs/tqyguqn0) | CLT | 4096 | 16 | 2 | finished |
| [clt\_local\_mse\_k16\_seed3](https://wandb.ai/mats-sprint/pile_multiseed_jose2/runs/ft6ljijh) | CLT | 4096 | 16 | 3 | finished |
| [clt\_local\_mse\_k16\_seed4](https://wandb.ai/mats-sprint/pile_multiseed_jose2/runs/iajwsefm) | CLT | 4096 | 16 | 4 | finished |

CLTs (e2e parallel)

| Run | Type | Dict Size | Top-k | Seed | Status |
| ----- | ----- | ----- | ----- | ----- | ----- |
| [clt\_e2e\_parallel\_k16\_seed0](https://wandb.ai/mats-sprint/pile_multiseed_jose2/runs/wf7mbave) | CLT | 4096 | 16 | 0 | finished |
| [clt\_e2e\_parallel\_k16\_seed1](https://wandb.ai/mats-sprint/pile_multiseed_jose2/runs/7o579nw1) | CLT | 4096 | 16 | 1 | finished |
| [clt\_e2e\_parallel\_k16\_seed2](https://wandb.ai/mats-sprint/pile_multiseed_jose2/runs/uj5xomc0) | CLT | 4096 | 16 | 2 | finished |
| [clt\_e2e\_parallel\_k16\_seed3](https://wandb.ai/mats-sprint/pile_multiseed_jose2/runs/st70dv8u) | CLT | 4096 | 16 | 3 | finished |
| [clt\_e2e\_parallel\_k16\_seed4](https://wandb.ai/mats-sprint/pile_multiseed_jose2/runs/ewotesmm) | CLT | 4096 | 16 | 4 | finished |

## **Summary**

| Project | Dict | Training | Finished | Running | Failed | Total |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| [pile\_local\_sweep\_jose](https://wandb.ai/mats-sprint/pile_local_sweep_jose) | 4k | local MSE | 8 | 0 | 0 | 8 |
| [pile\_local\_sweep\_jose\_32k](https://wandb.ai/mats-sprint/pile_local_sweep_jose_32k) | 32k | local MSE | 7 | 1 | 0 | 8 |
| [pile\_e2e\_sweep\_jose](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose) | 4k | e2e KL | 20 | 0 | 0 | 20 |
| [pile\_e2e\_sweep\_jose\_32k](https://wandb.ai/mats-sprint/pile_e2e_sweep_jose_32k) | 32k | e2e KL | 10 | 9 | 0 | 19 |
| [pile\_multiseed\_jose2](https://wandb.ai/mats-sprint/pile_multiseed_jose2) | 4k | multi-seed (4 modes × 5 seeds) | 20 | 0 | 0 | 20 |

All runs use: LR=3e-4, batch\_size=4096, seq\_len=512, 500M tokens, BatchTopK activation.

