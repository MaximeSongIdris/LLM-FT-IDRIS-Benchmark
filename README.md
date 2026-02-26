## Env installation

- From container: `singularity build nemo_2509.sif docker://nvcr.io/nvidia/nemo:25.09`

## Experience

### 1) FSDP2 + Selective Activation Checkpointing on H100 80 Go (Qwen2.5-7B-Instruct)

#### Required AC and GA for multi-gpus training with fixed batch size per GPU (effective batch size = 64)

|         | bs=1          | bs=2           | bs=4         | bs=8          |
|---------|---------------|----------------|--------------|---------------|
| GPUs=1  | AC=0.4, GA=64 | AC=0.85, GA=32 | -            | -             |
| GPUs=4  | -             | AC=0.0, GA=8   | AC=0.5, GA=4 | AC=0.95, GA=2 |
| GPUs=8  | -             | AC=0.0, GA=4   | AC=0.5, GA=2 | AC=0.90, GA=1 |
| GPUs=16 | -             | AC=0.0, GA=2   | AC=0.4, GA=1 | -             |

- **AC** (Activation Checkpointing): ratio of activation layers that are not in memory (0.0 = all in memory, 1.0 = nothing in memory). Trades compute for memory.
- **GA** (Gradient Accumulation): number of forward/backward passes before optimizer step. Trades compute for memory.
- **Effective batch size** = GPUs × bs × GA = 64 for all configurations

#### Training time depending on the number of GPUs and on the effect of Selective Activation Checkpointing

<img src="asset/training_time_vs_activation_ckpt.png" width="800">

By increasing the bs thanks to the selective activation checkpointing, we expected to speed-up the training as we reduce the costly gradient accumulation. Furthermore, since we are doing less forwards/backwards in total, it should be further speed-up as we reduce the number of communication. However as soon as we use FSDP2 (multi-gpus training), AC starts actually increasing the training time. Why ?

#### Analysis on 4 GPUs with effective batch size = 4

- With GA=4, we perform 4 forward+backward on bs=1 passes before 1 optimizer step.
- With GA=2 and activation checkpointing, we perform 2 forward+backward on bs=2, plus a recompute cost of at most 2 additional forwards on bs=2.
- In theory, forward/backward on bs=2 should be more efficient than 2 forward/backward on bs=1 as we use more efficiently the GPU (parallelizing on the dimension of the batch) and reduce the overhead of launching multiple kernels.
- In practice, when looking at the forward, the time actually increased almost linearly on the bs ([trace for bs=1](asset/forward_bs1.png) / [trace for bs=2](asset/forward_bs2.png)), passing from **138 ms** to **258 ms**. If we zoom at the forward, it is made of the forward of **28 attention layers** with each attention layer forward scaling linearly. Inside of these attention layer forward, we have 4 big kernels that dominate and among them, the biggest kernel scales from **1,123 ms** to **2,224 ms** ([trace for bs=1](asset/forward_attention_layer_bs1.png) / [trace for bs=2](asset/forward_attention_layer_bs2.png)).
- In bs=1 and bs=2, the kernel configuration is identical: 132 blocks and 384 threads per block, which means that we are actually asking for each thread to work twice as much ( twice the data transfer and twice the computation).
- Since we have 1 block per SM (132 SM in H100) and 4 schedulers per SM, and each scheduler deals by group of 32 threads. Each scheduler has 3 groups.
- By using Nsight compute, we can see that the average scheduler executed 530 524 instructions for bs=1 and 1 060 689 instructions for bs=2. Thus we can infer that each group of threads is dealing with twice the work by doing it sequentially.
- In Nsight compute, with bs=1, the Tensor Core is only active 31% of cycles and memory throughput reaches 60%. This suggests spare capacity exists. In theory, bs=2 instructions could fill the idle cycles by interleaving batch 0 and batch 1 operations within each warp. However, this interleaving would require storing two independent working contexts simultaneously in registers, which is limited. The bottleneck here may be due to the on-chip memory (we could try to verify by looking at the warp lifecycle and monitor the pipe usage and register usage).

### Issues

- I tried to monitor the GPU by capturing the trace via Nsight system (`nsys profile --trace osrt,cuda,cublas,cudnn,nvtx`), but I couldn't get the detailed GPU view, instead I've got only the CPU view and the Process view (which has some GPU metrics).

## Sources

- Original code: https://github.com/BertrandCabotPro/Democratizing-LLM-FT
- Source for FSDP + Selective activation checkpointing: https://pytorch.org/blog/maximizing-training/
- SLURM configuration for NeMo: https://docs.nvidia.com/nemo/automodel/latest/launcher/cluster.html
- Nvidia GPU guide: https://modal.com/gpu-glossary
