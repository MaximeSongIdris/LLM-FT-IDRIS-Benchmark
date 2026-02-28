#!/usr/bin/python3
"""SFT training script using HFAutoModelForCausalLM with FSDP2 strategy.

Supports SingleDevice and FSDP2+TP+CP strategies for distributed training.
For Megatron-native training (DDP+TP+CP), see nemo_megatron.py.

Source: https://docs.nvidia.com/nemo-framework/user-guide/25.04/automodel/sft.html
"""

from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from math import ceil
from pathlib import Path
import os

from datasets import load_dataset
from nemo import lightning as nl
from nemo.automodel.loss import masked_cross_entropy
from nemo.automodel.misc_utils import calculate_valid_accumulate_grad_batches
from nemo.collections import llm
from nemo.collections.llm.gpt.data.hf_dataset import HFDatasetDataModule
from nemo.lightning.pytorch.callbacks import JitConfig, JitTransform
from transformers import AutoTokenizer
import fiddle as fdl
import lightning.pytorch as pl

from utils import make_sft_collate, MyChronoCallback


def _get_offload_policy(enable_cpu_offload: bool) -> "CPUOffloadPolicy | None":
    """Return CPU offload policy if enabled, None otherwise."""
    if not enable_cpu_offload:
        return None
    from nemo.lightning.pytorch.strategies.fsdp2_strategy import HAS_CPU_OFFLOAD_POLICY, CPUOffloadPolicy
    assert HAS_CPU_OFFLOAD_POLICY, "Could not import offload policy"
    return CPUOffloadPolicy()

def create_strategy(
    strategy_name: str,
    model: llm.HFAutoModelForCausalLM,
    devices_per_node: int,
    num_nodes: int,
    dp_size: int,
    tp_size: int,
    cp_size: int,
    sequence_parallel: bool = False,
    enable_cpu_offload: bool = False,
    rank: int = 0,
) -> pl.strategies.Strategy:
    """Create a PyTorch Lightning training strategy for distributed training.

    Supports single-device training or multi-device parallelism with combinations
    of Data Parallel (DP), Tensor Parallel (TP), and Context Parallel (CP).
    """
    checkpoint_io = model.make_checkpoint_io()

    if strategy_name == 'SingleDevice':
        return pl.strategies.SingleDeviceStrategy(device='cuda:0', checkpoint_io=checkpoint_io)

    elif strategy_name == 'FSDP2+TP+CP':
        total_devices = devices_per_node * num_nodes
        expected = dp_size * tp_size * cp_size
        assert expected == total_devices, (f"DP*TP*CP = {expected} must equal devices*num_nodes = {total_devices}")
        
        if rank == 0:
            print(f"Using FSDP2Strategy (FSDP2+TP+CP) with DP={dp_size}, TP={tp_size}, CP={cp_size}")
        return nl.FSDP2Strategy(
            data_parallel_size=dp_size,
            tensor_parallel_size=tp_size,
            use_hf_tp_plan=True,
            sequence_parallel=sequence_parallel,
            context_parallel_size=cp_size,
            offload_policy=_get_offload_policy(enable_cpu_offload),
            checkpoint_io=checkpoint_io,
        )

    raise ValueError(
        f"Unknown strategy: {strategy_name}. "
        f"This script supports 'SingleDevice' and 'FSDP2+TP+CP'. "
        f"For MegatronStrategy (DDP+TP+CP), use nemo_megatron.py."
    )


def parse_args() -> Namespace:
    """Process command-line arguments."""
    parser = ArgumentParser()

    # Training related arguments
    parser.add_argument('--global-batch-size', type=int, default=128, help='Number of examples seen for one model update.')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size per GPU.')
    parser.add_argument('--seq-length', type=int, default=4096, help='Sequence length of each sample per GPU.')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs.')

    # Benchmarking / debugging arguments
    parser.add_argument('--test', action=BooleanOptionalAction, default=False, help='Run in test mode for a limited number of steps.')
    parser.add_argument('--test-nsteps', type=int, default=100, help='Number of steps to run in test mode.')

    # DataLoader related arguments
    parser.add_argument('--dataset-path', type=Path, help='HuggingFaceHub dataset path.')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers spawned by the dataloader.')

    # Optimizer related arguments
    parser.add_argument('--lr-warmup-ratio', type=float, default=0.1, help='Fraction of training steps for linear LR warmup (0.1 = 10%).')
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='Learning rate for AdamW.')
    parser.add_argument('--weight-decay', type=float, default=0.1, help='Weight decay for AdamW.')

    # Model related arguments
    parser.add_argument('--model-path', type=Path, help='HuggingFaceHub model path.')
    parser.add_argument('--trust-remote-code', action=BooleanOptionalAction, default=False, help='Trust remote code for HF models.')
    parser.add_argument('--attn-implementation', choices=['flash_attention_2', 'sdpa', 'eager'],
                        default='flash_attention_2', help='Attention implementation.')
    parser.add_argument('--activation-checkpointing', action=BooleanOptionalAction, default=False, help='Enable activation checkpointing.')
    parser.add_argument('--fp8', action=BooleanOptionalAction, default=False, help='Enable FP8 training.')
    parser.add_argument('--liger', action=BooleanOptionalAction, default=False, help='Enable Liger-Kernels.')
    parser.add_argument('--compile', action=BooleanOptionalAction, default=False, help='Whether or not to compile model.')

    # Distributed training arguments
    parser.add_argument('--strategy', choices=['SingleDevice', 'FSDP2+TP+CP'], default='SingleDevice', help='Training strategy.')
    parser.add_argument('--devices-per-node', type=int, default=1, help='Number of GPUs per node.')
    parser.add_argument('--num-nodes', type=int, default=1, help='Number of nodes.')
    parser.add_argument('--dp-size', type=int, default=1, help='Data parallel size.')
    parser.add_argument('--tp-size', type=int, default=1, help='Tensor parallel size.')
    parser.add_argument('--cp-size', type=int, default=1, help='Context parallel size.')
    parser.add_argument('--sequence-parallel', action=BooleanOptionalAction, default=False, help='Enable sequence parallelism (requires TP>1).')
    parser.add_argument('--enable-cpu-offload', action=BooleanOptionalAction, default=False, help='Enable CPU offloading (FSDP2 only).')

    # Logging / Checkpointing arguments
    parser.add_argument('--log-every-n-steps', type=int, default=10, help='Logging frequency.')
    parser.add_argument('--wandb-project', help='Wandb project name.')

    return parser.parse_args()


def main() -> None:
    """Run SFT with HuggingFace model using FSDP2 strategies."""
    # 1. Get command-line arguments
    args = parse_args()


    # 2. Distributed Training Setup
    world = args.devices_per_node * args.num_nodes
    rank = int(os.environ.get("RANK", 0))  # Set by torchrun
    
    assert args.dp_size * args.tp_size * args.cp_size == world, \
        f"3D mismatch: DP*TP*CP={args.dp_size*args.tp_size*args.cp_size} != world={world}"

    ## Print info
    grad_acc = calculate_valid_accumulate_grad_batches(
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.batch_size,
        devices=args.devices_per_node,
        num_nodes=args.num_nodes,
        tp_size=args.tp_size,
        cp_size=args.cp_size,
    )
    if rank == 0:
        print(f"World size               : {world}")
        print(f"Global batch size        : {args.global_batch_size}")
        print(f"Gradient accumulation    : {grad_acc}")
        print(f"Mini batch size (per GPU): {args.batch_size}")
        print(f"Sequence length          : {args.seq_length}")
        print(f"Activation checkpointing : {args.activation_checkpointing}")
        print(f"Compile                  : {args.compile}")
        print(f"Attention                : {args.attn_implementation}")
        print(f"Liger-Kernels            : {args.liger}")
        print(f"FP8 training             : {args.fp8}")


    # 3. Model processing
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left", use_fast=True)

    ## FP8
    model_accelerator = None
    if args.fp8:
        from nemo.lightning.pytorch.accelerate.transformer_engine import TEConfig
        model_accelerator = TEConfig(fp8_autocast=True)
        
    # HF
    model = llm.HFAutoModelForCausalLM(
        model_name=args.model_path,
        tokenizer=tokenizer,
        loss_fn=masked_cross_entropy,  # https://github.com/NVIDIA-NeMo/NeMo/blob/25.09-alpha.rc2/nemo/automodel/loss/masked_ce.py
        model_accelerator=model_accelerator,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        use_liger_kernel=args.liger,
        enable_grad_ckpt=args.activation_checkpointing,
    )  # https://github.com/NVIDIA-NeMo/NeMo/blob/25.09-alpha.rc2/nemo/collections/llm/gpt/model/hf_auto_model_for_causal_lm.py

    ## JIT
    callbacks = []
    if args.compile:
        jit_config = JitConfig(use_torch=True, torch_kwargs={'dynamic': False}, use_thunder=False)
        callbacks.append(JitTransform(jit_config))

    # Distributed training strategy
    strategy = create_strategy(
        strategy_name=args.strategy,
        model=model,
        devices_per_node=args.devices_per_node,
        num_nodes=args.num_nodes,
        dp_size=args.dp_size,
        tp_size=args.tp_size,
        cp_size=args.cp_size,
        sequence_parallel=args.sequence_parallel,
        enable_cpu_offload=args.enable_cpu_offload,
        rank=rank,
    )

    if rank == 0:
        print(f"Strategy                 : {args.strategy}")
        print(f"DP size                  : {args.dp_size}")
        print(f"TP size                  : {args.tp_size}")
        print(f"CP size                  : {args.cp_size}")
        print(f"Sequence parallel        : {args.sequence_parallel}")
        print(f"CPU offload              : {args.enable_cpu_offload}")


    # 4. Data processing
    dataset = load_dataset(str(args.dataset_path))

    def wrap_tuple_to_dict(old_collate_fn):
        def new_collate_fn(batch):
            t = old_collate_fn(batch)
            if isinstance(t, tuple) and len(t) >= 3:
                input_ids = t[0]
                labels = t[1]
                attention_mask = t[2]
                # loss_mask: compute loss where labels != -100 (Token ID used to pad labels
                loss_mask = (labels != -100).float()
                return {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention_mask,
                    "loss_mask": loss_mask,  # necessary for Context Parallelism
                }
            raise TypeError(f"collate_fn returned {type(t)}, expected tuple of 3 elements")
        return new_collate_fn
    collate_fn = wrap_tuple_to_dict(make_sft_collate(tokenizer, max_seq_length=args.seq_length))

    hf_dataset = HFDatasetDataModule(
        dataset,
        split="train",
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        seq_length=args.seq_length,
        micro_batch_size=args.batch_size,
        use_dist_sampler=True,
        pad_seq_len_divisible=16 if args.fp8 else None,  # https://docs.nvidia.com/nemo/automodel/latest/guides/dataset-overview.html#important-considerations
    )  # https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/llm/gpt/data/hf_dataset.py#L193


    # 5. Training preparation

    # Optimizer
    total_steps = args.epochs * ceil(len(dataset['train']) / args.global_batch_size)
    lr_warmup_iters = int(args.lr_warmup_ratio * total_steps)
    optimizer = fdl.build(
        llm.adam.pytorch_adam_with_cosine_annealing(
            warmup_steps=lr_warmup_iters,
            max_lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    )


    # 6. Training loop

    # Wandb logging
    wandb = None
    if args.wandb_project is not None:
        from lightning.pytorch.loggers import WandbLogger
        wandb = WandbLogger(
            project=args.wandb_project,
            name = (
                    f"{args.model_path.name}"
                    f"_nodes{args.num_nodes}"
                    f"_devices{args.devices_per_node}"
                    f"_strat_{args.strategy}"
                    f"_dp{args.dp_size}"
                    f"_tp{args.tp_size}"
                    f"_cp{args.cp_size}"
                    f"_sp{args.sequence_parallel}"
                    f"_gbs{args.global_batch_size}"
                    f"_mbs{args.batch_size}"
                    f"_seqlen{args.seq_length}"
            )
        )

    # Chrono logging
    callbacks.append(
        MyChronoCallback(
            rank,
            args.test_nsteps if args.test else total_steps,                # total number of weight updates
            args.global_batch_size,                                        # number of samples per weight update
            args.seq_length,                                               # number of tokens per sample
            ceil(len(dataset['train']) / (args.batch_size*args.dp_size)),  # number of batches per epoch
            grad_acc                                                       # number of batches per epoch / grad_acc = number of weight updates per epoch !
        )
    )

    llm.api.finetune(
        model=model,
        data=hf_dataset,
        trainer=nl.Trainer(
            accelerator='gpu',
            strategy=strategy,
            devices=args.devices_per_node,
            num_nodes=args.num_nodes,
            precision="bf16-mixed",
            logger=wandb,
            callbacks=callbacks,
            max_epochs=args.epochs,
            max_steps=args.test_nsteps if args.test else -1,
            limit_val_batches=0.0,  # no validation phase
            num_sanity_val_steps=0,  # no sanity check before training
            log_every_n_steps=args.log_every_n_steps,
            enable_checkpointing=False,
            enable_model_summary=False,
            accumulate_grad_batches=grad_acc,
            gradient_clip_val=0.0,  # doesn't work with Tensor Parallel due to sharding of some layers (embedding layer, LayerNorm, etc.)
            use_distributed_sampler=True,  # needed to use PL DistributedSampler
        ),
        optim=optimizer,
        peft=None,
    )  # https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/trainer/trainer.py#L89


if __name__ == '__main__':
    main()