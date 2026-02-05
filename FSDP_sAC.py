import os
# https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/72926996#72926996: Rust multi-threading conflicts when Dataloader forks its workers.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from argparse import ArgumentParser, Namespace, BooleanOptionalAction
from pathlib import Path
import random
import socket

from datasets import load_dataset
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.aggregation import RunningMean
from torchmetrics.text import Perplexity
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.distributed as dist

from utils import (
    apply_fsdp_checkpointing,
    expand_hostlist,
    make_sft_collate,
    memory_usage,
    TrainingChronometer,
)


def cleanup():
    """Clean up distributed training"""
    dist.destroy_process_group()

def get_rank():
    """Get the rank of the current process"""
    return dist.get_rank()

def get_world_size():
    """Get the total number of processes"""
    return dist.get_world_size()

def is_main_process():
    """Check if this is the main process"""
    return get_rank() == 0

def setup():
    """Initialize distributed training.

    With SLURM:
        All environment variables (SLURM_JOB_NODELIST, SLURM_JOB_ID, SLURM_NTASKS, etc.)
        are automatically set by the scheduler. The master port is derived from SLURM_JOB_ID
        to minimize the chance of using an already allocated port.

    Without SLURM (we suppose that it is a mono-gpu script):
        Defaults to localhost with a random port. For multiple single-GPU scripts on the
        same node, set job to select the GPU for each script:
            CUDA_VISIBLE_DEVICES=0 python FSDP_sAC.py

    Returns:
        torch.device: The CUDA device assigned to this process.
    """
    # SLURM environment variables
    master_addr = expand_hostlist(str(os.environ.get("SLURM_JOB_NODELIST", "localhost")))[0]
    master_port = 10000 + int(os.environ.get("SLURM_JOB_ID", random.randint(0, 9999))) % 10000
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    n_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
    gpus_per_node = int(os.environ.get("SLURM_GPUS_ON_NODE", 1))
    node_id = int(os.environ.get("SLURM_NODEID", 0))
    rank = int(os.environ.get("SLURM_PROCID", 0))
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))

    # Set environment variables for PyTorch
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = str(master_addr)
    os.environ["MASTER_PORT"] = str(master_port)

    # Initialize process group
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)

    # Summary
    if is_main_process():
        print(f"{rank} - Master node    : {master_addr}")
        print(f"{rank} - Port           : {master_port}")
        print(f"{rank} - World size     : {world_size}")
        print(f"{rank} - Number of nodes: {n_nodes}")
        print(f"{rank} - GPUs per node  : {gpus_per_node}")
        print(f"{rank} - Hostname       : {socket.gethostname()}")
        print(f"{rank} - Node ID        : {node_id}")
        print(f"{rank} - Local rank     : {local_rank}")

    return torch.device(f'cuda:{local_rank}')

def parse_args() -> Namespace:
    """Process command-line arguments."""
    parser = ArgumentParser()

    # Training related arguments
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size per GPU.')
    parser.add_argument("--seq-length", type=int, default=4096, help='Sequence length of each sample per GPU.')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs.')
    parser.add_argument('--grad-acc', type=int, default=1, help='Number of batches used for a single weights update.')

    # Benchmarking / debugging arguments
    parser.add_argument('--test', action=BooleanOptionalAction, default=False, help='Run in test mode for a limited number of steps.')
    parser.add_argument('--test-nsteps', type=int, default=100, help='Number of steps to run in test mode.')
    parser.add_argument("--display-optimizer-dtype", action=BooleanOptionalAction, default=False, help="Print precision of optimizer state tensors.")

    # DataLoader related arguments
    parser.add_argument('--dataset-path', type=Path, help="HuggingFaceHub dataset path.")
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers spawned by the dataloader.')
    parser.add_argument('--prefetch-factor', type=int, default=2, help='Number of batches loaded in RAM per worker.')

    # Optimizer AdamW related arguments
    parser.add_argument("--lr-warmup-ratio", type=float, default=0.1, help="Fraction of training steps for linear LR warmup (0.1 = 10%).")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate for AdamW.")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay for AdamW.")

    # Model related arguments
    parser.add_argument("--model-path", type=Path, help="HuggingFaceHub model path.")
    parser.add_argument("--selective-activation-checkpointing", type=str, default=None, help='For a given ac ratio p, we should essentially apply ac on every "1/p" blocks.')
    parser.add_argument("--compile", action=BooleanOptionalAction, default=False, help="Whether or not to compile model.")

    return parser.parse_args()


def main():
    # 1. Get command-line arguments
    args = parse_args()


    # 2. Distributed Training Setup (SLURM/NCCL)
    device = setup()

    if is_main_process():
        chrono = TrainingChronometer()

        print(f"Global batch size                       : {args.batch_size*args.grad_acc*get_world_size()}")
        print(f"Gradient accumulation                   : {args.grad_acc}")
        print(f"Mini batch size (per GPU)               : {args.batch_size}")
        print(f"Sequence length                         : {args.seq_length}")
        print(f"Selective activation checkpointing ratio: {args.selective_activation_checkpointing}")
        print(f"Compile                                 : {args.compile}")
    else:
        chrono = None


    # 3. Model processing

    ## bf16 model
    if is_main_process(): chrono.timer(start=True)

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="bfloat16", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left", use_fast=True)

    if is_main_process(): print(f"Time to load the model (bf16) and its tokenizer: {chrono.timer(start=False):.3f} s")

    ## sAC
    if args.selective_activation_checkpointing:
        model.config.use_cache = False  # deactivate KV caching (conflicts with activation checkpointing)
        BlockCls = type(model.model.layers[0])
        if is_main_process(): print(BlockCls)
        apply_fsdp_checkpointing(model, BlockCls, args.selective_activation_checkpointing)

    ## FSDP
    if is_main_process(): chrono.timer(start=True)

    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    }

    for layer in model.model.layers:
        fully_shard(layer.type(torch.float32), **fsdp_kwargs)
    fully_shard(model.type(torch.float32), **fsdp_kwargs)

    if is_main_process(): print(f"Time to shard the model: {chrono.timer(start=False):.3f} s")

    ## Transfer to GPU
    model = model.to(device)

    if is_main_process(): print(f"Time to transfer the model to GPU: {chrono.timer(start=False):.3f} s")

    ## JIT
    if args.compile:
        model = torch.compile(model)

        if is_main_process(): print(f"Time for torch.compile: {chrono.timer(start=False):.3f} s")

    if is_main_process():
        print(f"Model: {model}")
        print(f"Number of parameters: {sum(param.numel() for param in model.parameters())}")
        print(f'Pre-loop GPU memory usage: {torch.cuda.max_memory_allocated()/2**30} GB')


    # 4. Data processing
    train_dataset = load_dataset(str(args.dataset_path), split="train")
    collate_fn = make_sft_collate(tokenizer, max_seq_length=args.seq_length)

    sampler = DistributedSampler(
        dataset=train_dataset,
        rank=get_rank(),
        num_replicas=get_world_size(),
        shuffle=True,
    )
    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        sampler=sampler,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True,
        pin_memory=True,
        shuffle=False
    )

    if is_main_process(): print(f"Time to load dataset and initialize dataloader: {chrono.timer(start=False):.3f} s")


    # 5. Training preparation
    criterion = CrossEntropyLoss(ignore_index=-100)
    optimizer = AdamW(
        params=model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-05,
    )

    if is_main_process():
        print(f"DataLoader: num_workers={args.num_workers} / prefetch_factor={args.prefetch_factor}")
        print(f"Optimizer: {optimizer}")

    total_steps = len(dataloader) * args.epochs // args.grad_acc
    lr_warmup_iters = int(args.lr_warmup_ratio * total_steps)
    warmup_lr_scheduler = LinearLR(
        optimizer,
        start_factor=1e-6,
        end_factor=1,
        total_iters=lr_warmup_iters,
    )
    annealing_lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - lr_warmup_iters,
        eta_min=0.,
    )
    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_lr_scheduler, annealing_lr_scheduler],
        milestones=[lr_warmup_iters]
    )

    avg_loss = RunningMean(window=10*args.grad_acc).to(device)  # average of the last 10 steps
    perplexity = Perplexity(ignore_index=-100).to(device)  # save all perplexity computations


    # 6. Training loop
    if is_main_process():
        chrono.track_training_time(start=True)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)  # set epoch for a sampler

        if is_main_process():
            chrono.timer(start=True)  # track the first step duration
            chrono.track_dataloading_step_time(start=True)

        for i, (input_ids, labels, attention_mask) in enumerate(dataloader, start=1):
            if args.test and i > args.test_nsteps * args.grad_acc: break

            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            if is_main_process():
                chrono.track_dataloading_step_time(start=False)
                chrono.track_forward_step_time(start=True)

            # Forward
            local_logits: torch.Tensor = model(input_ids, attention_mask=attention_mask).logits
            bs, seq_len, vocab_size = local_logits.shape
            local_loss: torch.Tensor = criterion(local_logits.view(bs * seq_len, vocab_size), labels.view(bs * seq_len))
            local_loss /= args.grad_acc  # take into account gradient accumulation impact

            # Global metrics
            avg_loss.update(local_loss)
            perplexity.update(local_logits, labels)

            if is_main_process():
                chrono.track_forward_step_time(start=False)
                chrono.track_backward_step_time(start=True)

            local_loss.backward()  # gradients are automatically divided by world_size: https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_runtime_utils.py#L831

            if i % args.grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if is_main_process():
                chrono.track_backward_step_time(start=False)
                chrono.track_dataloading_step_time(start=True)

            # Log training info
            step = ( (i-1) // args.grad_acc) + 1
            if step % 10 == 0 and i % args.grad_acc == 0:  # at the end of every 10 steps
                L = avg_loss.compute()
                perp = perplexity.compute()
                last_lr = lr_scheduler.get_last_lr()[0]
                if is_main_process():
                    print(f"Rank {get_rank()}: Step {step} / {args.test_nsteps if args.test else len(dataloader) // args.grad_acc} | Local loss on 10 steps: {L.item():.3f} | Perplexity from start: {perp.item():.3f} | Last LR: {last_lr:0.3e}")

            if i == 1 and is_main_process():
                print(f"Time to first step (may include torch.compile tracing): {chrono.timer(start=False):.3f} s")


        # Checkpointing by rank 0
        if not args.test and is_main_process():
            checkpoint_name = f"checkpoint/model_state_dict_{os.environ.get('SLURM_JOB_ID', 'XXX')}_epoch{epoch}.pt"

            print(f"Model Checkpointing - Building the {checkpoint_name} file")
            model_state_dict = get_model_state_dict(
                model=model,
                options=StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                )
            )
            torch.save(model_state_dict, checkpoint_name)
    dist.barrier()

    if is_main_process():
        chrono.track_training_time(start=False)
        chrono.display_training_results(len(dataloader), args.grad_acc)

        memory_usage()
        print(f'Post-loop GPU memory usage: {torch.cuda.max_memory_allocated()/2**30} GBytes')

        if args.display_optimizer_dtype:
            for k, v in optimizer.state.items():
                print("Optimizer state dtypes:", {kk: vv.dtype for kk, vv in v.items()})
                break

    cleanup()


if __name__ == "__main__":
    main()
