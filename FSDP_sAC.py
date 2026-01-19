import os
# https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/72926996#72926996: Rust multi-threading conflicts when Dataloader forks its workers.
os.environ["TOKENIZER_PARALLELISM"] = "false"

from argparse import ArgumentParser, Namespace, BooleanOptionalAction
from pathlib import Path

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
    Chronometer,
    make_sft_collate,
    memory_usage
)


# 1. Distributed Training Setup (SLURM/NCCL)
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
    """Initialize distributed training with SLURM"""
    # SLURM environment variables
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))

    # Set environment variables for PyTorch
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    if "SLURM_JOB_NODELIST" in os.environ:  # Get master address and port from SLURM
        hostnames = expand_hostlist(os.environ["SLURM_JOB_NODELIST"])
        master_addr = hostnames[0]
        os.environ["MASTER_ADDR"] = master_addr
    else:
        os.environ["MASTER_ADDR"] = "localhost"

    os.environ["MASTER_PORT"] = str(10000 + int(os.environ["SLURM_JOB_ID"]) % 10000)

    # Initialize process group
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)

    # Summary
    if is_main_process():
        PREFIX = "%i - " % rank
        print(PREFIX + "Number of nodes: %i" % int(os.environ["SLURM_JOB_NUM_NODES"]))
        print(PREFIX + "Node ID        : %i" % int(os.environ["SLURM_NODEID"]))
        print(PREFIX + "World size     : %i" % world_size)
        print(PREFIX + "GPUs per node  : %i" % int(os.environ["SLURM_GPUS_ON_NODE"]))
        print(PREFIX + "Local rank     : %i" % local_rank)
        print(PREFIX + "Master node    : %s" % master_addr)
        print(PREFIX + "Hostname       : %s" % socket.gethostname())
        print(PREFIX + "Port           : %s" % os.environ["MASTER_PORT"])

    return torch.device(f'cuda:{local_rank}')

device = setup()


# 2. Process command-line arguments

def parse_args() -> Namespace:
    parser = ArgumentParser()

    # Training related arguments
    parser.add_argument('--batch-size', default=1, type=int, help='Batch size per GPU')
    parser.add_argument("--seq-length", default=4096, type=int, help='Sequence length of each sample per GPU')
    parser.add_argument('--epochs', default=2, type=int, help='Number of epochs')
    parser.add_argument('--grad-acc', default=1, type=int, help='Number of batches used for a single weights update')

    # Benchmarking
    parser.add_argument('--test', default=False, action=BooleanOptionalAction, help='Test 100 iterations')
    parser.add_argument('--test-nsteps', default=100, type=int, help='The number of steps in test mode.')
    parser.add_argument("--optimizer-precision", default=False, action=BooleanOptionalAction, help="Whether to print precision of optimizer states item.")
    parser.add_argument("--cpu-usage", default=False, action=BooleanOptionalAction, help="Whether to print CPU memory Usage.")

    # JIT related arguments
    parser.add_argument("--compile", default=False, action=BooleanOptionalAction, help="whether or not to compile model")

    # DataLoader related arguments
    parser.add_argument('--dataset-path', type=Path, help="HuggingFaceHub dataset's name.")
    parser.add_argument('--num-workers', default=4, type=int, help='Number of workers spawned by the dataloader')
    parser.add_argument('--prefetch-factor', default=2, type=int, help='Number of batches loaded in RAM per worker.')
    parser.add_argument('--persistent-workers', default=True, action=BooleanOptionalAction, help='Workers persist after the end of an epoch.')

    # Optimizer AdamW related arguments
    parser.add_argument("--lr-warmup-ratio", default=0.1, type=float, help="Fraction of training steps for linear LR warmup (0.1 = 10%).")
    parser.add_argument("--learning-rate", default=1e-5, type=float, help="Learning rate for AdamW.")
    parser.add_argument("--weight-decay", default=0.1, type=float, help="Weight decay for AdamW.")

    # Model related arguments
    parser.add_argument("--model-path", type=Path, help="HuggingFaceHub model's name.")
    parser.add_argument("--selective-activation-checkpointing", default=None, type=str, help='For a given ac ratio p, we should essentially apply ac on every "1/p" blocks.')

    return parser.parse_args()

args = parse_args()


if is_main_process():
    chrono = Chronometer()

    print(f"Global batch size                       : {args.batch_size*args.grad_acc*get_world_size()}")
    print(f"Gradient accumulation                   : {args.grad_acc}")
    print(f"Mini batch size (per GPUs)              : {args.batch_size}")
    print(f"Sequence length                         : {args.seq_length}")
    print(f"Selective activation checkpointing ratio: {args.selective_activation_checkpointing}")
    print(f"Compile                                 : {args.compile}")


# 3. Model processing

if is_main_process(): chrono.timer(start=True)
model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="bfloat16", trust_remote_code=True)
num_parameters = sum(param.numel() for param in model.parameters())
tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
if is_main_process(): print(f"Time to load the model and its tokenizer: {chrono.timer(start=False):.3f} s")



if args.selective_activation_checkpointing:
    model.config.use_cache = False  # deactivate KV caching (conflicts with activation checkpointing)
    BlockCls = type(model.model.layers[0])
    apply_fsdp_checkpointing(model, BlockCls, args.selective_activation_checkpointing)




#### Distribute the Model
if RANK == 0: chrono.timer(start=True)

fsdp_kwargs = {}
fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )

for layer in model.model.layers:
    fully_shard(layer.type(torch.float32), **fsdp_kwargs)
fully_shard(model.type(torch.float32), **fsdp_kwargs)

if RANK == 0: print(f"Time to shard the model: {chrono.timer(start=False):.3f}s")

# Transfer to GPU
model = model.to(device, non_blocking=args.non_blocking)

if RANK == 0: print(f"Time to transfer the model to GPU: {chrono.timer(start=False):.3f}s")

#### JIT
if args.compile:
    model = torch.compile(model)

    if RANK == 0: print(f"Time to instantiate torch.compile: {chrono.timer(start=False):.3f}s")
####

if RANK == 0:
    #print(f"model: {model}")
    print(f"number of parameters: {num_parameters}")
    print(f'Pre-loop Model MaxMemory for GPU:{RANK} {torch.cuda.max_memory_allocated()/2**30} GBytes')

"""
#### Data Loading
if RANK == 0: chrono.timer(start=True)
train_dataset = load_dataset("parquet", data_files=str(dataset_path) + '/*.parquet', split="train")  # 
collate_fn = make_sft_collate(tokenizer, max_seq_length=args.seq_length)

sampler = DistributedSampler(
    dataset=train_dataset,
    rank=RANK,
    num_replicas=WORLD_SIZE,
    shuffle=True,
)

dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
    pin_memory=True,
    persistent_workers=args.persistent_workers,
    prefetch_factor=args.prefetch_factor,
    sampler=sampler,
)
####

if RANK == 0: print(f"Time to load dataset and initialize dataloader: {chrono.timer(start=False):.3f}s")

#### Training step
criterion = CrossEntropyLoss(ignore_index=-100)
optimizer = AdamW(
    params=model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
    eps=1e-05,
)

if RANK == 0:
    print(f'global batch size: {args.batch_size * WORLD_SIZE} - mini batch size: {args.batch_size}')
    print(f"DATALOADER {args.num_workers} {args.persistent_workers} {args.prefetch_factor}")
    print(f"Optimizer: {optimizer}")

lr_warmup_iters = int(len(dataloader) * args.lr_warmup_ratio)  * args.epochs / args.grad_acc
warmup_lr_scheduler = LinearLR(
    optimizer,
    start_factor=1e-9,
    end_factor=1,
    total_iters=lr_warmup_iters,
)
annealing_lr_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=len(dataloader) * args.epochs / args.grad_acc - lr_warmup_iters,
    eta_min=0.,
)
lr_scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_lr_scheduler,annealing_lr_scheduler],
    milestones=[lr_warmup_iters]
)

loss_metric = RunningMean(window=5).to(device)
perplexity = Perplexity(ignore_index=-100).to(device)


#### Training loop
if args.test: chrono.start_training()
    
if RANK == 0: chrono.timer(start=True)

if args.test: args.epochs = 1 #Test with only 100 steps
for epoch in range(args.epochs):
    #set epoch for sampler
    sampler.set_epoch(epoch)
    if args.test: chrono.dataload()
    for i, (input_ids, attention_mask, labels) in enumerate(dataloader, start=1):
        if args.test and i > args.test_nsteps * args.grad_acc: break
    
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=args.non_blocking)
    
        if args.test:
            chrono.dataload()
            chrono.forward()
    
        # passes and weights update
        logits: torch.Tensor = model(input_ids, attention_mask=attention_mask).logits
        bsz, seq_len, vocab_size = logits.shape
        loss: torch.Tensor = criterion(logits.view(bsz * seq_len, vocab_size), labels.view(bsz * seq_len))
        loss /= WORLD_SIZE
        loss /= args.grad_acc
        
        loss_metric.update(loss)
        perplexity.update(logits, labels)
        
        if args.test: 
            chrono.forward()
            chrono.backward()
            
        loss.backward()
        
        if i % args.grad_acc == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        if args.test:
            chrono.backward()
        
        step = (i // args.grad_acc) + 1
        if step % 10 == 0 and i % args.grad_acc == 0:
            L = loss_metric.compute()
            perp = perplexity.compute()
            last_lr = lr_scheduler.get_last_lr()[0]
            if RANK == 0:
                print(f"Step {step} / {args.test_nsteps if args.test else len(dataloader) // args.grad_acc} | Loss: {L.item():.3f} | Perplexity: {perp.item():.3f} | LR: {last_lr:0.3e} | Wall: {chrono.tac_time():.3f}")

        if i==1 and RANK == 0: print(f"Time to first step - compile Graph Building: {chrono.timer(start=False):.3f}s")
####

    ######### Model Checkpointing at each epoch ############
    
    if not args.test:
        ckeckpoint_name =  f"model_state_dict_{os.environ['SLURM_JOB_ID']}_epoch{epoch}.pt"
        print(f"Model Checkpointing - Building the {ckeckpoint_name} file")
        model_state_dict = get_model_state_dict(
            model=model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            )
        )
        torch.save(model_state_dict, ckeckpoint_name)



    ###############################

if args.test: chrono.display_training_results()

dist.barrier()
if RANK == 0:
    print(f'MaxMemory for GPU:{RANK} {torch.cuda.max_memory_allocated()/2**30} GBytes')

if args.optimizer_precision and RANK == 0:
    for k, v in optimizer.state.items():
        print(k, {kk: vv.dtype for kk, vv in v.items()})
        break

if args.cpu_usage and RANK == 0:
    print(f"RANK: {RANK}")
    memory_usage()



dist.barrier()
dist.destroy_process_group()
"""