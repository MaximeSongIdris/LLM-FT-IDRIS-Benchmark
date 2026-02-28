from typing import Any

from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning.pytorch as pl
import torch

from .chrono import TrainingChronometer
from .cpu_mem_usage import memory_usage


class MyChronoCallback(pl.Callback):
    def __init__(self, rank: int, n_steps: int, global_batch_size: int, seq_len: int, total_batches_per_epoch: int, grad_acc: int = 1) -> None:
        self.rank = rank
        self.n_steps = n_steps
        self.global_batch_size = global_batch_size
        self.seq_len = seq_len
        self.total_batches_per_epoch = total_batches_per_epoch
        self.grad_acc = grad_acc
        self.chronometer = TrainingChronometer()

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.rank == 0:
            print(f'Pre-loop GPU memory usage: {torch.cuda.max_memory_allocated()/2**30:.2f} GB')

            # Check parameter sharding across devices
            for name, param in pl_module.named_parameters():
                if hasattr(param, 'device_mesh'):
                    print(f"{name}:")
                    print(f"  mesh: {param.device_mesh}")
                    print(f"  shape: {param.device_mesh.shape}")
                    print(f"  placements: {param.placements}")

            self.chronometer.track_cpu_training_time(start=True)

    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        if self.rank == 0:
            self.chronometer.track_gpu_HtoD_step_time(start=True)  # no hook before forward to observe data transfer
            self.chronometer.track_gpu_HtoD_step_time(start=False)
            self.chronometer.track_gpu_fwd_step_time(start=True)

    def on_before_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", loss: torch.Tensor) -> None:
        if self.rank == 0:
            self.chronometer.track_gpu_fwd_step_time(start=False)
            self.chronometer.track_gpu_bwd_step_time(start=True)

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if self.rank == 0:
            self.chronometer.track_gpu_bwd_step_time(start=False)

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.rank == 0:
            ## Training results
            self.chronometer.track_cpu_training_time(start=False)
            training_duration = self.chronometer.display_training_results(self.total_batches_per_epoch, self.grad_acc)
            print(f'Throughput: {self.n_steps*self.global_batch_size*self.seq_len/training_duration:.1f} tokens/s')

            ## Memory Usage
            memory_usage()
            print(f'Post-loop GPU memory usage: {torch.cuda.max_memory_allocated()/2**30} GBytes')
