"""
Author: Bertrand CABOT from IDRIS(CNRS)
"""

from typing import Any

from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning.pytorch as pl
import torch

from .chrono import TrainingChronometer
from .cpu_mem_usage import memory_usage


class MyChronoCallback(pl.Callback):

    def __init__(self, rank: int, total_batches: int, grad_acc: int = 1) -> None:
        self.rank = rank
        self.total_batches = total_batches
        self.grad_acc = grad_acc
        self.chronometer = TrainingChronometer()

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.rank == 0:
            print(f'Pre-loop GPU memory usage: {torch.cuda.max_memory_allocated()/2**30:.2f} GB')
        self.chronometer.track_training_time(start=True)
        self.chronometer.track_dataloading_step_time(start=True)

    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        self.chronometer.track_dataloading_step_time(start=False)
        self.chronometer.track_forward_step_time(start=True)

    def on_before_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", loss: torch.Tensor) -> None:
        self.chronometer.track_forward_step_time(start=False)
        self.chronometer.track_backward_step_time(start=True)

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        self.chronometer.track_backward_step_time(start=False)
        self.chronometer.track_dataloading_step_time(start=True)

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.chronometer.track_training_time(start=False)
        self.chronometer.display_training_results(self.total_batches, self.grad_acc)

        if self.rank == 0:
            memory_usage()
            print(f'Post-loop GPU memory usage: {torch.cuda.max_memory_allocated()/2**30} GBytes')
        