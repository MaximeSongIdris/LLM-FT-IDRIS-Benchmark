from datetime import datetime
from time import time
import numpy as np


class TrainingChronometer:
    def __init__(self) -> None:
        self._reset_state()

    def reset(self) -> None:
        self._reset_state()

    def _reset_state(self) -> None:
        self.training_duration = None
        self.time_perf_dataloading = []
        self.time_perf_forward = []
        self.time_perf_backward = []
        
        self.saved_time = None
        self.start_training_time = None
        self.start_dataloading_time = None
        self.start_forward_time = None
        self.start_backward_time = None
    
    def timer(self, start: bool=True) -> None | float:
        """Start or stop a simple wall-clock timer.

        When `start` is True, records the current time as the reference.
        When `start` is False, returns the elapsed time since the last reference
        and updates the reference to the current time.
        """
        if start:
            self.saved_time = time()
        else:
            if self.saved_time is None:
                raise RuntimeError("Timer was not started. Call timer(start=True) first.")
            else:
                previous_time = self.saved_time
                self.saved_time = time()
                return self.saved_time - previous_time
        
    def track_training_time(self, start: bool=True) -> None | float:
        """Start or stop a wall-clock training timer."""
        if start:
            self.start_training_time = datetime.now()
        else:
            if self.start_training_time is None:
                raise RuntimeError("Training timer was not started. Call track_training_time(start=True) first.")
            else:
                self.training_duration = datetime.now() - self.start_training_time
                self.start_training_time = None
            
    def track_dataloading_step_time(self, start: bool=True) -> None | float:
        """Start or stop a wall-clock data loading timer."""
        if start:
            self.start_dataloading_time = time()
        else:
            if self.start_dataloading_time is None:
                raise RuntimeError("Dataloading timer was not started. Call track_dataloading_step_time(start=True) first.")
            else:
                self.time_perf_dataloading.append(time() - self.start_dataloading_time)
                self.start_dataloading_time = None
                
    def track_forward_step_time(self, start: bool=True) -> None | float:
        """Start or stop a wall-clock forward pass timer."""
        if start:
            self.start_forward_time = time()
        else:
            if self.start_forward_time is None:
                raise RuntimeError("Forward timer was not started. Call track_forward_step_time(start=True) first.")
            else:
                self.time_perf_forward.append(time() - self.start_forward_time)
                self.start_forward_time = None
                
    def track_backward_step_time(self, start: bool=True) -> None | float:
        """Start or stop a wall-clock backward pass timer."""
        if start:
            self.start_backward_time = time()
        else:
            if self.start_backward_time is None:
                raise RuntimeError("Backward timer was not started. Call track_backward_step_time(start=True) first.")
            else:                
                self.time_perf_backward.append(time() - self.start_backward_time)
                self.start_backward_time = None

    def print_percentile_summary(self, data: list, name: str) -> None:
        """Helper function to display sample statistics.
        
        Shows the variability in the actual measurements using percentiles.
        """
        if len(data) >= 10:
            # Use percentiles since it works for any distribution
            p0 = np.min(data)
            p10 = np.percentile(data, 10)
            p50 = np.median(data)
            p90 = np.percentile(data, 90)
            p100 = np.max(data)

            print(f">>> {name}: min {p0:.3f} s | 10th percentile {p10:.3f} s | median {p50:.3f} s | 90th percentile {p90:.3f} s | max {p100:.3f} s")
        else:
            print(f">>> {name}: insufficient data (n={len(data)})")
                
    def display_training_results(self, total_batches: int, grad_acc: int) -> None:
        # Sample statistics
        print(">>> Training complete in: " + str(self.training_duration))
        self.print_percentile_summary(self.time_perf_dataloading[1:], "Data loading performance time")
        self.print_percentile_summary(self.time_perf_forward[1:], "Forward pass performance time")
        self.print_percentile_summary(self.time_perf_backward[1:], "Backward pass performance time")

        # Total training time estimation
        time_perf_train = [x + y + z for (x,y,z) in zip(self.time_perf_dataloading, self.time_perf_forward, self.time_perf_backward)]
        self.print_percentile_summary(time_perf_train[1:], "Step performance time")
        print(f'>>> Number of weight updates per epoch: {(total_batches + grad_acc - 1) // grad_acc}')
        print(f'>>> Estimated training time of 1 epoch: {np.median(time_perf_train[1:]) * total_batches / 3600} h')
