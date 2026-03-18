import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def wire_bytes_per_gpu(coll_op: str, nccl_count: int, datatype: int, nranks: int):
    """
    Estimate the number of bytes transferred per GPU for a given NCCL collective.

    The computation follows the same conventions as NCCL-tests bandwidth calculations.

    Source: https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#bandwidth
    Source: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/mpi.html#send-and-receive-counts
    Source: https://github.com/NVIDIA/nccl/blob/master/contrib/nccl_ep/python/nccl_ep/nccl_wrapper.py#L56
    """
    N = nranks
    
    # `count` in NCCL logs are either sendcount or recvcount
    # sendcount is the number of elements in sendbuff involved for this collective
    # recvcount is the number of elements in recvbuff involved for this collective
    # sometimes sendcount != recvcount, but what we want is max(sendcount,recvcount)
    count_factor = {
        'AllReduce':     1,
        'ReduceScatter': N,
        'AllGather':     N,
        'AlltoAll':      1,
        "Send":          1,
        "Recv":          1,
    }

    # Convert element counts to bytes
    datatype_factor = {
        0: 1,  # ncclInt8
        1: 1,  # ncclUint8
        2: 4,  # ncclInt32
        4: 8,  # ncclInt64
        6: 2,  # ncclFloat16
        7: 4,  # ncclFloat32
        8: 8,  # ncclFloat64
        9: 2,  # ncclBfloat16
    }

    # Ratio between size of the data involved in the collective and the average volume of data transfer per GPU
    bus_bandwidth_factor = {
        'AllReduce':     2 * (N - 1) / N,
        'ReduceScatter': (N - 1) / N,
        'AllGather':     (N - 1) / N,
        'AlltoAll':      (N - 1) / N,
        "Send":          1,
        "Recv":          1,
    }

    return nccl_count * count_factor[coll_op] * datatype_factor[datatype] * bus_bandwidth_factor[coll_op]

def comm_profiler(log_files: list[str]) -> dict:
    """
    Parse NCCL INFO logs for a single GPU and visualize collective communication volumes.

    Expects a log file where training markers are interleaved with NCCL traces via `NCCLTagger`:

        >>> --- Step 1 --- Phase Forward
        [2026-03-12 13:57:45] host:pid:tid [rank] NCCL INFO AllGather: opCount ...
        >>> --- Step 1 --- Phase Backward
        ...
    """

    coll_dict = {
        'timestamp': [],
        'rank': [],
        'coll_operation': [],
        'comm_per_gpu': [],
        'datatype': [],
        'op': [],
        'nranks': [],
        'train_step': [],
        'phase': [],
    }

    for log_file in log_files:
        current_step = None   # current training step
        current_phase = None  # current phase in the training step

        rank = None           # rank in the nccl communicator
        nranks = None         # total ranks in the nccl communicator
        with open(log_file, "r") as f:
            for line in f:

                ## DETERMINE FIRST NCCL CONFIG
                if 'ncclCommInitRankConfig' in line:
                    tokens = line.strip().split()
                    rank = int(tokens[tokens.index("rank") + 1])
                    nranks = int(tokens[tokens.index("nranks") + 1])
                if rank is None:
                    continue

                ## THEN FIND THE USER-DEFINED TAG
                # Training tag: >>> --- Step 1 --- Phase Forward
                if line.startswith(">>> --- Step"):
                    # tokens: ['>>>', '---', 'Step', '1', '---', 'Phase', 'Forward']
                    tokens = line.strip().split()
                    current_step = int(tokens[3])
                    current_phase = tokens[6]
                    continue
                if current_step is None:
                    continue

                # Example line:
                # [2026-03-12 13:57:49] jzxh115:3915132:3917207 [3] NCCL INFO ReduceScatter: opCount 0 sendbuff 0x14df1c000000 recvbuff 0x14e100000000 count 58264448 ...
                if "count" not in line:
                    continue

                tokens = line.split()
    
                timestamp = tokens[0][1:] + " " + tokens[1][:-1]  # "2026-03-12 13:57:45"
                coll_op = tokens[6][:-1]                          # AllGather: -> AllGather
                count = int(tokens[tokens.index("count") + 1])
                datatype = int(tokens[tokens.index("datatype") + 1])
                op = int(tokens[tokens.index("op") + 1])
    
                coll_dict['timestamp'].append(timestamp)
                coll_dict['rank'].append(rank)
                coll_dict['coll_operation'].append(coll_op)
                coll_dict['comm_per_gpu'].append(wire_bytes_per_gpu(coll_op, count, datatype, nranks))
                coll_dict['datatype'].append(datatype)
                coll_dict['op'].append(op)
                coll_dict['nranks'].append(nranks)
                coll_dict['train_step'].append(current_step)
                coll_dict['phase'].append(current_phase)

    return coll_dict

def _print_percentile_summary(data: list, name: str) -> None:
    """Display distribution statistics for a list of volume measurements.
    
    Uses percentiles (min, p10, median, p90, max) to show the variability of the measurements.
    Requires at least 10 data points.
    """
    if len(data) >= 10:
        # Use percentiles since it works for any distribution
        p0 = np.min(data)
        p10 = np.percentile(data, 10)
        p50 = np.median(data)
        p90 = np.percentile(data, 90)
        p100 = np.max(data)

        print(f">>> {name}: min {p0/1e9:.1f} Go | 10th percentile {p10/1e9:.1f} Go | median {p50/1e9:.1f} Go | 90th percentile {p90/1e9:.1f} Go | max {p100/1e9:.1f} Go")
    else:
        print(f">>> {name}: insufficient data (n={len(data)})")

def get_comm_results(coll: dict, skip_steps: int=0) -> pd.DataFrame:
    df = pd.DataFrame(coll)
    df = df[df['train_step'] > skip_steps].reset_index()

    # Group by `train_step` and `phase`
    df_1gpu = df[df['rank'] == 0].copy()
    df_1gpu = df_1gpu.groupby(['train_step', 'phase'], sort=False)['comm_per_gpu'].sum().reset_index()


    # Communication volumes estimation
    phases = df_1gpu['phase'].unique()
    phase_volumes = {}
    for phase in phases:
        phase_volumes[phase] = df_1gpu[df_1gpu['phase'] == phase]['comm_per_gpu'].tolist()
        _print_percentile_summary(phase_volumes[phase], f"{phase} communication volume per step")

    # Total communication volumes estimation
    min_len = min(len(v) for v in phase_volumes.values())
    comm_vol_step = [
        sum(phase_volumes[phase][i] for phase in phases)
        for i in range(min_len)
    ]
    _print_percentile_summary(comm_vol_step, "Total communication volume per step")

    return df_1gpu

def plot_comm_profiler(coll: dict, n_display=50) -> None: 

    df = pd.DataFrame(coll)

    # Check if df is empty 
    assert not df.empty, "No NCCL collective operations found in the log file."

    # Check that every rank has the same number of collective operations
    counts = df.groupby('rank').size()
    assert counts.nunique() == 1, f"Mismatch in collective counts per rank: {counts.to_dict()}"

    df_plot = pd.DataFrame()
    for r in df['rank'].unique():
        df_plot[f'rank: {r}'] = df[df['rank'] == r].comm_per_gpu.reset_index(drop=True)
    
    # Build operation labels
    nccldtype = {
        0: 'ncclInt8',
        1: 'ncclUint8',
        2: 'ncclInt32',
        3: 'ncclUint32',
        4: 'ncclInt64',
        5: 'ncclUint64',
        6: 'ncclFloat16',
        7: 'ncclFloat32',
        8: 'ncclFloat64',
        9: 'ncclBfloat16',
    }
    rank0 = df[df['rank'] == 0].reset_index(drop=True)

    df_plot['label'] = (
        "Step "
        + rank0['train_step'].astype(str)
        + " - "
        + rank0['phase']
        + " - "
        + rank0['coll_operation']
        + " ("
        + rank0['datatype'].map(nccldtype).astype(str)
        + ")"
    )
    df_plot = df_plot.set_index('label')

    # --- Per-operation bar plot ---
    plot_title = f'Collective Communication Profiler - Total operations: {len(df)}'

    if n_display:
        df_plot.iloc[:n_display].plot.bar(
            figsize=(18, 6), rot=90,
            title=f'{plot_title} (first {n_display})',
            ylabel='Bytes',
        )
        plt.tight_layout()
        plt.show()
        
        df_plot.iloc[-n_display:].plot.bar(
            figsize=(18, 6), rot=90,
            title=f'{plot_title} (last {n_display})',
            ylabel='Bytes',
        )
        plt.tight_layout()
        plt.show()
    else:
        df_plot.plot.bar(
            figsize=(18, 6), rot=90,
            title=plot_title,
            ylabel='Bytes',
        )
        plt.tight_layout()
        plt.show()

    # --- Aggregate bar plot grouped by label ---
    dfagg = df_plot.groupby('label', sort=False).sum()

    if n_display:
        dfagg.iloc[:n_display].plot.bar(
            figsize=(18, 6), rot=90,
            title=f'Aggregate Collective Communication Profiler - Total volume: {df[df['rank'] == 0].comm_per_gpu.sum()} Bytes/GPU (first {n_display})',
            ylabel='Bytes',
        )
        plt.tight_layout()
        plt.show()
        
        dfagg.iloc[-n_display:].plot.bar(
            figsize=(18, 6), rot=90,
            title=f'Aggregate Collective Communication Profiler - Total volume: {df[df['rank'] == 0].comm_per_gpu.sum()} Bytes/GPU (last {n_display})',
            ylabel='Bytes',
        )
        plt.tight_layout()
        plt.show()
    else:
        dfagg.plot.bar(
            figsize=(18, 6), rot=90,
            title=f'Aggregate Collective Communication Profiler - Total volume: {df[df['rank'] == 0].comm_per_gpu.sum()} Bytes/GPU',
            ylabel='Bytes',
        )
        plt.tight_layout()
        plt.show()

    # --- Total communication per step per phase ---
    dfphase = df_plot.copy().reset_index(drop=True)  # vire l'index 'label'
    
    rank0 = df[df['rank'] == 0].reset_index(drop=True)
    dfphase['step_phase'] = (
        "Step "
        + rank0['train_step'].astype(str)
        + " - "
        + rank0['phase']
    )
    dfphase = dfphase.groupby('step_phase', sort=False).sum()

    if n_display:
        dfphase.iloc[:n_display].plot.bar(
            figsize=(18, 6), rot=90,
            title=f'Total Communication per Step per Phase (first {n_display})',
            ylabel='Bytes',
        )
        plt.tight_layout()
        plt.show()
        
        dfphase.iloc[-n_display:].plot.bar(
            figsize=(18, 6), rot=90,
            title=f'Total Communication per Step per Phase (last {n_display})',
            ylabel='Bytes',
        )
        plt.tight_layout()
        plt.show()
    else:
        dfphase.plot.bar(
            figsize=(18, 6), rot=90,
            title='Total Communication per Step per Phase',
            ylabel='Bytes',
        )
        plt.tight_layout()
        plt.show()

    return df