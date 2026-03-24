import json
import numpy as np

from .analyze_sync_pytorch_profiler import get_gpu_step_info_from_trace, categorize_cuda_event


def compute_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> tuple[float, float, float]:
    """Intersection between two intervals [a_start, a_end) and [b_start, b_end)."""
    inter_end = min(a_end, b_end)
    inter_start = max(a_start, b_start)
    o = inter_end - inter_start 
    return max(0.0, o), inter_start, inter_end

def classify_default_stream_idle_gaps(compute_events: list, nccl_events: list, step_start: float, step_end: float) -> dict:
    """For each idle gap on the compute stream, determine how much is communication-bound vs CPU-bound.
 
    Args:
        compute_events: sorted list of events on the compute stream (stream 7)
        nccl_events:    sorted list of NCCL events on other streams
        step_start:     step start timestamp (ms)
        step_end:       step end timestamp (ms)
    """
    ## Build list of idle gaps on the compute stream
    gaps = []
 
    # Gap before first event
    if compute_events[0]["ts"] > step_start:
        gaps.append((step_start, compute_events[0]["ts"]))
 
    # Gaps between consecutive events
    for i in range(len(compute_events) - 1):
        gap_start = compute_events[i]["ts"] + compute_events[i]["dur"]
        gap_end = compute_events[i + 1]["ts"]
        if gap_end > gap_start:
            gaps.append((gap_start, gap_end))
 
    # Gap after last event
    last_end = compute_events[-1]["ts"] + compute_events[-1]["dur"]
    if last_end < step_end:
        gaps.append((last_end, step_end))


    ## For each gap, compute how much is covered by NCCL on other streams
    total_comm_bound = 0.0
    total_cpu_bound = 0.0
    gap_details = []
 
    for gap_start, gap_end in gaps:
        gap_dur = gap_end - gap_start
        if gap_dur <= 0:
            continue
 
        # Filter NCCL events that is within this gap
        nccl_in_gap = []
        for nccl_ev in nccl_events:
            nccl_start = nccl_ev["ts"]
            nccl_end = nccl_ev["ts"] + nccl_ev["dur"]
            o, inter_start, inter_end = compute_overlap(gap_start, gap_end, nccl_start, nccl_end)
            if o > 0:
                nccl_in_gap.append((inter_start, inter_end))
 
        # Merge overlapping NCCL intervals within this gap
        nccl_in_gap.sort(key=lambda x: x[0])
        merged = []
        for start, end in nccl_in_gap:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
 
        comm_time = sum(end - start for start, end in merged)
        cpu_time = gap_dur - comm_time
 
        total_comm_bound += comm_time
        total_cpu_bound += cpu_time
 
        gap_details.append({
            "start": gap_start,
            "end": gap_end,
            "duration": gap_dur,
            "comm_bound": comm_time,
            "cpu_bound": cpu_time,
        })
 
    return {
        "comm_bound": total_comm_bound,
        "cpu_bound": total_cpu_bound,
        "total_idle": total_comm_bound + total_cpu_bound,
        "gaps": gap_details,
    }

def parse_overlap_trace(profiler_file: str='profile/xp/jzxh018_3913107.1774026182903860725.pt.trace.json') -> tuple[list, list, list]:
    """Parse a PyTorch profiler trace JSON and extract per-step GPU events.

    Reads the Chrome Trace format exported by torch.profiler, identifies training steps via gpu_user_annotation events, 
    assigns each GPU event to its corresponding step, and computes the idle latency between consecutive GPU events.
    Assumes asynchronous execution (default CUDA behavior).
    Separates compute stream events from NCCL communication events to enable overlap analysis.
    """
    ## LOAD PROFILE
    with open(profiler_file, "r") as f:
        trace = json.load(f)
    
    
    ## GET STEP DURATION
    gpu_step_annotations = get_gpu_step_info_from_trace(trace)


    ## GET EVENTS PER STEP
    ts_per_step = [event['ts'] for event in gpu_step_annotations]

    # Categorize compute stream events per STEP and NCCL events per STEP
    default_stream_event_per_step = [[] for event in gpu_step_annotations]
    communication_stream_event_per_step = [[] for event in gpu_step_annotations]
    for event in trace["traceEvents"]:
        if event.get("args", {}).get("stream", -1) == 7 and event.get("cat", "") in ["kernel", "gpu_memcpy", "gpu_memset"]:
            idx = max((i for i, v in enumerate(ts_per_step) if event['ts'] >= v), default=-1)
            if idx != -1:
                default_stream_event_per_step[idx].append(event)
        elif event.get("args", {}).get("stream", -1) != -1 and event.get("cat", "") == "kernel" and "nccl" in event.get("name", ""):
            idx = max((i for i, v in enumerate(ts_per_step) if event['ts'] >= v), default=-1)
            if idx != -1:
                communication_stream_event_per_step[idx].append(event)

    # Sort by events order
    for (default_stream_event, comm_stream_event) in zip(default_stream_event_per_step, communication_stream_event_per_step):
        default_stream_event.sort(key=lambda x: x['ts'])
        comm_stream_event.sort(key=lambda x: x['ts'])

    # Compute latency between events on GPU
    for step in range(len(default_stream_event_per_step)-1):  # skip the last training step
        step_default_event = default_stream_event_per_step[step]
        for i in range(len(step_default_event)-1):
            step_default_event[i]['latency'] = step_default_event[i+1]['ts'] - (step_default_event[i]['ts'] + step_default_event[i]['dur'])
        step_default_event[-1]['latency'] = gpu_step_annotations[step+1]['ts'] - (step_default_event[-1]['ts'] + step_default_event[-1]['dur'])

        step_comm_event = communication_stream_event_per_step[step]
        for i in range(len(step_comm_event)-1):
            step_comm_event[i]['latency'] = step_comm_event[i+1]['ts'] - (step_comm_event[i]['ts'] + step_comm_event[i]['dur'])
        step_comm_event[-1]['latency'] = gpu_step_annotations[step+1]['ts'] - (step_comm_event[-1]['ts'] + step_comm_event[-1]['dur'])


    ## PROCESSING
    # Remove last step
    gpu_step_annotations = gpu_step_annotations[:-1]
    default_stream_event_per_step = default_stream_event_per_step[:-1]
    communication_stream_event_per_step = communication_stream_event_per_step[:-1]

    # Convert microseconds to millisecond
    for event in gpu_step_annotations:
        event['ts'] = event['ts'] / 1000
        event['dur'] = event['dur'] / 1000

    for step_event in default_stream_event_per_step + communication_stream_event_per_step:
        for event in step_event:
            event['ts'] = event['ts'] / 1000
            event['dur'] = event['dur'] / 1000
            event['latency'] = event['latency'] / 1000

    return gpu_step_annotations, default_stream_event_per_step, communication_stream_event_per_step

def analyze_overlap_step_breakdown(gpu_step_annotations: list,
                                   default_stream_event_per_step: list,
                                   communication_stream_event_per_step: list,
                                   step_idx: int = None) -> dict:
    """Compute time breakdown for a training step with overlap analysis.

    Sums event durations by category and classifies idle gaps on the default stream
    as either communication-bound (hidden by NCCL on other streams) or CPU-bound.
    Assumes asynchronous execution (default CUDA behavior).

    Categories:
        cpu_bound:        Idle time on default stream not overlapped by NCCL.
                          Captures Python/PyTorch overhead, kernel launch latency.
                          High values suggest CPU bottleneck.
        cpu_gpu_transfer: Data movement over PCIe (HtoD, DtoH).
        compute:          CUDA kernels excluding NCCL (can overlap with communication).
        comm_overhead:    Tensor processing for NCCL collectives on default stream.
                          Cannot overlap with compute.
        comm_bound:       Idle time on default stream overlapped by NCCL.
        other:            GPU-to-GPU memcpy, memset, etc.
    """
    # Select step with median duration if not specified
    if step_idx is None:
        median_dur = np.median([annotation["dur"] for annotation in gpu_step_annotations])
        step_idx = int(np.argmin([abs(annotation["dur"] - median_dur) for annotation in gpu_step_annotations]))
    
    # Classify idle gaps in default stream
    step_start = gpu_step_annotations[step_idx]["ts"]
    step_end = step_start + gpu_step_annotations[step_idx]["dur"]
    
    idle_classification = classify_default_stream_idle_gaps(
        compute_events=default_stream_event_per_step[step_idx],
        nccl_events=communication_stream_event_per_step[step_idx],
        step_start=step_start,
        step_end=step_end,
    )

    # Categorize default stream events
    events_by_category = {
        "cpu_gpu_transfer": [],
        "compute": [],
        "comm_overhead": [],
        "other": [],
    }
    for event in default_stream_event_per_step[step_idx]:
        events_by_category[categorize_cuda_event(event)].append(event)

    print(f"{gpu_step_annotations[step_idx]['name']}:")
    print(f"  CPU-bound:                               {idle_classification['cpu_bound']:8.1f} ms")
    print(f"  CPU-GPU transfer:                        {sum(event['dur'] for event in events_by_category['cpu_gpu_transfer']):8.1f} ms")
    print(f"  Compute (overlapped with communication): {sum(event['dur'] for event in events_by_category['compute']):8.1f} ms")
    print(f"  Communication overhead:                  {sum(event['dur'] for event in events_by_category['comm_overhead']):8.1f} ms")
    print(f"  Communication-bound:                     {idle_classification['comm_bound']:8.1f} ms")
    print(f"  Other:                                   {sum(event['dur'] for event in events_by_category['other']):8.1f} ms")
    print(f"  STEP TOTAL:                              {gpu_step_annotations[step_idx]['dur']:8.1f} ms")

    return {
        "step": gpu_step_annotations[step_idx]["name"],
        "cpu_bound": idle_classification["cpu_bound"],
        "cpu_gpu_transfer": sum(event["dur"] for event in events_by_category["cpu_gpu_transfer"]),
        "compute": sum(event["dur"] for event in events_by_category["compute"]),
        "comm_overhead": sum(event["dur"] for event in events_by_category["comm_overhead"]),
        "comm_bound": idle_classification["comm_bound"],
        "other": sum(event["dur"] for event in events_by_category["other"]),
        "step_duration": gpu_step_annotations[step_idx]["dur"],
    }
