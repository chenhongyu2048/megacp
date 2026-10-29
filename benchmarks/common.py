"""CUDA-event timing with per-sample rank-max aggregation."""

import argparse
import math
import os
import statistics


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def initialize():
    import torch
    import torch.distributed as dist

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError("Mega-CP requires Hopper SM90 GPUs")
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != int(os.environ["LOCAL_WORLD_SIZE"]) or world not in (2, 4, 8):
        raise ValueError("run on one node with 2, 4, or 8 processes")
    torch.manual_seed(2026 + rank)
    return device, rank, world


def event_sampler(operation):
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def sample():
        start.record()
        operation()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    return sample


def measure(sample, warmup, iters, device):
    import torch
    import torch.distributed as dist

    for _ in range(warmup):
        sample()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(iters):
        dist.barrier(device_ids=[device.index])
        samples.append(sample())
        torch.cuda.synchronize(device)
    values = torch.tensor(samples, device=device, dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    samples = sorted(values.cpu().tolist())
    return {
        "iterations": iters,
        "aggregation": "per_sample_max_across_ranks",
        "mean_ms": statistics.mean(samples),
        "p50_ms": statistics.median(samples),
        "p90_ms": samples[math.ceil(0.9 * len(samples)) - 1],
    }
