"""Benchmark MegaDCP eager kernel time or full CUDA Graph replay."""

import argparse
import json

from .common import event_sampler, initialize, measure, positive_int


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--query-length", type=positive_int, default=16)
    parser.add_argument("--history-length", type=positive_int, default=16384,
                        help="global history length per sequence")
    parser.add_argument("--heads", type=int, choices=(4, 8), default=4,
                        help="query heads per rank; one local KV head")
    parser.add_argument("--dcp-size", type=int, choices=(2, 4, 8),
                        help="defaults to the process count")
    parser.add_argument("--comm-sms", type=positive_int, default=4)
    parser.add_argument("--num-splits", type=int, default=0, help="0 selects automatic splits")
    parser.add_argument("--mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=positive_int, default=50)
    args = parser.parse_args()
    if args.warmup < 0 or not 0 <= args.num_splits <= 128:
        parser.error("warmup must be nonnegative and num-splits must be in [0, 128]")

    import torch
    import torch.distributed as dist
    from megacp import DCPMegaAttentionRunner

    device, rank, world = initialize()
    try:
        dcp_size = args.dcp_size or world
        if world % dcp_size or args.history_length < dcp_size:
            parser.error("DCP size must divide the process count and not exceed history length")
        group = None
        for first in range(0, world, dcp_size):
            candidate = dist.new_group(list(range(first, first + dcp_size)), backend="nccl", device_id=device)
            if first <= rank < first + dcp_size:
                group = candidate
        dcp_rank = dist.get_rank(group)
        local_history = (args.history_length + dcp_size - 1 - dcp_rank) // dcp_size
        total_q = args.batch_size * args.query_length
        cu_q_host = torch.arange(args.batch_size + 1, dtype=torch.int32) * args.query_length
        cu_h_host = torch.arange(args.batch_size + 1, dtype=torch.int32) * local_history
        cu_q, cu_h = cu_q_host.to(device), cu_h_host.to(device)
        k = torch.randn((args.batch_size * local_history, 1, 128), device=device, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        ck = torch.randn((total_q, 1, 128), device=device, dtype=torch.bfloat16)
        cv = torch.randn_like(ck)
        with DCPMegaAttentionRunner(
            group, dist.group.WORLD, max_total_q=total_q, max_batch=args.batch_size,
            Hq_local=args.heads, num_comm_sm=args.comm_sms,
        ) as runner:
            q = runner.q_local(total_q)
            q.normal_()
            runner.forward_chunk_prefill_varlen(
                q, k, v, ck, cv, cu_q, cu_h, args.query_length, local_history,
                cu_seqlens_q_host=cu_q_host, cu_seqlens_history_local_host=cu_h_host,
                num_splits=args.num_splits, return_lse=True,
            )
            graph = None
            try:
                if args.mode == "graph":
                    graph = runner.capture_last_forward(capture_warmup=3)
                    sample = event_sampler(graph.replay)
                else:
                    def sample():
                        runner.prepare_last_forward_replay()
                        _, elapsed = runner.replay_last_forward(return_timing_ms=True)
                        return elapsed
                timing = measure(sample, args.warmup, args.iters, device)
                if rank == 0:
                    print(json.dumps({
                        "method": "megadcp", "world_size": world, "dcp_size": dcp_size,
                        "batch_size": args.batch_size, "query_length": args.query_length,
                        "history_length": args.history_length, "heads_per_rank": args.heads,
                        "comm_sms": args.comm_sms,
                        "timing": "kernel_only" if args.mode == "eager" else "graph_replay",
                        "block_n": runner.last_dispatch.block_n,
                        "split_plan_rank0": runner.last_queue_counts["sequence_splits"],
                        **timing,
                    }, indent=2))
            finally:
                if graph is not None:
                    graph.close()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
