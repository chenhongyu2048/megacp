"""Benchmark causal MegaRing forward or backward using preallocated workspaces."""

import argparse
import json

from .common import event_sampler, initialize, measure, positive_int


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--sequence-length", type=positive_int, default=2048,
                        help="tokens per sequence per rank; a multiple of 256")
    parser.add_argument("--query-heads", type=positive_int, default=16)
    parser.add_argument("--kv-heads", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--comm-sms", type=positive_int, default=8)
    parser.add_argument("--compute-sms", type=positive_int, help="defaults to all remaining SMs")
    parser.add_argument("--direction", choices=("forward", "backward"), default="forward")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=positive_int, default=50)
    args = parser.parse_args()
    if args.sequence_length % 256 or args.query_heads % args.kv_heads or args.warmup < 0:
        parser.error("sequence-length must be 256-aligned, query-heads divisible by kv-heads, and warmup nonnegative")

    import torch
    import torch.distributed as dist
    import min_fa3_op
    from megacp import TKParallelTensor, backward_varlen_mega_ring, forward_varlen_mega_ring

    device, rank, world = initialize()
    try:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        compute_sms = args.compute_sms or sms - args.comm_sms
        if compute_sms <= 0 or compute_sms + args.comm_sms > sms:
            parser.error("compute and communication SM counts exceed the device capacity")
        total = args.batch_size * args.sequence_length
        cu_host = torch.arange(args.batch_size + 1, dtype=torch.int32) * args.sequence_length
        cu = cu_host.to(device)
        ring_sizes = torch.full((args.batch_size,), world, dtype=torch.int32)
        ring_starts = torch.zeros(args.batch_size, dtype=torch.int32)
        global_lengths = torch.full((args.batch_size,), args.sequence_length * world, dtype=torch.int32)
        q = torch.randn((total, args.query_heads, 128), device=device, dtype=torch.bfloat16)
        arenas = [TKParallelTensor([world * total, args.kv_heads, 128], torch.bfloat16, rank, world, False)
                  for _ in range(2)]
        k, v = (arena.data_ for arena in arenas)
        for tensor in (k, v):
            tensor.zero_()
            tensor[rank * total:(rank + 1) * total].normal_()
        common = dict(
            cu_seqlens_q_host=cu_host, cu_seqlens_k_host=cu_host,
            remote_k=arenas[0], remote_v=arenas[1], num_comp_sm=compute_sms,
            num_comm_sm=args.comm_sms, global_seqlens_host=global_lengths,
            ring_sizes_host=ring_sizes, ring_starts_host=ring_starts,
        )
        workspace = min_fa3_op._prepare_mega_ring_forward_call_workspace(
            q, cu_host, cu_host, ring_sizes, rank, True,
        )
        torch.cuda.synchronize(device)
        dist.barrier(device_ids=[device.index])
        out, lse = forward_varlen_mega_ring(
            q, k, v, cu, cu, args.sequence_length, args.sequence_length, True,
            **common, workspace=workspace, return_lse=True,
        )
        if args.direction == "forward":
            def operation():
                return forward_varlen_mega_ring(
                    q, k, v, cu, cu, args.sequence_length, args.sequence_length, True,
                    **common, workspace=workspace, out=out, lse=lse, return_lse=True,
                )
        else:
            dout = torch.randn_like(q)
            padded = ((total + args.batch_size * 128 + 127) // 128) * 128
            gradients = [TKParallelTensor([args.kv_heads * padded * 128], torch.float32, rank, world, False)
                         for _ in range(2)]
            completion = TKParallelTensor([1], torch.int32, rank, world, False)
            for arena in (*gradients, completion):
                arena.data_.zero_()
            backward_workspace = min_fa3_op._create_backward_varlen_mega_ring_workspace(
                q, cu_host, cu_host, world, args.kv_heads, total,
            )
            torch.cuda.synchronize(device)
            dist.barrier(device_ids=[device.index])

            def operation():
                return backward_varlen_mega_ring(
                    dout, q, k, v, out, lse, cu, cu, args.sequence_length, args.sequence_length,
                    **common, remote_dk_accum=gradients[0], remote_dv_accum=gradients[1],
                    remote_dkv_completion=completion, workspace=backward_workspace,
                )
        timing = measure(event_sampler(operation), args.warmup, args.iters, device)
        if rank == 0:
            print(json.dumps({
                "method": "megaring", "direction": args.direction, "world_size": world,
                "batch_size": args.batch_size, "sequence_length_per_rank": args.sequence_length,
                "query_heads": args.query_heads, "kv_heads": args.kv_heads,
                "compute_sms": compute_sms, "comm_sms": args.comm_sms,
                "timing": "cuda_events_around_preallocated_call", **timing,
            }, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
